"""Embed the unified catalog -> data/eval/tool_embeddings.npy.

CONTRACT: row i of the matrix is line i of data/corpus/catalog.jsonl. Both
generate_intents.py (--mode confusion) and roundtrip_check.py index into it by that
position. Rebuild this whenever the catalog changes or those scripts read garbage.
"""

import argparse, hashlib, json, re
from pathlib import Path

import numpy as np

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

CORPUS, OUT = Path("data/corpus/catalog.jsonl"), Path("data/eval/tool_embeddings.npy")

# Retrieval-tuned encoders expect instruction prefixes, and they are ASYMMETRIC: BGE
# prefixes the query only, E5 prefixes BOTH sides with different strings. Scoring E5
# without "passage: " on the documents measures a handicap, not the encoder. Both are
# recorded in the sidecar so consumers embed queries the same way the matrix was built.
QUERY_PREFIX = {
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-large-v2": "query: ",
    "intfloat/e5-base-v2": "query: ",
}
DOC_PREFIX = {
    "intfloat/e5-large-v2": "passage: ",
    "intfloat/e5-base-v2": "passage: ",
}


def humanize(s):
    """'/v1/timezone/{area}/{location}/{region}.txt' -> 'v1 timezone area location region txt'

    OpenAPI paths carry the discriminating signal for near-duplicate operations (the
    worldtimeapi region pair differs ONLY by path depth), so it has to be tokenized,
    not dropped. Strips braces, splits camelCase, keeps alphanumerics.
    """
    s = re.sub(r"\{[^}]*\}", " ", s or "")
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return " ".join(re.findall(r"[A-Za-z0-9]+", s))


def tool_text(t, variant="v0"):
    """Document construction for the tool index. Measured R@10 on standard_L2 (MiniLM):
    v0 0.473 -> v1 0.486 -> v2 0.526 -> v3 0.525; on splitD_sep 0.544 -> 0.613. The
    thin-description subgroup (gold desc <= 5 words) is where it pays: 0.348 -> 0.436.

      v0  name + summary + description + param names   <- the SHIPPED matrix
      v1  + tags                                       <- matches retrieval_baseline.py
      v2  + api_title + provider (service context)
      v3  + HTTP method + humanized path               <- default for new matrices

    KNOWN INCONSISTENCY (historical): the shipped tool_embeddings.npy -- the one split D
    was actually generated from -- is v0, while retrieval_baseline.py computes its own
    embeddings at v1. 2,868/3,551 tools have tags, so most rows differ (mean cosine
    0.986) and confusion-pair selection overlaps only 93%. Split D's sim>=0.82 pair
    selection is therefore defined under MiniLM-v0 geometry; do not overwrite that
    artifact, write new matrices to new filenames.
    """
    parts = [t["name"], t.get("summary", ""), t["description"]]
    p = t.get("parameters", {}).get("properties", {})
    if p:
        parts.append(" ".join(list(p)[:15]))
    if variant in ("v1", "v2", "v3") and t.get("tags"):
        parts.append(" ".join(t["tags"]))
    if variant in ("v2", "v3"):
        parts += [t.get("api_title", ""), t.get("provider", "")]
    if variant == "v3":
        h = t.get("http") or {}
        if isinstance(h, dict):
            parts += [(h.get("method") or "").upper(), humanize(h.get("path", ""))]
    return " ".join(x for x in parts if x).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--catalog", default=str(CORPUS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--text", default="v0", choices=["v0", "v1", "v2", "v3"],
                    help="document construction (see tool_text); v0 reproduces the shipped matrix")
    ap.add_argument("--tags", action="store_true", help="deprecated alias for --text v1")
    ap.add_argument("--device", default=None, help="e.g. mps, cuda, cpu")
    a = ap.parse_args()

    variant = "v1" if a.tags and a.text == "v0" else a.text
    if Path(a.out) == OUT and variant != "v0":
        ap.error(f"refusing to overwrite the shipped v0 matrix {OUT} with {variant}; "
                 f"split D indexes it positionally -- pass --out data/eval/tool_embeddings_<tag>.npy")

    cat = [json.loads(l) for l in Path(a.catalog).read_text().splitlines() if l.strip()]
    dpre = DOC_PREFIX.get(a.model, "")
    texts = [dpre + tool_text(t, variant) for t in cat]
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(a.model, device=a.device)
    E = m.encode(texts, batch_size=128, normalize_embeddings=True,
                 show_progress_bar=False).astype(np.float32)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(a.out, E)

    # Sidecar: the matrix is row-aligned to the catalog and meaningless without the
    # model + text variant + query prefix that built it. Pin all four (§15 versioning).
    meta = {
        "matrix": a.out,
        "model": a.model,
        "text_variant": variant,
        "query_prefix": QUERY_PREFIX.get(a.model, ""),
        "doc_prefix": dpre,
        "dim": int(E.shape[1]),
        "rows": int(E.shape[0]),
        "catalog": a.catalog,
        "catalog_sha256": hashlib.sha256(Path(a.catalog).read_bytes()).hexdigest()[:16],
        "mean_doc_words": round(sum(len(x.split()) for x in texts) / len(texts), 1),
    }
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{a.out}: {E.shape} from {len(cat)} tools "
          f"({a.model}, text={variant}, mean {meta['mean_doc_words']} words/doc)")


if __name__ == "__main__":
    main()
