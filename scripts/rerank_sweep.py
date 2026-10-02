"""Step 3: cross-encoder reranking over the dense shortlist -> data/eval/rerank_sweep.json.

The case for it, from the gte-large/v3 numbers: R@100 is 0.83-0.95 while R@10 is
0.56-0.76, so ~20-25 points of gold is already retrieved but ranked too low to reach
the selector. A reranker's whole job is converting that recall into precision; it cannot
exceed R@depth, which is printed as the ceiling on every row.

One scoring pass per (encoder, reranker) gives EVERY depth: reranking the top-d is just
re-sorting the first d of the top-100 dense order and leaving the tail in place, so
depths 10/25/50/100 are derived from the same 100 scores rather than re-scored.

Decide on --partition dev. Confirm the winner once on test.

  ./.venv/bin/python scripts/rerank_sweep.py --partition dev --n 150 --device mps
"""

import argparse, json, time
from pathlib import Path

import numpy as np

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from build_embeddings import tool_text
from eval_retrieval import CORPUS, sidecar, splits

OUT = Path("data/eval/rerank_sweep.json")
DEPTHS = [10, 25, 50, 100]
KS = [1, 5, 10, 25]
SEED = 13


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/eval/tool_embeddings_gtelarge_v3.npy")
    ap.add_argument("--rerankers", nargs="+",
                    default=["BAAI/bge-reranker-base", "BAAI/bge-reranker-large"])
    ap.add_argument("--text", default=None,
                    help="document construction fed to the reranker; defaults to the matrix's")
    ap.add_argument("--partition", default="dev", choices=["all", "dev", "test"])
    ap.add_argument("--n", type=int, default=150, help="queries per split")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    cat = [json.loads(l) for l in (CORPUS / "catalog.jsonl").read_text().splitlines() if l.strip()]
    pos = {t["tool_id"]: i for i, t in enumerate(cat)}
    model, qpre, meta = sidecar(a.matrix, None, None)
    variant = a.text or meta.get("text_variant", "v3")
    docs = [tool_text(t, variant) for t in cat]
    E = np.load(a.matrix)
    if E.shape[0] != len(cat):
        raise SystemExit(f"{a.matrix}: {E.shape[0]} rows != {len(cat)} catalog rows")

    sp = {k: v for k, v in splits(a.partition).items() if not k.startswith("splitE")}
    rng = np.random.default_rng(SEED)
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(model, device=a.device)
    print(f"dense: {model} text={variant} partition={a.partition}", flush=True)

    # --- dense top-100 per split
    shortlists = {}
    for name, items in sp.items():
        if a.n and len(items) > a.n:
            items = [items[i] for i in rng.choice(len(items), a.n, replace=False)]
        items = [it for it in items if any(g in pos for g in it["gold_tool_ids"])]
        Q = enc.encode([qpre + it["query"] for it in items], batch_size=128,
                       normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        order = np.argsort(-(Q @ E.T), axis=1)[:, :100]
        golds = [[pos[g] for g in it["gold_tool_ids"] if g in pos] for it in items]
        shortlists[name] = (items, order, golds)
        print(f"  {name:22s} n={len(items)}", flush=True)
    del enc

    def recalls(orders, golds, ks=KS):
        out = {}
        ranks = []
        for o, g in zip(orders, golds):
            hit = [int(np.where(o == j)[0][0]) + 1 for j in g if j in set(o.tolist())]
            ranks.append(min(hit) if hit else 10 ** 6)
        for k in ks:
            out[f"R@{k}"] = round(float(np.mean([r <= k for r in ranks])), 4)
        out["mrr"] = round(float(np.mean([1.0 / r if r <= 100 else 0.0 for r in ranks])), 4)
        return out

    results = {"matrix": a.matrix, "dense_model": model, "text_variant": variant,
               "partition": a.partition, "splits": {}}
    for name, (items, order, golds) in shortlists.items():
        results["splits"][name] = {"n": len(items), "dense": recalls(order, golds),
                                   "ceiling": {f"R@{d}": recalls(order, golds, [d])[f"R@{d}"]
                                               for d in DEPTHS},
                                   "rerank": {}}

    from sentence_transformers import CrossEncoder

    for rr_name in a.rerankers:
        try:
            ce = CrossEncoder(rr_name, device=a.device, max_length=512)
        except Exception as e:
            print(f"  !! {rr_name} unavailable: {type(e).__name__}: {str(e)[:90]}")
            continue
        for name, (items, order, golds) in shortlists.items():
            t0 = time.time()
            pairs, idx = [], []
            for i, it in enumerate(items):
                for d in order[i]:
                    pairs.append((it["query"], docs[d]))
                    idx.append((i, int(d)))
            scores = np.asarray(ce.predict(pairs, batch_size=a.batch,
                                           show_progress_bar=False), dtype=np.float32)
            smat = np.full(order.shape, -np.inf, dtype=np.float32)
            colof = {}
            for p, (i, d) in enumerate(idx):
                colof.setdefault(i, {})[d] = p
            for i in range(order.shape[0]):
                for c, d in enumerate(order[i]):
                    smat[i, c] = scores[colof[i][int(d)]]

            per_depth = {}
            for depth in DEPTHS:
                new = np.empty_like(order)
                for i in range(order.shape[0]):
                    head = order[i, :depth][np.argsort(-smat[i, :depth], kind="stable")]
                    new[i] = np.concatenate([head, order[i, depth:]])
                per_depth[f"depth_{depth}"] = recalls(new, golds)
            el = time.time() - t0
            per_depth["_pairs"] = len(pairs)
            per_depth["_sec"] = round(el, 1)
            per_depth["_ms_per_query"] = round(1000 * el / max(len(items), 1), 1)
            results["splits"][name]["rerank"][rr_name] = per_depth
            print(f"  {rr_name.split('/')[-1]:20s} {name:22s} "
                  f"{len(pairs)} pairs in {el:.0f}s ({per_depth['_ms_per_query']:.0f} ms/query)",
                  flush=True)
        del ce

    # --- report
    keys = [f"R@{k}" for k in KS]
    for name, r in results["splits"].items():
        print(f"\n=== {name}  (n={r['n']}, dense ceiling R@100={r['ceiling']['R@100']:.3f})")
        print(f"{'config':34s} " + " ".join(f"{k:>7s}" for k in keys) + f" {'MRR':>7s}")
        print(f"{'dense only':34s} " + " ".join(f"{r['dense'][k]:7.3f}" for k in keys)
              + f" {r['dense']['mrr']:7.3f}")
        for rr, per in r["rerank"].items():
            for depth in DEPTHS:
                d = per[f"depth_{depth}"]
                lab = f"{rr.split('/')[-1]} top-{depth}"
                delta = d["R@10"] - r["dense"]["R@10"]
                print(f"{lab:34s} " + " ".join(f"{d[k]:7.3f}" for k in keys)
                      + f" {d['mrr']:7.3f}   R@10 {delta:+.3f}")

    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
