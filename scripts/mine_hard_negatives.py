"""Mine hard negatives for retriever fine-tuning -> data/corpus/train_triplets.jsonl

Why mined negatives rather than in-batch ones: MultipleNegativesRankingLoss only learns
from what sits in its softmax denominator. Random in-batch negatives are mostly already
far away and contribute near-zero gradient. The errors that matter here are near-misses
(§4: 844 APIs.guru pairs above 0.85 similarity), so the denominator has to contain the
top-ranked WRONG tools from the full 3,551 catalog -- that is what makes the training
distribution resemble the evaluation distribution.

Two filters, both load-bearing:

  * drop candidates whose cosine to the GOLD document exceeds --max-sim. Those are the
    `separable: false` artifacts of §4b -- wrapper-vs-core endpoints, same operation
    differing only by HTTP verb. Training the model to separate them teaches a
    distinction that does not exist and that §4d confirmed the selector also cannot make.
  * skip the gold itself (obviously) and any other gold for the same query.

KNOWN ASYMMETRY, recorded because it is the main risk of the whole fine-tune: train.jsonl
has gold tools from only 160 MetaTool tools, and MetaTool contains ZERO pairs above 0.85
similarity (§4), so every hard negative necessarily comes from APIs.guru. The training
signal therefore says "APIs.guru documents are the wrong answer" while evaluation on the
generated splits says they are the right one. If any of that lands unconditionally on the
document tower, transfer goes negative rather than merely flat. Measured, not assumed:
see scripts/finetune_retriever.py --probe.

  ./.venv/bin/python scripts/mine_hard_negatives.py --device mps
"""

import argparse, json
from pathlib import Path

import numpy as np

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from build_embeddings import tool_text
from eval_retrieval import CORPUS, sidecar

OUT = CORPUS / "train_triplets.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/eval/tool_embeddings_gtebase_v3.npy",
                    help="matrix to mine with; use the SAME base model you will train")
    ap.add_argument("--split", default="train.jsonl")
    ap.add_argument("--n-neg", type=int, default=4)
    ap.add_argument("--depth", type=int, default=50, help="rank depth to draw negatives from")
    ap.add_argument("--max-sim", type=float, default=0.95,
                    help="drop negatives this similar to the gold (inseparable twins)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    cat = [json.loads(l) for l in (CORPUS / "catalog.jsonl").read_text().splitlines() if l.strip()]
    pos = {t["tool_id"]: i for i, t in enumerate(cat)}
    model, qpre, meta = sidecar(a.matrix, None, None)
    variant = meta.get("text_variant", "v3")
    docs = [tool_text(t, variant) for t in cat]
    E = np.load(a.matrix)
    if E.shape[0] != len(cat):
        raise SystemExit(f"{a.matrix}: {E.shape[0]} rows != {len(cat)} catalog rows")

    items = [json.loads(l) for l in (CORPUS / a.split).read_text().splitlines() if l.strip()]
    items = [it for it in items if any(g in pos for g in it["gold_tool_ids"])]
    print(f"{len(items)} labeled pairs; mining with {model} text={variant}", flush=True)

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(model, device=a.device)
    Q = enc.encode([qpre + it["query"] for it in items], batch_size=128,
                   normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    del enc

    n_rows, dropped_sim, short = 0, 0, 0
    src_counts = {}
    with open(a.out, "w") as fh:
        for i, it in enumerate(items):
            golds = [pos[g] for g in it["gold_tool_ids"] if g in pos]
            gset = set(golds)
            sims = Q[i] @ E.T
            cand = np.argsort(-sims)[: a.depth]
            gold_vecs = E[golds]
            negs = []
            for c in cand:
                c = int(c)
                if c in gset:
                    continue
                if float(np.max(gold_vecs @ E[c])) > a.max_sim:
                    dropped_sim += 1
                    continue
                negs.append(c)
                if len(negs) >= a.n_neg:
                    break
            if len(negs) < a.n_neg:
                short += 1
                if not negs:
                    continue
            row = {"anchor": it["query"], "positive": docs[golds[0]]}
            for j, c in enumerate(negs, 1):
                row[f"negative_{j}"] = docs[c]
                src_counts[cat[c]["source"]] = src_counts.get(cat[c]["source"], 0) + 1
            # pad short rows by repeating the last negative so every row has the same
            # columns -- datasets requires a uniform schema
            for j in range(len(negs) + 1, a.n_neg + 1):
                row[f"negative_{j}"] = docs[negs[-1]]
            fh.write(json.dumps(row) + "\n")
            n_rows += 1

    print(f"wrote {a.out}: {n_rows} rows x {a.n_neg} negatives")
    print(f"  dropped {dropped_sim} candidates above cosine {a.max_sim} to gold "
          f"(inseparable twins)")
    print(f"  {short} rows had fewer than {a.n_neg} usable negatives (padded)")
    tot = sum(src_counts.values())
    print(f"  negative provenance: " +
          ", ".join(f"{k} {v} ({v/tot:.1%})" for k, v in sorted(src_counts.items())))
    print("  ^ the asymmetry in the module docstring: this is the population the model "
          "is taught to reject.")


if __name__ == "__main__":
    main()
