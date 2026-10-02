"""Compare several tool-embedding matrices at once -> data/eval/encoder_sweep.json.

Answers the question DATA.md 6 left open: the bge-base + v3 change lifted every
generated split and DEPRESSED both MetaTool splits. Is that specific to bge, or general
to retrieval-tuned encoders (i.e. a property of the two catalogs, not the model)?

So the summary column that matters is not mean R@10 -- it is the SIGN SPLIT between the
generated splits (APIs.guru operations, Claude-phrased intents) and the MetaTool anchor
(plugin catalog, GPT-4-phrased intents). If every encoder shows the same sign split, the
divergence is about the catalogs. If only bge does, it is about bge.

SELECTION HYGIENE: the encoder decision is made on the two already-burned dev splits
(standard_L2, splitD_separable). Clean splits are reported for context but must not be
used to pick the winner -- see --decide-on.

  ./.venv/bin/python scripts/encoder_sweep.py --matrices data/eval/tool_embeddings*.npy
"""

import argparse, glob, json
from pathlib import Path

import numpy as np

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from eval_retrieval import CORPUS, DEV_SPLITS, evaluate, sidecar, splits

OUT = Path("data/eval/encoder_sweep.json")
GEN_SPLITS = ["standard_L2", "standard_L3", "standard_L4",
              "splitD_separable", "splitD_inseparable"]
ANCHOR_SPLITS = ["test_A", "C_unseen_tools"]


def label(meta, path):
    m = (meta.get("model") or Path(path).stem).split("/")[-1]
    return f"{m}/{meta.get('text_variant', '?')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", nargs="+", required=True,
                    help="paths or globs; each needs a .meta.json sidecar")
    ap.add_argument("--baseline", default="data/eval/tool_embeddings.npy",
                    help="matrix all deltas are taken against (the shipped MiniLM v0)")
    ap.add_argument("--baseline-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=10, help="which R@k to tabulate")
    ap.add_argument("--n", type=int, default=1200, help="cap per split")
    ap.add_argument("--device", default=None)
    ap.add_argument("--decide-on", nargs="+", default=sorted(DEV_SPLITS),
                    help="splits the winner may be chosen on (default: the dev splits)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    paths = []
    for p in a.matrices:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])
    paths = [p for p in dict.fromkeys(paths) if Path(p).resolve() != Path(a.baseline).resolve()]

    cat = [json.loads(l) for l in (CORPUS / "catalog.jsonl").read_text().splitlines() if l.strip()]
    pos = {t["tool_id"]: i for i, t in enumerate(cat)}
    sp = splits()
    order = [s for s in ANCHOR_SPLITS + GEN_SPLITS if s in sp]

    runs = {}
    E = np.load(a.baseline)
    runs["MiniLM-L6-v2/v0 (base)"] = evaluate(E, a.baseline_model, "", sp, pos, cat, a.n, a.device)
    print(f"base: {a.baseline}", flush=True)

    for p in paths:
        try:
            E = np.load(p)
            if E.shape[0] != len(cat):
                print(f"  !! {p}: {E.shape[0]} rows != {len(cat)} catalog rows, skipping")
                continue
            model, qpre, meta = sidecar(p, None, None)
            name = label(meta, p)
            print(f"  {name:28s} <- {p}", flush=True)
            runs[name] = evaluate(E, model, qpre, sp, pos, cat, a.n, a.device)
        except Exception as e:
            print(f"  !! {p} failed: {type(e).__name__}: {str(e)[:120]}")

    kk = f"R@{a.k}"
    base = runs["MiniLM-L6-v2/v0 (base)"]

    hdr = f"{'config':30s} " + " ".join(f"{s[:13]:>14s}" for s in order)
    print("\n" + "=" * len(hdr))
    print(f"{kk} (n<={a.n}/split). [A]=MetaTool anchor, [G]=generated primary set")
    print(f"{'':30s} " + " ".join(f"{'[A]' if s in ANCHOR_SPLITS else '[G]':>14s}" for s in order))
    print(hdr)
    print("-" * len(hdr))
    for name, res in runs.items():
        cells = []
        for s in order:
            r = res[s]
            cells.append("           n/a" if r.get("abstain") else f"{r[kk]:14.3f}")
        print(f"{name:30s} " + " ".join(cells))
        if name != "MiniLM-L6-v2/v0 (base)":
            d = []
            for s in order:
                if res[s].get("abstain"):
                    d.append("           n/a")
                else:
                    d.append(f"{res[s][kk] - base[s][kk]:+14.3f}")
            print(f"{'  Δ vs base':30s} " + " ".join(d))

    print("\n" + "=" * 78)
    print("SIGN SPLIT: mean Δ" + kk + " on the anchor vs on the generated set")
    print(f"{'config':30s} {'anchor Δ':>12s} {'generated Δ':>14s} {'diverges?':>12s}")
    print("-" * 78)
    summary = {}
    for name, res in runs.items():
        if name == "MiniLM-L6-v2/v0 (base)":
            continue
        ad = [res[s][kk] - base[s][kk] for s in ANCHOR_SPLITS if s in res]
        gd = [res[s][kk] - base[s][kk] for s in GEN_SPLITS if s in res]
        am, gm = float(np.mean(ad)), float(np.mean(gd))
        div = "YES" if am < 0 < gm else "no"
        summary[name] = {"anchor_mean_delta": round(am, 4),
                         "generated_mean_delta": round(gm, 4), "diverges": div == "YES"}
        print(f"{name:30s} {am:+12.3f} {gm:+14.3f} {div:>12s}")

    print("\nDecision splits (" + ", ".join(a.decide_on) + f") -- ranked by {kk}:")
    rank = sorted(runs.items(),
                  key=lambda kv: -float(np.mean([kv[1][s][kk] for s in a.decide_on if s in kv[1]])))
    for name, res in rank:
        m = float(np.mean([res[s][kk] for s in a.decide_on if s in res]))
        print(f"  {m:.3f}  {name}")

    Path(a.out).write_text(json.dumps(
        {"k": a.k, "n_cap": a.n, "runs": runs, "sign_split": summary,
         "decide_on": a.decide_on}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
