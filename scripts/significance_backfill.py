"""Back-fill paired significance tests onto §7 and §8 -> data/eval/significance_backfill.json

DATA.md §9 item 5: `rerank_sweep.py` and `fusion_sweep.py` persist aggregates only, so
§7 and §8 report point estimates at n=150/split with no significance tests -- despite
every comparison being paired on identical queries, which is the easy case. §10 is the
argument for fixing it: there, the paired tests denied a result the point estimate had
passed.

This needs NO recompute. `fusion_scores_dev_n150.npz` already holds the per-query x
per-candidate score matrices (isgold, dense, and both cross-encoders over the dense
top-100), which is strictly more information than the aggregates. Re-deriving per-item
gold ranks from it costs seconds.

It covers §7 as well as §8, because §7's "replacement" is the alpha=1 corner of §8's
fusion: at alpha=1 the zscore rule orders by z(rerank) alone, and z is a monotone affine
map, so the ordering is identical to sorting by raw cross-encoder score inside the
window -- exactly what rerank_sweep.py did. Verified against rerank_sweep.json below.

Tests, both paired on identical queries:
  - McNemar exact (two-sided binomial on discordant pairs) for R@1. The right test for
    paired binary outcomes: only queries whose hit/miss flips carry information.
  - Bootstrap CI over items for the R@1 and MRR deltas.

  ./.venv/bin/python scripts/significance_backfill.py
"""

import argparse, json
from pathlib import Path

import numpy as np

from bm25_fusion import paired_tests
from fusion_sweep import DEPTHS, recalls, reorder

CACHE = Path("data/eval/fusion_scores_dev_n150.npz")
PRIOR_RERANK = Path("data/eval/rerank_sweep.json")
PRIOR_FUSION = Path("data/eval/fusion_sweep.json")
OUT = Path("data/eval/significance_backfill.json")
RANKS = Path("data/eval/significance_backfill_ranks.npz")

HEADLINE = "BAAI/bge-reranker-large"
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def ranks_from_isgold(isgold):
    """1-based rank of the first gold in the CURRENT ordering; 10**6 if absent.

    Same convention as fusion_sweep.recalls, which is what makes the derived aggregates
    reproduce §7/§8 exactly rather than approximately.
    """
    n = isgold.shape[0]
    r = np.full(n, 10**6, dtype=np.int64)
    hit = isgold.any(axis=1)
    r[hit] = isgold[hit].argmax(axis=1) + 1
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--model", default=HEADLINE)
    ap.add_argument("--rules", nargs="+", default=["zscore", "rrf"])
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    if not Path(a.cache).exists():
        raise SystemExit(f"{a.cache} missing -- run fusion_sweep.py first (it caches the "
                         f"cross-encoder scores; this script only re-reads them)")
    d = np.load(a.cache, allow_pickle=False)
    names = sorted({k.rsplit("|", 1)[0] for k in d.files})
    store = {n: {k.rsplit("|", 1)[1]: d[k] for k in d.files if k.startswith(n + "|")}
             for n in names}
    meta = json.loads(Path(a.cache).with_suffix(".meta.json").read_text())
    print(f"loaded {a.cache}: {len(store)} splits, model={meta['dense_model']}, "
          f"n={meta['n']}, seed={meta['seed']}")

    # --- guardrail: derived aggregates must reproduce the persisted ones, else the ranks
    # are not the ranks §7/§8 reported on and no test built from them means anything.
    pf = json.loads(PRIOR_FUSION.read_text())["splits"] if PRIOR_FUSION.exists() else {}
    pr = json.loads(PRIOR_RERANK.read_text())["splits"] if PRIOR_RERANK.exists() else {}
    n_chk = 0
    for name, s in store.items():
        got = recalls(s["isgold"])
        if name in pf:
            want = pf[name]["dense"]
            bad = {k: (got[k], want[k]) for k in got if k in want and abs(got[k] - want[k]) > 1e-9}
            if bad:
                raise SystemExit(f"dense MISMATCH vs fusion_sweep.json {name}: {bad}")
            n_chk += 1
        # §7 replacement == alpha=1 fusion under zscore. rerank_sweep stores it as
        # splits[name]["rerank"][model]["depth_<d>"]; compare where both exist. This is
        # the load-bearing check: it proves the alpha=1 corner of §8's harness really is
        # the thing §7 measured, so these tests apply to §7's claim and not just §8's.
        blocks = pr.get(name, {}).get("rerank", {}).get(a.model, {})
        for depth in DEPTHS:
            want = blocks.get(f"depth_{depth}")
            if not want:
                continue
            mine = recalls(reorder(s["isgold"], s["dense"], s[a.model], 1.0, "zscore", depth))
            bad = {k: (mine[k], want[k]) for k in mine
                   if k in want and abs(mine[k] - want[k]) > 1e-9}
            if bad:
                raise SystemExit(f"replacement MISMATCH vs rerank_sweep.json "
                                 f"{name} depth_{depth}: {bad}")
            n_chk += 1
    print(f"guardrail ok: {n_chk} derived aggregate blocks reproduce the persisted ones")

    gen = [n for n in store if n.startswith(("standard", "splitD"))]
    ranks_out, results = {}, {"cache": a.cache, "model": a.model, **meta, "splits": {}}

    for name, s in store.items():
        dense_r = ranks_from_isgold(s["isgold"])
        ranks_out[f"{name}|dense"] = dense_r
        rec = {"n": int(len(dense_r)), "dense": recalls(s["isgold"]), "configs": {}}
        for rule in a.rules:
            for depth in DEPTHS:
                for al in ALPHAS:
                    key = f"{rule}|depth{depth}|a{al:g}"
                    rr = ranks_from_isgold(
                        reorder(s["isgold"], s["dense"], s[a.model], al, rule, depth))
                    ranks_out[f"{name}|{key}"] = rr
                    rec["configs"][key] = rr
        results["splits"][name] = rec

    # --- the two headline claims, tested
    CLAIMS = {
        "§8 fusion (zscore depth100 a0.2)": "zscore|depth100|a0.2",
        "§8 fusion (rrf depth100 a0.2)": "rrf|depth100|a0.2",
        "§7 replacement (zscore depth100 a1)": "zscore|depth100|a1",
        "§7 replacement (zscore depth10 a1)": "zscore|depth10|a1",
    }
    tested = {}
    for label, key in CLAIMS.items():
        per_split, pooled_b, pooled_n = {}, [], []
        for name in store:
            base = ranks_out[f"{name}|dense"]
            new = results["splits"][name]["configs"][key]
            per_split[name] = paired_tests(base, new, 1, a.boot)
            if name in gen:
                pooled_b.append(base)
                pooled_n.append(new)
        per_split["POOLED_generated"] = paired_tests(
            np.concatenate(pooled_b), np.concatenate(pooled_n), 1, a.boot)
        tested[label] = per_split
    results["tested_claims"] = tested

    # serialise ranks as lists in the JSON-free path: keep JSON to aggregates + tests
    for name in results["splits"]:
        results["splits"][name]["configs"] = {
            k: {"R@1": round(float((v <= 1).mean()), 4),
                "R@10": round(float((v <= 10).mean()), 4),
                "mrr": round(float(np.where(v <= 100, 1.0 / v, 0.0).mean()), 4)}
            for k, v in results["splits"][name]["configs"].items()}

    np.savez_compressed(RANKS, **ranks_out)

    # ---------------- report ----------------
    for label, per_split in tested.items():
        print("\n" + "=" * 94)
        print(f"{label}  vs dense -- paired on identical queries")
        print("=" * 94)
        print(f"{'split':22s}{'dR@1':>8s}{'R@1 95% CI':>19s}{'McNemar b/c':>14s}{'p':>9s}"
              f"{'dMRR':>8s}{'MRR 95% CI':>19s}")
        for name, t in per_split.items():
            ci = f"[{t['R@1_ci95'][0]:+.3f},{t['R@1_ci95'][1]:+.3f}]"
            mci = f"[{t['mrr_ci95'][0]:+.3f},{t['mrr_ci95'][1]:+.3f}]"
            bc = f"{t['mcnemar_b_c'][0]}/{t['mcnemar_b_c'][1]}"
            star = " *" if t["mcnemar_p"] < 0.05 else ""
            print(f"{name:22s}{t['d_R@1']:+8.3f}{ci:>19s}{bc:>14s}"
                  f"{t['mcnemar_p']:9.4f}{t['d_mrr']:+8.3f}{mci:>19s}{star}")

    print("\n" + "=" * 94)
    print("ALPHA PLATEAU, pooled generated splits, zscore/depth100: is the argmax")
    print("distinguishable from its neighbours, or is §8's alpha=0.2 within noise?")
    print("=" * 94)
    print(f"{'alpha':>7s}{'dR@1':>9s}{'R@1 95% CI':>20s}{'dMRR':>9s}{'MRR 95% CI':>20s}"
          f"{'p(MRR<=0)':>11s}")
    base = np.concatenate([ranks_out[f"{n}|dense"] for n in gen])
    for al in ALPHAS[1:]:
        new = np.concatenate([ranks_out[f"{n}|zscore|depth100|a{al:g}"] for n in gen])
        t = paired_tests(base, new, 1, a.boot)
        ci = f"[{t['R@1_ci95'][0]:+.3f},{t['R@1_ci95'][1]:+.3f}]"
        mci = f"[{t['mrr_ci95'][0]:+.3f},{t['mrr_ci95'][1]:+.3f}]"
        print(f"{al:7.1f}{t['d_R@1']:+9.3f}{ci:>20s}{t['d_mrr']:+9.3f}{mci:>20s}"
              f"{t['boot_p_mrr_gt0']:11.4f}")

    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {a.out}")
    print(f"wrote {RANKS}  ({len(ranks_out)} per-item rank vectors)")


if __name__ == "__main__":
    main()
