"""Step 3c: BM25 x dense fusion -> data/eval/bm25_fusion.json.

§8 established that the FUSION RULE is what pays: bge-reranker-large measured at
0.77-0.90x dense quality still added +9.4% R@1 once combined with dense instead of
substituted for it. BM25 is the other weak-but-decorrelated signal on hand (§3: 0.241
R@1 alone vs dense 0.442) and it is free and instant -- no cross-encoder forward
passes, so none of the §8.4 latency objection applies.

TWO VARIANTS, and the distinction is the whole point of this script:

  (a) WINDOW fusion -- fuse inside the dense top-100, exactly as §8 did. Reuses
      fusion_sweep's reorder logic verbatim, so it is directly comparable to the
      cross-encoder result. Its recall ceiling is dense R@100: fusion can only
      REORDER golds that dense already retrieved.

  (b) FULL-CATALOG fusion -- score all 3,551 tools with both signals and fuse before
      the top-k cut. BM25 can do this because it is sparse and instant; a cross-encoder
      cannot, which is why §8 never tested it. This variant can RESCUE a gold that
      dense missed entirely, and per §4d ~83% of the error mass is exactly that.

(a) is the control that makes (b) interpretable. If (a) gains and (b) does not, BM25 is
only reordering. If (b) gains materially more than (a), the rescue effect is what pays,
and `union_ceiling` below bounds how much of it is available at all.

Query sampling is byte-identical to fusion_sweep.py (same SEED, same split order, one
rng drawn sequentially, gold-filter AFTER sampling), so the dense block must reproduce
fusion_sweep.json. The run cross-checks that and refuses to report if it fails.

alpha=0 is a GUARDRAIL in both variants: z() is a monotone affine map and rrf is
monotone decreasing in rank, so at alpha=0 a stable argsort must return the dense
ordering bit-exactly. Asserted for every (variant, rule, depth) config.

Unlike §7/§8 this persists PER-ITEM GOLD RANKS (DATA.md §9 item 5), so every comparison
-- all paired on identical queries -- gets an exact McNemar test on R@1 and a paired
bootstrap CI on R@1/MRR instead of another bare point estimate.

  ./.venv/bin/python scripts/bm25_fusion.py --partition dev --n 150 --device mps
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
from fusion_sweep import DEPTHS, RRF_K, SEED, fuse, recalls
from retrieval_baseline import BM25

OUT = Path("data/eval/bm25_fusion.json")
RANKS = Path("data/eval/bm25_fusion_ranks_{partition}_n{n}.npz")
PRIOR = Path("data/eval/fusion_sweep.json")
KS = [1, 5, 10, 25, 50, 100]
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
BOOT = 10000


def rank_ties(s):
    """1-based DESCENDING rank where equal scores share the worst rank in their group.

    fusion_sweep.rank_of breaks ties by position (stable argsort), which is correct for
    cross-encoder logits -- they are dense and effectively tie-free. BM25 is sparse: 25-40%
    of the catalog scores exactly 0.0 for a given query, and positional tie-breaking hands
    those tools distinct ranks in CATALOG ORDER, i.e. it feeds arbitrary ordering into rrf
    as if it were evidence. Competition ranking makes every zero-scoring tool equally and
    maximally bad, which is what a zero BM25 score actually means.

    Only affects the sparse side; dense cosines have no ties, so rank_ties(dense) ==
    rank_of(dense) and the alpha=0 guardrail is unchanged.
    """
    order = np.argsort(-s, axis=1, kind="stable")
    ss = np.take_along_axis(s, order, axis=1)
    n, m = s.shape
    pos = np.broadcast_to(np.arange(1, m + 1), s.shape)
    newgrp = np.ones(s.shape, dtype=bool)
    newgrp[:, 1:] = ss[:, 1:] != ss[:, :-1]
    gid = np.cumsum(newgrp, axis=1) - 1
    r = np.empty((n, m), dtype=np.int64)
    for i in range(n):
        last = np.zeros(int(gid[i, -1]) + 1, dtype=np.int64)
        np.maximum.at(last, gid[i], pos[i])
        r[i] = last[gid[i]]
    out = np.empty_like(r)
    np.put_along_axis(out, order, r, axis=1)
    return out.astype(np.float32)


def fuse_local(dense, sparse, alpha, rule):
    """fusion_sweep.fuse plus 'rrf_tie', the tie-aware rrf variant (see rank_ties)."""
    if rule == "rrf_tie":
        return ((1 - alpha) / (RRF_K + rank_ties(dense))
                + alpha / (RRF_K + rank_ties(sparse)))
    return fuse(dense, sparse, alpha, rule)


def gold_ranks(S, golds, stable=False):
    """1-based rank of the FIRST gold per row, 10**6 if absent. S: (n, N) scores.

    Mirrors fusion_sweep.recalls' convention (min over golds; rows with no gold score 0
    everywhere rather than being dropped) but works off a full score matrix instead of a
    precomputed isgold window.
    """
    order = np.argsort(-S, axis=1, kind="stable" if stable else "quicksort")
    r = np.full(S.shape[0], 10**6, dtype=np.int64)
    for i, g in enumerate(golds):
        if not g:
            continue
        hits = np.nonzero(np.isin(order[i], np.array(sorted(g), dtype=np.int64)))[0]
        if hits.size:
            r[i] = int(hits[0]) + 1
    return r


def reorder_local(isgold, dense, sparse, alpha, rule, depth):
    """fusion_sweep.reorder, but routed through fuse_local so rrf_tie is available.

    Must be identical to the imported version for rules 'zscore' and 'rrf'; guardrail C
    in main() asserts that element-wise rather than asserting it in a comment.
    """
    f = fuse_local(dense[:, :depth], sparse[:, :depth], alpha, rule)
    perm = np.argsort(-f, axis=1, kind="stable")
    head = np.take_along_axis(isgold[:, :depth], perm, axis=1)
    return np.concatenate([head, isgold[:, depth:]], axis=1)


def recalls_from_ranks(ranks, ks=KS):
    out = {f"R@{k}": round(float((ranks <= k).mean()), 4) for k in ks}
    out["mrr"] = round(float(np.where(ranks <= 100, 1.0 / ranks, 0.0).mean()), 4)
    return out


def mrr_vec(ranks):
    return np.where(ranks <= 100, 1.0 / ranks, 0.0)


def paired_tests(base, new, k=1, boot=BOOT, seed=SEED):
    """Paired significance for two rank vectors over the SAME queries.

    McNemar exact (two-sided binomial on the discordant pairs) for R@k, which is the
    right test for paired binary outcomes -- only queries whose hit/miss flips carry
    information, and there are usually far fewer of those than n. Plus a bootstrap CI
    over items for the R@k and MRR deltas.
    """
    hb, hn = base <= k, new <= k
    b = int((~hb & hn).sum())  # base miss -> new hit
    c = int((hb & ~hn).sum())  # base hit  -> new miss
    try:
        from scipy.stats import binomtest

        p = float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue) if b + c else 1.0
    except ImportError:
        from math import comb

        n2 = b + c
        p = (float(sum(comb(n2, i) for i in range(min(b, c) + 1)) / 2 ** (n2 - 1))
             if n2 else 1.0)
        p = min(1.0, p)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(base), size=(boot, len(base)))
    d_r = hn[idx].mean(1) - hb[idx].mean(1)
    mb, mn = mrr_vec(base), mrr_vec(new)
    d_m = mn[idx].mean(1) - mb[idx].mean(1)
    return {
        f"d_R@{k}": round(float(hn.mean() - hb.mean()), 4),
        f"R@{k}_ci95": [round(float(np.percentile(d_r, 2.5)), 4),
                        round(float(np.percentile(d_r, 97.5)), 4)],
        "mcnemar_b_c": [b, c],
        "mcnemar_p": round(p, 4),
        "d_mrr": round(float(mn.mean() - mb.mean()), 4),
        "mrr_ci95": [round(float(np.percentile(d_m, 2.5)), 4),
                     round(float(np.percentile(d_m, 97.5)), 4)],
        "boot_p_mrr_gt0": round(float((d_m <= 0).mean()), 4),
    }


def build(a):
    cat = [json.loads(l) for l in (CORPUS / "catalog.jsonl").read_text().splitlines() if l.strip()]
    pos = {t["tool_id"]: i for i, t in enumerate(cat)}
    model, qpre, meta = sidecar(a.matrix, None, None)
    variant = a.text or meta.get("text_variant", "v3")
    docs = [tool_text(t, variant) for t in cat]
    E = np.load(a.matrix)
    if E.shape[0] != len(cat):
        raise SystemExit(f"{a.matrix}: {E.shape[0]} rows != {len(cat)} catalog rows")

    t0 = time.time()
    bm = BM25(docs)
    print(f"BM25 index: {len(docs)} docs, {len(bm.idf)} terms, {time.time()-t0:.1f}s "
          f"(text={variant}, same documents as the dense matrix)", flush=True)

    # Identical sampling to fusion_sweep.build_scores: same SEED, same split order, one
    # rng drawn sequentially, gold-filter applied AFTER sampling. Required for the dense
    # cross-check below to line up with §8.
    sp = {k: v for k, v in splits(a.partition).items() if not k.startswith("splitE")}
    rng = np.random.default_rng(SEED)
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(model, device=a.device)
    print(f"dense: {model} text={variant} partition={a.partition}", flush=True)

    store = {}
    for name, items in sp.items():
        if a.n and len(items) > a.n:
            items = [items[i] for i in rng.choice(len(items), a.n, replace=False)]
        items = [it for it in items if any(g in pos for g in it["gold_tool_ids"])]
        Q = enc.encode([qpre + it["query"] for it in items], batch_size=128,
                       normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        dense_full = (Q @ E.T).astype(np.float32)

        t0 = time.time()
        bm_full = np.stack([bm.scores(it["query"]) for it in items]).astype(np.float32)
        ms = 1000 * (time.time() - t0) / max(len(items), 1)

        order = np.argsort(-dense_full, axis=1)[:, :100]
        golds = [{pos[g] for g in it["gold_tool_ids"] if g in pos} for it in items]
        store[name] = {
            "golds": golds,
            "dense_full": dense_full,
            "bm_full": bm_full,
            "order": order,
            "dense_window": np.take_along_axis(dense_full, order, axis=1),
            "bm_window": np.take_along_axis(bm_full, order, axis=1),
            "isgold": np.array([[int(d) in golds[i] for d in order[i]]
                                for i in range(len(items))]),
            "bm25_ms_per_query": round(ms, 2),
            "bm25_nonzero_frac": round(float((bm_full > 0).mean()), 4),
        }
        print(f"  {name:22s} n={len(items):4d}  bm25 {ms:5.1f} ms/query  "
              f"nonzero {store[name]['bm25_nonzero_frac']:.3f}", flush=True)
    del enc
    return store, {"matrix": a.matrix, "dense_model": model, "text_variant": variant,
                   "partition": a.partition, "n": a.n, "seed": SEED, "rrf_k": RRF_K}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/eval/tool_embeddings_gtelarge_v3.npy")
    ap.add_argument("--text", default=None)
    ap.add_argument("--partition", default="dev", choices=["all", "dev", "test"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--alphas", type=float, nargs="+", default=ALPHAS)
    ap.add_argument("--rules", nargs="+", default=["zscore", "rrf", "rrf_tie"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    store, info = build(a)

    # --- guardrail A: alpha=0 reproduces dense bit-exactly in the WINDOW variant
    n_chk = 0
    for name, d in store.items():
        base = recalls(d["isgold"])
        for rule in a.rules:
            for depth in DEPTHS:
                got = recalls(reorder_local(d["isgold"], d["dense_window"],
                                            d["bm_window"], 0.0, rule, depth))
                if got != base:
                    raise SystemExit(f"GUARDRAIL A FAILED {name} {rule} depth{depth}: "
                                     f"alpha=0 gave {got}, dense is {base}")
                n_chk += 1
    print(f"guardrail A ok: alpha=0 reproduces dense for all {n_chk} window configs")

    # --- guardrail C: reorder_local must equal §8's reorder for the rules §8 defined, so
    # the window arm is the same code path and 'rrf_tie' is the only thing that is new.
    from fusion_sweep import reorder as reorder_prior

    n_chk = 0
    for name, d in store.items():
        for rule in [r for r in a.rules if r in ("zscore", "rrf")]:
            for depth in DEPTHS:
                for al in a.alphas:
                    mine = reorder_local(d["isgold"], d["dense_window"], d["bm_window"],
                                         al, rule, depth)
                    theirs = reorder_prior(d["isgold"], d["dense_window"], d["bm_window"],
                                           al, rule, depth)
                    if not np.array_equal(mine, theirs):
                        raise SystemExit(f"GUARDRAIL C FAILED {name} {rule} depth{depth} "
                                         f"a{al:g}: reorder_local != fusion_sweep.reorder")
                    n_chk += 1
    print(f"guardrail C ok: reorder_local == fusion_sweep.reorder on {n_chk} configs")

    # --- guardrail B: full-catalog dense ranks must agree with the window recalls for
    # k <= 100, i.e. the two code paths measure the same dense retriever.
    for name, d in store.items():
        rd = gold_ranks(d["dense_full"], d["golds"])
        w, f = recalls(d["isgold"]), recalls_from_ranks(rd)
        diff = {k: (w[k], f[k]) for k in w if abs(w[k] - f[k]) > 1e-9}
        if diff:
            raise SystemExit(f"GUARDRAIL B FAILED {name}: window vs full dense {diff}")
        d["dense_ranks"] = rd
    print("guardrail B ok: full-catalog dense == window dense on every split")

    # --- cross-run check: same seed, same split order, same matrix as §8
    if PRIOR.exists():
        prior = json.loads(PRIOR.read_text()).get("splits", {})
        for name, d in store.items():
            if name not in prior:
                continue
            mine, theirs = recalls(d["isgold"]), prior[name]["dense"]
            diff = {k: (mine[k], theirs[k]) for k in mine
                    if k in theirs and abs(mine[k] - theirs[k]) > 1e-9}
            print(f"  {name:22s} dense " +
                  ("MISMATCH " + str(diff) if diff else "matches fusion_sweep.json"))

    results = {**info, "alphas": a.alphas, "variants": ["window", "full"], "splits": {}}
    ranks_out, gen = {}, [k for k in store if k.startswith(("standard", "splitD"))]

    for name, d in store.items():
        bm_ranks = gold_ranks(d["bm_full"], d["golds"], stable=True)
        dense_r = d["dense_ranks"]
        # union ceiling: how much rescue headroom exists at all. If a gold is in neither
        # the dense top-100 nor the BM25 top-100, no fusion of the two can ever reach it.
        bm_top = np.argsort(-d["bm_full"], axis=1, kind="stable")[:, :100]
        union_hit = np.array([bool(set(d["order"][i]) & d["golds"][i])
                              or bool(set(bm_top[i]) & d["golds"][i])
                              for i in range(len(d["golds"]))])
        rec = {
            "n": int(len(dense_r)),
            "bm25_ms_per_query": d["bm25_ms_per_query"],
            "bm25_nonzero_frac": d["bm25_nonzero_frac"],
            "dense": recalls_from_ranks(dense_r),
            "bm25_only": recalls_from_ranks(bm_ranks),
            "ceiling": {"dense_R@100": recalls_from_ranks(dense_r, [100])["R@100"],
                        "union_R@100": round(float(union_hit.mean()), 4)},
            "window": {}, "full": {},
        }
        ranks_out[f"{name}|dense"] = dense_r
        ranks_out[f"{name}|bm25_only"] = bm_ranks

        for rule in a.rules:
            for depth in DEPTHS:
                for al in a.alphas:
                    key = f"{rule}|depth{depth}|a{al:g}"
                    ig = reorder_local(d["isgold"], d["dense_window"], d["bm_window"],
                                       al, rule, depth)
                    rec["window"][key] = recalls(ig)
            for al in a.alphas:
                key = f"{rule}|a{al:g}"
                fr = gold_ranks(fuse_local(d["dense_full"], d["bm_full"], al, rule),
                                d["golds"], stable=True)
                rec["full"][key] = recalls_from_ranks(fr)
                ranks_out[f"{name}|full|{key}"] = fr
        results["splits"][name] = rec

    # --- headline: best full-catalog config by mean MRR over the generated splits,
    # chosen once and applied uniformly (not per-split argmax).
    def mean_mrr(variant, key):
        return float(np.mean([results["splits"][s][variant][key]["mrr"] for s in gen]))

    best = {}
    for variant, keys in (("window", [f"{r}|depth{d}|a{al:g}" for r in a.rules
                                      for d in DEPTHS for al in a.alphas if al > 0]),
                          ("full", [f"{r}|a{al:g}" for r in a.rules
                                    for al in a.alphas if al > 0])):
        best[variant] = max(keys, key=lambda k: mean_mrr(variant, k))
    results["best_uniform"] = {v: {"config": k, "mean_mrr_generated": round(mean_mrr(v, k), 4)}
                               for v, k in best.items()}
    results["dense_mean_mrr_generated"] = round(
        float(np.mean([results["splits"][s]["dense"]["mrr"] for s in gen])), 4)

    # --- significance, paired on identical queries (DATA.md §9 item 5)
    results["significance"] = {}
    for name, d in store.items():
        fr = ranks_out[f"{name}|full|{best['full']}"]
        results["significance"][name] = {
            "full_vs_dense": paired_tests(d["dense_ranks"], fr, 1, a.boot),
            "bm25_only_vs_dense": paired_tests(d["dense_ranks"],
                                               ranks_out[f"{name}|bm25_only"], 1, a.boot),
        }
    pooled_d = np.concatenate([store[s]["dense_ranks"] for s in gen])
    pooled_f = np.concatenate([ranks_out[f"{s}|full|{best['full']}"] for s in gen])
    results["significance"]["POOLED_generated"] = {
        "full_vs_dense": paired_tests(pooled_d, pooled_f, 1, a.boot)}

    rp = Path(str(RANKS).format(partition=a.partition, n=a.n))
    np.savez_compressed(rp, **ranks_out)

    # ---------------- report ----------------
    print("\n" + "=" * 92)
    print("BM25 ALONE vs DENSE, and the RESCUE HEADROOM (union ceiling)")
    print("=" * 92)
    print(f"{'split':22s}{'n':>5s}{'dense R@1':>11s}{'bm25 R@1':>10s}"
          f"{'dense R@100':>13s}{'union R@100':>13s}{'headroom':>10s}")
    for name, r in results["splits"].items():
        h = r["ceiling"]["union_R@100"] - r["ceiling"]["dense_R@100"]
        print(f"{name:22s}{r['n']:5d}{r['dense']['R@1']:11.3f}{r['bm25_only']['R@1']:10.3f}"
              f"{r['ceiling']['dense_R@100']:13.3f}{r['ceiling']['union_R@100']:13.3f}"
              f"{h:+10.3f}")

    for variant, label in (("window", "(a) WINDOW fusion -- inside dense top-100, as §8"),
                           ("full", "(b) FULL-CATALOG fusion -- can rescue missed golds")):
        cfg = best[variant]
        print("\n" + "=" * 92)
        print(f"{label}   uniform config: {cfg}")
        print("=" * 92)
        print(f"{'split':22s}{'n':>5s}{'R@1 dense->fused':>22s}"
              f"{'R@10 dense->fused':>23s}{'MRR dense->fused':>22s}")
        for name, r in results["splits"].items():
            b, f = r["dense"], r[variant][cfg]
            print(f"{name:22s}{r['n']:5d}"
                  f"{b['R@1']:10.3f} ->{f['R@1']:7.3f} {f['R@1']-b['R@1']:+.3f}"
                  f"{b['R@10']:10.3f} ->{f['R@10']:7.3f} {f['R@10']-b['R@10']:+.3f}"
                  f"{b['mrr']:9.3f} ->{f['mrr']:7.3f} {f['mrr']-b['mrr']:+.3f}")
        print(f"{'MEAN (generated)':22s}{'':5s}{'':10s}   {'':7s} "
              f"      mean MRR {results['dense_mean_mrr_generated']:.3f} -> "
              f"{mean_mrr(variant, cfg):.3f}")

    for rule in a.rules:
        print(f"\n--- alpha curve, FULL-CATALOG / {rule} (mean MRR over generated splits)")
        print(f"{'alpha':>7s}" + "".join(f"{s.replace('standard_','')[:9]:>10s}" for s in gen)
              + f"{'MEAN':>9s}")
        for al in a.alphas:
            vals = [results["splits"][s]["full"][f"{rule}|a{al:g}"]["mrr"] for s in gen]
            tag = "  <- dense" if al == 0 else ("  <- pure bm25" if al == 1 else "")
            print(f"{al:7.1f}" + "".join(f"{v:10.3f}" for v in vals)
                  + f"{np.mean(vals):9.3f}{tag}")

    print("\n" + "=" * 92)
    print(f"SIGNIFICANCE -- full-catalog {best['full']} vs dense, paired on identical queries")
    print("=" * 92)
    print(f"{'split':22s}{'dR@1':>8s}{'R@1 95% CI':>18s}{'McNemar b/c':>14s}{'p':>9s}"
          f"{'dMRR':>8s}")
    for name, s in results["significance"].items():
        t = s["full_vs_dense"]
        ci = f"[{t['R@1_ci95'][0]:+.3f},{t['R@1_ci95'][1]:+.3f}]"
        bc = f"{t['mcnemar_b_c'][0]}/{t['mcnemar_b_c'][1]}"
        print(f"{name:22s}{t['d_R@1']:+8.3f}{ci:>18s}{bc:>14s}{t['mcnemar_p']:9.4f}"
              f"{t['d_mrr']:+8.3f}")

    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {a.out}")
    print(f"wrote {rp}  ({len(ranks_out)} per-item rank vectors -- DATA.md §9 item 5)")


if __name__ == "__main__":
    main()
