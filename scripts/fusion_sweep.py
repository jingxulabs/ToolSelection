"""Step 3b: score FUSION over the dense shortlist -> data/eval/fusion_sweep.json.

rerank_sweep.py (DATA.md §7) reranked by REPLACEMENT: inside the top-d it sorted purely
by cross-encoder score and discarded the dense ordering entirely. That is the mechanical
reason it decayed monotonically with depth -- the deeper it reranked, the more of a
stronger signal it threw away. But the reranker is not noise: it scores 5.5-7.8x above a
random shuffle of the same shortlist (§7.2), it is merely weaker than gte-large/v3. So
COMBINING the two scores should dominate replacing one with the other.

Two fusion rules, both reducing to dense at alpha=0 and pure rerank at alpha=1:

  zscore  fused = (1-a)*z(dense) + a*z(rerank)            per-query z over the window
  rrf     fused = (1-a)/(K+rank_dense) + a/(K+rank_rerank)  rank-based, scale-free

Cosine similarities and cross-encoder logits live on different scales, so raw addition is
meaningless; zscore puts them on a common scale, rrf throws scale away altogether. The
two agreeing is evidence the effect is not a normalisation artifact.

alpha=0 is a GUARDRAIL, not a datapoint. Dense scores along a shortlist row are already
descending, so at alpha=0 a stable argsort must return the identity permutation and
reproduce the dense baseline bit-exactly. The run asserts this and refuses to report if
it fails. It also cross-checks the dense block against rerank_sweep.json from the earlier
run -- same seed, same split order, so the shortlists must be identical.

Cross-encoder scoring is the only expensive part (~157,800 pairs, ~70 min on MPS), so it
is cached to an .npz keyed by (matrix, partition, n, seed). The alpha x depth x rule
sweep then runs off the cache in seconds: re-sweep freely with --alphas, and only pass
--rescore if the shortlists themselves change.

  ./.venv/bin/python scripts/fusion_sweep.py --partition dev --n 150 --device mps
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

OUT = Path("data/eval/fusion_sweep.json")
CACHE = Path("data/eval/fusion_scores_{partition}_n{n}.npz")
PRIOR = Path("data/eval/rerank_sweep.json")
DEPTHS = [10, 25, 50, 100]
KS = [1, 5, 10, 25]
SEED = 13
RRF_K = 60


def recalls(isgold, ks=KS):
    """isgold: (n, 100) bool, column j = "position j of the CURRENT ranking is gold".

    Mirrors rerank_sweep.recalls: rank of the FIRST gold (min over golds), and rows with
    no gold in the shortlist score 0 everywhere rather than being dropped.
    """
    n = isgold.shape[0]
    ranks = np.full(n, 10**6, dtype=np.int64)
    hit = isgold.any(axis=1)
    ranks[hit] = isgold[hit].argmax(axis=1) + 1
    out = {f"R@{k}": round(float((ranks <= k).mean()), 4) for k in ks}
    out["mrr"] = round(float(np.where(ranks <= 100, 1.0 / ranks, 0.0).mean()), 4)
    return out


def rank_of(s):
    """1-based rank by descending score, per row. Ties broken by position (stable)."""
    idx = np.argsort(-s, axis=1, kind="stable")
    r = np.empty_like(idx)
    cols = np.broadcast_to(np.arange(1, s.shape[1] + 1), s.shape)
    np.put_along_axis(r, idx, cols, axis=1)
    return r.astype(np.float32)


def z(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)


def fuse(dense, rr, alpha, rule):
    if rule == "zscore":
        return (1 - alpha) * z(dense) + alpha * z(rr)
    if rule == "rrf":
        return (1 - alpha) / (RRF_K + rank_of(dense)) + alpha / (RRF_K + rank_of(rr))
    raise ValueError(rule)


def reorder(isgold, dense, rr, alpha, rule, depth):
    """Fuse inside the top-`depth` window, leave the tail in dense order."""
    f = fuse(dense[:, :depth], rr[:, :depth], alpha, rule)
    perm = np.argsort(-f, axis=1, kind="stable")
    head = np.take_along_axis(isgold[:, :depth], perm, axis=1)
    return np.concatenate([head, isgold[:, depth:]], axis=1)


def build_scores(a):
    """-> {split: {"isgold": (n,100) bool, "dense": (n,100) f32, model: (n,100) f32}}"""
    cache = Path(str(CACHE).format(partition=a.partition, n=a.n))
    if cache.exists() and not a.rescore:
        d = np.load(cache, allow_pickle=False)
        names = sorted({k.rsplit("|", 1)[0] for k in d.files})
        out = {n: {k.rsplit("|", 1)[1]: d[k] for k in d.files if k.startswith(n + "|")}
               for n in names}
        print(f"loaded cached scores from {cache}")
        return out, json.loads(cache.with_suffix(".meta.json").read_text())

    cat = [json.loads(l) for l in (CORPUS / "catalog.jsonl").read_text().splitlines() if l.strip()]
    pos = {t["tool_id"]: i for i, t in enumerate(cat)}
    model, qpre, meta = sidecar(a.matrix, None, None)
    variant = a.text or meta.get("text_variant", "v3")
    docs = [tool_text(t, variant) for t in cat]
    E = np.load(a.matrix)
    if E.shape[0] != len(cat):
        raise SystemExit(f"{a.matrix}: {E.shape[0]} rows != {len(cat)} catalog rows")

    # Identical sampling to rerank_sweep.py: same SEED, same split order, one rng drawn
    # sequentially, gold-filter applied AFTER sampling. Required for the shortlists --
    # and so the cross-run dense check below -- to line up.
    sp = {k: v for k, v in splits(a.partition).items() if not k.startswith("splitE")}
    rng = np.random.default_rng(SEED)
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(model, device=a.device)
    print(f"dense: {model} text={variant} partition={a.partition}", flush=True)

    store, queries = {}, {}
    for name, items in sp.items():
        if a.n and len(items) > a.n:
            items = [items[i] for i in rng.choice(len(items), a.n, replace=False)]
        items = [it for it in items if any(g in pos for g in it["gold_tool_ids"])]
        Q = enc.encode([qpre + it["query"] for it in items], batch_size=128,
                       normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        sims = Q @ E.T
        order = np.argsort(-sims, axis=1)[:, :100]
        dense = np.take_along_axis(sims, order, axis=1)
        gold = [{pos[g] for g in it["gold_tool_ids"] if g in pos} for it in items]
        isgold = np.array([[int(d) in gold[i] for d in order[i]] for i in range(len(items))])
        store[name] = {"isgold": isgold, "dense": dense.astype(np.float32)}
        queries[name] = (items, order)
        print(f"  {name:22s} n={len(items)}", flush=True)
    del enc

    from sentence_transformers import CrossEncoder

    for rr_name in a.rerankers:
        try:
            ce = CrossEncoder(rr_name, device=a.device, max_length=512)
        except Exception as e:
            print(f"  !! {rr_name} unavailable: {type(e).__name__}: {str(e)[:90]}")
            continue
        for name, (items, order) in queries.items():
            t0 = time.time()
            pairs = [(it["query"], docs[int(d)]) for i, it in enumerate(items) for d in order[i]]
            s = np.asarray(ce.predict(pairs, batch_size=a.batch, show_progress_bar=False),
                           dtype=np.float32).reshape(len(items), 100)
            store[name][rr_name] = s
            el = time.time() - t0
            print(f"  {rr_name.split('/')[-1]:20s} {name:22s} {len(pairs)} pairs "
                  f"in {el:.0f}s ({1000*el/max(len(items),1):.0f} ms/query)", flush=True)
        del ce

    flat = {f"{n}|{k}": v for n, d in store.items() for k, v in d.items()}
    np.savez_compressed(cache, **flat)
    info = {"matrix": a.matrix, "dense_model": model, "text_variant": variant,
            "partition": a.partition, "n": a.n, "seed": SEED}
    cache.with_suffix(".meta.json").write_text(json.dumps(info, indent=2))
    print(f"cached scores -> {cache}")
    return store, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/eval/tool_embeddings_gtelarge_v3.npy")
    ap.add_argument("--rerankers", nargs="+",
                    default=["BAAI/bge-reranker-base", "BAAI/bge-reranker-large"])
    ap.add_argument("--text", default=None)
    ap.add_argument("--partition", default="dev", choices=["all", "dev", "test"])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--rules", nargs="+", default=["zscore", "rrf"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rescore", action="store_true", help="ignore the score cache")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    store, info = build_scores(a)
    models = [m for m in a.rerankers if any(m in d for d in store.values())]

    # --- guardrail: alpha=0 must reproduce dense bit-exactly, under both rules
    for name, d in store.items():
        base = recalls(d["isgold"])
        for m in models:
            for rule in a.rules:
                for depth in DEPTHS:
                    got = recalls(reorder(d["isgold"], d["dense"], d[m], 0.0, rule, depth))
                    if got != base:
                        raise SystemExit(f"GUARDRAIL FAILED {name} {m} {rule} depth{depth}: "
                                         f"alpha=0 gave {got}, dense is {base}")
    print(f"guardrail ok: alpha=0 reproduces dense for all "
          f"{len(store)*len(models)*len(a.rules)*len(DEPTHS)} configs")

    # --- cross-run check: same seed and split order, so shortlists must match §7's
    if PRIOR.exists():
        prior = json.loads(PRIOR.read_text())["splits"]
        for name, d in store.items():
            if name not in prior:
                continue
            mine, theirs = recalls(d["isgold"]), prior[name]["dense"]
            diff = {k: (mine[k], theirs[k]) for k in mine if abs(mine[k] - theirs[k]) > 1e-9}
            flag = "MISMATCH " + str(diff) if diff else "matches rerank_sweep.json"
            print(f"  {name:22s} dense {flag}")

    results = {**info, "rrf_k": RRF_K, "alphas": a.alphas, "splits": {}}
    for name, d in store.items():
        rec = {"n": int(d["isgold"].shape[0]), "dense": recalls(d["isgold"]),
               "ceiling": {f"R@{x}": recalls(d["isgold"], [x])[f"R@{x}"] for x in DEPTHS},
               "fusion": {}}
        for m in models:
            for rule in a.rules:
                for depth in DEPTHS:
                    for al in a.alphas:
                        key = f"{m}|{rule}|depth{depth}|a{al:g}"
                        rec["fusion"][key] = recalls(
                            reorder(d["isgold"], d["dense"], d[m], al, rule, depth))
        results["splits"][name] = rec

    # --- report: best fusion per split, and the alpha curve at the best depth
    gen = [k for k in results["splits"] if k.startswith(("standard", "splitD"))]
    print("\n" + "=" * 78)
    print("BEST FUSION PER SPLIT (by MRR; alpha=0 is dense, alpha=1 is pure rerank)")
    print("=" * 78)
    print(f"{'split':20s}{'dense MRR':>10s}{'best MRR':>10s}{'dR@1':>8s}{'dR@10':>8s}   config")
    for name, r in results["splits"].items():
        best = max(r["fusion"].items(), key=lambda kv: kv[1]["mrr"])
        k, v = best
        d1, d10 = v["R@1"] - r["dense"]["R@1"], v["R@10"] - r["dense"]["R@10"]
        m, rule, dep, al = k.split("|")
        print(f"{name:20s}{r['dense']['mrr']:10.3f}{v['mrr']:10.3f}{d1:+8.3f}{d10:+8.3f}   "
              f"{m.split('/')[-1]} {rule} {dep} {al}")

    for m in models:
        for rule in a.rules:
            print(f"\n--- alpha curve, {m.split('/')[-1]} / {rule} / depth100 "
                  f"(mean MRR over generated splits)")
            print(f"{'alpha':>7s}" + "".join(f"{s.replace('standard_','')[:9]:>10s}"
                                             for s in gen) + f"{'MEAN':>9s}")
            for al in a.alphas:
                vals = [results["splits"][s]["fusion"][f"{m}|{rule}|depth100|a{al:g}"]["mrr"]
                        for s in gen]
                tag = "  <- dense" if al == 0 else ("  <- pure rerank" if al == 1 else "")
                print(f"{al:7.1f}" + "".join(f"{v:10.3f}" for v in vals)
                      + f"{np.mean(vals):9.3f}{tag}")

    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
