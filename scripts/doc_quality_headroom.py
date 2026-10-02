"""Does arm 2 (description rewriting) have headroom left? -> data/eval/doc_quality_headroom.json

§10.7's lesson: screen a proposed lever on how much error mass it can physically reach
BEFORE spending anything on it. For BM25 fusion that screen was union recall headroom,
and it would have predicted the negative result in minutes. The analogue for arm 2 is:

  Of the queries whose gold tool is NOT retrieved, how many of those golds are poorly
  documented? If retrieval failures sit mostly on WELL-documented tools, rewriting
  descriptions cannot fix them however good the rewriting is.

The tension this resolves: §6 found v3 document text (tags, provider, HTTP method,
tokenized path) already lifted thin-description tools from R@10 0.348 -> 0.568, which is
the same subgroup arm 2 targets by a different mechanism. So some of arm 2's nominal
headroom has already been harvested, and the question is how much is left.

HEADROOM ESTIMATE. Bucket items by the doc_quality of their gold tool, then ask what
R@k would be if the low-quality buckets retrieved as well as the well-documented ones:

  gain_upper = sum over bad buckets of  n_q * (R@k_ref - R@k_q) / N        (ref = dq>=3)

This is an UPPER BOUND, not a prediction, and it is loose in a specific direction:
doc_quality correlates with how obscure a tool is, so part of the gap is intrinsic
difficulty that no rewriting touches. Read it as a ceiling -- if the ceiling is small,
stop; if it is large, the measurement is worth $7.

Runs over the FULL dev partition (not n=150) because it is free and this estimate is
power-sensitive. Dense-only, no cross-encoder, ~1 min.

  ./.venv/bin/python scripts/doc_quality_headroom.py --device mps
"""

import argparse, json
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from eval_retrieval import CORPUS, sidecar, splits

OUT = Path("data/eval/doc_quality_headroom.json")
KS = [1, 10, 25]
DQ_BUCKETS = [("dq0_none", [0]), ("dq1_poor", [1]), ("dq2_fair", [2]),
              ("dq3_4_good", [3, 4])]
REF_BUCKET = "dq3_4_good"


def desc_words(t):
    return len((t.get("description") or "").split())


def bucket_stats(ranks, labels, names, ks=KS):
    """R@k per bucket label, plus the share of miss@10 mass each bucket carries."""
    out, miss10 = {}, ranks > 10
    total_miss = int(miss10.sum())
    for nm in names:
        sel = labels == nm
        n = int(sel.sum())
        if not n:
            continue
        row = {"n": n, "share_of_items": round(n / len(ranks), 4)}
        for k in ks:
            row[f"R@{k}"] = round(float((ranks[sel] <= k).mean()), 4)
        row["n_miss@10"] = int((miss10 & sel).sum())
        row["share_of_miss@10"] = round(float((miss10 & sel).sum() / total_miss), 4) if total_miss else 0.0
        out[nm] = row
    return out, total_miss


def headroom(stats, ref=REF_BUCKET, ks=KS):
    """Upper bound on R@k gain if every weak bucket matched the reference bucket.

    Returns None when the reference bucket is absent -- which is the case for the
    MetaTool anchor, whose 199 tools are ALL doc_quality <= 2 (175 of them at 1). There
    is no well-documented subgroup there to serve as the target, so the headroom is
    unmeasurable by this method, NOT zero. Do not print it as 0.
    """
    if ref not in stats:
        return None
    n_tot = sum(v["n"] for v in stats.values())
    out = {}
    for k in ks:
        r_ref = stats[ref][f"R@{k}"]
        gain = sum(v["n"] * max(0.0, r_ref - v[f"R@{k}"])
                   for nm, v in stats.items() if nm != ref)
        out[f"R@{k}"] = round(gain / n_tot, 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/eval/tool_embeddings_gtelarge_v3.npy")
    ap.add_argument("--partition", default="dev", choices=["all", "dev", "test"])
    ap.add_argument("--n", type=int, default=0, help="0 = all items (recommended)")
    ap.add_argument("--thin", type=int, default=5, help="description word count = 'thin'")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    cat = [json.loads(l) for l in (CORPUS / "catalog.jsonl").read_text().splitlines() if l.strip()]
    pos = {t["tool_id"]: i for i, t in enumerate(cat)}
    model, qpre, meta = sidecar(a.matrix, None, None)
    E = np.load(a.matrix)
    if E.shape[0] != len(cat):
        raise SystemExit(f"{a.matrix}: {E.shape[0]} rows != {len(cat)} catalog rows")

    dq_of = {}
    for nm, vals in DQ_BUCKETS:
        for v in vals:
            dq_of[v] = nm

    sp = {k: v for k, v in splits(a.partition).items() if not k.startswith("splitE")}
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(model, device=a.device)
    print(f"{model} text={meta.get('text_variant')} partition={a.partition}", flush=True)

    results = {"matrix": a.matrix, "dense_model": model,
               "text_variant": meta.get("text_variant"), "partition": a.partition,
               "thin_threshold": a.thin, "ref_bucket": REF_BUCKET, "splits": {}}
    pooled = {"ranks": [], "dq": [], "thin": [], "gold_rows": []}

    for name, items in sp.items():
        if a.n and len(items) > a.n:
            items = items[: a.n]
        items = [it for it in items if any(g in pos for g in it["gold_tool_ids"])]
        Q = enc.encode([qpre + it["query"] for it in items], batch_size=128,
                       normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        S = Q @ E.T
        order = np.argsort(-S, axis=1)

        ranks, dq_lab, thin_lab, gold_rows = [], [], [], []
        for i, it in enumerate(items):
            g = [pos[x] for x in it["gold_tool_ids"] if x in pos]
            rank_of_gold = {j: int(np.where(order[i] == j)[0][0]) + 1 for j in g}
            best = min(rank_of_gold, key=lambda j: rank_of_gold[j])
            ranks.append(rank_of_gold[best])
            t = cat[best]
            dq_lab.append(dq_of.get(t.get("doc_quality"), "dq_unknown"))
            thin_lab.append("thin" if desc_words(t) <= a.thin else "rich")
            gold_rows.append(best)
        ranks = np.array(ranks)
        dq_lab, thin_lab = np.array(dq_lab), np.array(thin_lab)

        dq_stats, n_miss = bucket_stats(ranks, dq_lab, [n for n, _ in DQ_BUCKETS])
        th_stats, _ = bucket_stats(ranks, thin_lab, ["thin", "rich"])
        results["splits"][name] = {
            "n": int(len(ranks)), "n_miss@10": n_miss,
            "overall": {f"R@{k}": round(float((ranks <= k).mean()), 4) for k in KS},
            "by_doc_quality": dq_stats, "by_desc_length": th_stats,
            "headroom_upper_bound": headroom(dq_stats),
        }
        for key, val in (("ranks", ranks), ("dq", dq_lab), ("thin", thin_lab),
                         ("gold_rows", gold_rows)):
            pooled[key].append(np.asarray(val))
        print(f"  {name:22s} n={len(ranks):5d}  R@10 {float((ranks<=10).mean()):.3f}  "
              f"miss@10 {n_miss}", flush=True)
    del enc

    # --- pooled over the GENERATED splits (the primary set; dev_A is the anchor)
    gen_idx = [i for i, nm in enumerate(sp) if nm.startswith(("standard", "splitD"))]
    pr = np.concatenate([pooled["ranks"][i] for i in gen_idx])
    pdq = np.concatenate([pooled["dq"][i] for i in gen_idx])
    pth = np.concatenate([pooled["thin"][i] for i in gen_idx])
    dq_stats, n_miss = bucket_stats(pr, pdq, [n for n, _ in DQ_BUCKETS])
    th_stats, _ = bucket_stats(pr, pth, ["thin", "rich"])
    results["POOLED_generated"] = {
        "n": int(len(pr)), "n_miss@10": n_miss,
        "overall": {f"R@{k}": round(float((pr <= k).mean()), 4) for k in KS},
        "by_doc_quality": dq_stats, "by_desc_length": th_stats,
        "headroom_upper_bound": headroom(dq_stats),
    }

    # --- rewrite scope. Criteria must be LABEL-INDEPENDENT: selecting tools to rewrite by
    # "is gold for some eval intent" would leak the labels into the index. Count by
    # document property only, and report gold coverage purely as context.
    all_gold = set(np.concatenate([np.asarray(x) for x in pooled["gold_rows"]]).tolist())
    scope = {}
    for nm, pred in (("doc_quality<=1", lambda t: (t.get("doc_quality") or 0) <= 1),
                     ("desc_words<=5", lambda t: desc_words(t) <= a.thin),
                     ("no_description", lambda t: not (t.get("description") or "").strip())):
        rows = [i for i, t in enumerate(cat) if pred(t)]
        scope[nm] = {"n_tools": len(rows),
                     "n_that_are_gold_in_this_partition": len(set(rows) & all_gold),
                     "mean_desc_words": round(float(np.mean([desc_words(cat[i]) for i in rows])), 1)}
    results["rewrite_scope"] = scope

    # --- SELECTION EFFECT: are poorly-documented tools represented as golds at the rate
    # they occur in the catalog? generate_intents.py had to READ a tool's documentation to
    # write an intent for it, so tools with no usable description are plausibly
    # under-sampled as golds by construction. If so, this benchmark structurally
    # understates what description rewriting is worth in production, and no amount of
    # rewriting fixes that -- only regenerating intents for those tools would.
    gold_dq = Counter(dq_of.get(cat[i].get("doc_quality"), "dq_unknown")
                      for i in np.concatenate([np.asarray(x) for x in
                                               [pooled["gold_rows"][i] for i in gen_idx]]).tolist())
    cat_dq = Counter(dq_of.get(t.get("doc_quality"), "dq_unknown") for t in cat)
    n_gold, n_cat = sum(gold_dq.values()), sum(cat_dq.values())
    results["gold_vs_catalog_dq"] = {
        nm: {"catalog_share": round(cat_dq[nm] / n_cat, 4),
             "gold_share": round(gold_dq[nm] / n_gold, 4),
             "representation_ratio": round((gold_dq[nm] / n_gold) / (cat_dq[nm] / n_cat), 3)
             if cat_dq[nm] else None}
        for nm, _ in DQ_BUCKETS}

    print("\n" + "=" * 86)
    print("SELECTION EFFECT -- are weak-doc tools even represented as golds?")
    print("(ratio < 1 means under-sampled relative to the catalog, so this benchmark")
    print(" understates what rewriting them is worth)")
    print("=" * 86)
    print(f"{'bucket':14s}{'catalog share':>16s}{'gold share':>13s}{'ratio':>9s}")
    for nm, v in results["gold_vs_catalog_dq"].items():
        print(f"{nm:14s}{v['catalog_share']:16.4f}{v['gold_share']:13.4f}"
              f"{v['representation_ratio']:9.3f}")

    # ---------------- report ----------------
    p = results["POOLED_generated"]
    print("\n" + "=" * 86)
    print(f"RETRIEVAL BY GOLD doc_quality -- pooled generated splits, n={p['n']}, "
          f"dev partition")
    print("=" * 86)
    print(f"{'bucket':14s}{'n':>7s}{'share':>8s}{'R@1':>8s}{'R@10':>8s}{'R@25':>8s}"
          f"{'miss@10':>9s}{'% of miss':>11s}")
    for nm, v in p["by_doc_quality"].items():
        print(f"{nm:14s}{v['n']:7d}{v['share_of_items']:8.3f}{v['R@1']:8.3f}"
              f"{v['R@10']:8.3f}{v['R@25']:8.3f}{v['n_miss@10']:9d}"
              f"{v['share_of_miss@10']:11.3f}")
    print(f"{'OVERALL':14s}{p['n']:7d}{1.0:8.3f}{p['overall']['R@1']:8.3f}"
          f"{p['overall']['R@10']:8.3f}{p['overall']['R@25']:8.3f}{p['n_miss@10']:9d}")

    print(f"\n{'by desc length':14s}{'n':>7s}{'share':>8s}{'R@1':>8s}{'R@10':>8s}"
          f"{'R@25':>8s}{'miss@10':>9s}{'% of miss':>11s}")
    for nm, v in p["by_desc_length"].items():
        print(f"{nm:14s}{v['n']:7d}{v['share_of_items']:8.3f}{v['R@1']:8.3f}"
              f"{v['R@10']:8.3f}{v['R@25']:8.3f}{v['n_miss@10']:9d}"
              f"{v['share_of_miss@10']:11.3f}")

    print("\n" + "=" * 86)
    print("HEADROOM UPPER BOUND -- if every weak doc_quality bucket retrieved as well")
    print(f"as {REF_BUCKET}. Loose: doc_quality correlates with obscurity, and that part")
    print("is intrinsic difficulty no rewriting reaches.")
    print("=" * 86)
    def hrow(label, h):
        if h is None:
            print(f"{label:22s}" + f"{'n/a -- no dq3_4 golds exist in this split':>30s}")
        else:
            print(f"{label:22s}" + "".join(f"{h['R@'+str(k)]:+10.4f}" for k in KS))

    print(f"{'split':22s}" + "".join(f"{'+R@'+str(k):>10s}" for k in KS))
    for name, r in results["splits"].items():
        hrow(name, r["headroom_upper_bound"])
    hrow("POOLED_generated", p["headroom_upper_bound"])

    print("\n" + "=" * 86)
    print("REWRITE SCOPE (label-independent criteria only)")
    print("=" * 86)
    for nm, v in scope.items():
        print(f"  {nm:18s} {v['n_tools']:5d} tools, mean {v['mean_desc_words']:5.1f} "
              f"desc words, {v['n_that_are_gold_in_this_partition']:4d} are gold here")

    Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
