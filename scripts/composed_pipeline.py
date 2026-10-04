"""The composed pipeline, end to end: retrieve 100 -> fuse -> top k -> select.

DATA.md §9 item 4 has sat open because the three stages were only ever measured
apart. §6-§8 score retrieval and never call a selector; §4d calls a selector but
*forces gold into the shortlist*, which deletes the failure mode the whole
experiment is about. So the headline "end-to-end ~= 0.60" is an arithmetic
composition of two independent measurements, not an observed number. This script
observes it.

    user intent -> dense top-100 -> fuse cross-encoder at alpha -> top k -> selector -> 1

Two arms over the SAME items, so the reranker's contribution is paired:

    dense   top-k straight off the retriever        (the shipped path)
    fused   top-k after the depth-100 fusion rerank (the proposed path)

Why this is the only way to settle the reranker question. Reordering the top k
cannot change R@k -- it is a permutation of a fixed set, so recall is invariant
(verify: rerank_sweep depth_10 leaves R@10 and R@25 bit-identical). A reranker can
therefore only help by reaching DEEPER than the shortlist it feeds, pulling gold
from ranks k+1..depth into the top k. Measured on dev, fusion does exactly that
and it survives a paired test -- R@25 0.745 -> 0.759, McNemar p = 0.012 -- but the
left factor is only part of the story. If the rescued items are ones the selector
then fumbles, or if the reshuffled distractors cost more than the rescues gain,
the end-to-end number will not move. That is what this measures.

COST. Stage 3 calls `claude` and spends real money on the user's account, so
--dry-run is the default and reports the free half plus an itemised estimate.
Nothing is spent without --run. Shortlists identical across the two arms are
deduplicated, because a selector reading the same candidate set twice returns the
same answer and paying twice measures nothing.

Sampling is byte-identical to rerank_sweep.py and fusion_sweep.py -- same SEED,
same split order, one rng drawn sequentially, gold-filter applied AFTER sampling --
so the cached cross-encoder scores in data/eval/fusion_scores_{part}_n{n}.npz line
up and stage 2 is free. The alignment is asserted against the cached isgold flags
rather than assumed; a mismatch aborts instead of silently scoring garbage.

    ./.venv/bin/python scripts/composed_pipeline.py --partition dev --n 150
    ./.venv/bin/python scripts/composed_pipeline.py --partition dev --n 150 --run
"""

import argparse, json, random, re, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import comb
from pathlib import Path

import numpy as np

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_embeddings import tool_text
from eval_retrieval import CORPUS, sidecar, splits

OUT = Path("data/eval/composed_pipeline.json")
CACHE = Path("data/eval/fusion_scores_{partition}_n{n}.npz")
MATRIX = Path("data/eval/tool_embeddings_gtelarge_v3.npy")
SEED = 13
RERANKER = "BAAI/bge-reranker-large"

_lock = threading.Lock()
_c = {"usd": 0.0, "calls": 0, "err": 0}


# ── stage 3 plumbing, reused verbatim from roundtrip_check.py so the selector is
#    the same one the 0.93 was measured with ──

def call(prompt, timeout=300):
    try:
        p = subprocess.run(["claude", "-p", prompt, "--output-format", "json"],
                           capture_output=True, text=True, timeout=timeout)
        d = json.loads(p.stdout)
    except Exception:
        with _lock:
            _c["err"] += 1
        return None
    with _lock:
        _c["usd"] += d.get("total_cost_usd") or 0.0
        _c["calls"] += 1
        if d.get("is_error"):
            _c["err"] += 1
            return None
    return d.get("result", "")


def parse(txt):
    if not txt:
        return []
    m = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S)
    if m:
        txt = m.group(1)
    i, j = txt.find("["), txt.rfind("]")
    if i < 0:
        return []
    try:
        o = json.loads(txt[i : j + 1])
        return o if isinstance(o, list) else []
    except json.JSONDecodeError:
        return []


def build_prompt(items, by_id):
    head = [
        "For each REQUEST below, choose which ONE candidate operation best satisfies it.",
        'If no candidate fits, answer "none". If two or more fit equally well, answer the best',
        'one and set "tied": true.',
        'Output ONLY a JSON array of {"id":..., "choice":"<candidate tool_id or none>", "tied":true|false}.',
        "No prose.",
        "",
    ]
    for it in items:
        head.append(f'REQUEST {it["qid"]}: {it["query"]}')
        head.append("CANDIDATES:")
        for cid in it["cands"]:
            t = by_id[cid]
            d = (t["description"] or t.get("summary") or "(no description)")[:180]
            head.append(f'  - {cid} | {t["name"]} | {t["api_title"][:40]} | {d}')
        head.append("")
    return "\n".join(head)


def zscore(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot_ci(a, b, seed=SEED, reps=10000):
    """Paired bootstrap on the arm difference, resampling items."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    d = [b[i].mean() - a[i].mean()
         for i in (rng.integers(0, len(a), len(a)) for _ in range(reps))]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="dev", choices=["dev", "test"])
    ap.add_argument("--n", type=int, default=150, help="items per split, matched to the cache")
    ap.add_argument("--k", type=int, default=25, help="shortlist size handed to the selector")
    ap.add_argument("--depth", type=int, default=100, help="rerank window")
    ap.add_argument("--alpha", type=float, default=0.2, help="cross-encoder weight in the fusion")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch", type=int, default=8, help="items per selector call")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget-usd", type=float, default=6.0, help="hard stop; truncates silently")
    ap.add_argument("--run", action="store_true",
                    help="actually call the selector and spend money; off by default")
    a = ap.parse_args()

    if a.partition == "test" and a.run:
        print("REFUSING: --partition test is the single held-out access (DATA.md §9 item 4).\n"
              "Run it deliberately by editing this guard, not by passing a flag.", file=sys.stderr)
        return 2

    # ── catalog ──
    cat = [json.loads(l) for l in (CORPUS / "catalog.jsonl").read_text().splitlines() if l.strip()]
    by_id = {t["tool_id"]: t for t in cat}
    ids = [t["tool_id"] for t in cat]
    pos = {t: i for i, t in enumerate(ids)}

    E = np.load(MATRIX).astype(np.float32)
    if E.shape[0] != len(cat):
        raise SystemExit(f"matrix rows {E.shape[0]} != catalog {len(cat)}; regenerate")
    model, qpre, meta = sidecar(str(MATRIX), None, None)
    variant = meta.get("text_variant")
    qpre = qpre or ""

    # ── stage 1: dense top-100. Sampling mirrors fusion_sweep.py exactly. ──
    sp = {k: v for k, v in splits(a.partition).items() if not k.startswith("splitE")}
    rng = np.random.default_rng(SEED)
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(model, device=a.device)
    print(f"stage 1  dense {model} text={variant} partition={a.partition}", flush=True)

    store = {}
    for name, items in sp.items():
        if a.n and len(items) > a.n:
            items = [items[i] for i in rng.choice(len(items), a.n, replace=False)]
        items = [it for it in items if any(g in pos for g in it["gold_tool_ids"])]
        Q = enc.encode([qpre + it["query"] for it in items], batch_size=128,
                       normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        sims = Q @ E.T
        order = np.argsort(-sims, axis=1)[:, : a.depth]
        dense = np.take_along_axis(sims, order, axis=1)
        gold = [{pos[g] for g in it["gold_tool_ids"] if g in pos} for it in items]
        isgold = np.array([[int(d) in gold[i] for d in order[i]] for i in range(len(items))])
        store[name] = {"items": items, "order": order, "dense": dense, "isgold": isgold}
        print(f"  {name:22s} n={len(items)}", flush=True)
    del enc

    # ── stage 2: fusion. Reuse cached cross-encoder scores, asserting alignment. ──
    cache = Path(str(CACHE).format(partition=a.partition, n=a.n))
    if not cache.exists():
        raise SystemExit(f"need {cache}; run scripts/fusion_sweep.py --partition {a.partition} "
                         f"--n {a.n} first (~70 min local, no API spend)")
    z = np.load(cache, allow_pickle=True)
    print(f"stage 2  fusion alpha={a.alpha} depth={a.depth} from {cache.name}", flush=True)

    for name, s in store.items():
        key = f"{name}|{RERANKER}"
        if key not in z:
            raise SystemExit(f"{cache} has no {key}")
        ce = z[key][:, : a.depth]
        cached_isgold = z[f"{name}|isgold"][:, : a.depth]
        if cached_isgold.shape != s["isgold"].shape or not (cached_isgold == s["isgold"]).all():
            raise SystemExit(
                f"ALIGNMENT FAILURE on {name}: the dense top-{a.depth} recomputed here does not "
                f"match the one the cached cross-encoder scores were computed over. The cache is "
                f"keyed by (matrix, partition, n, seed) but not by catalog content -- if the "
                f"catalog was rebuilt, re-run fusion_sweep.py. Refusing to score misaligned rows.")
        s["fused"] = (1 - a.alpha) * zscore(s["dense"]) + a.alpha * zscore(ce)

    # ── build the two shortlists and the free left factor ──
    work, meta_by_qid, left = [], {}, {}
    seen = {}            # candidate-set signature -> qid that already carries it
    for name, s in store.items():
        items, order, isgold = s["items"], s["order"], s["isgold"]
        gd, gf = [], []
        for i, it in enumerate(items):
            arms = {"dense": np.argsort(-s["dense"][i])[: a.k],
                    "fused": np.argsort(-s["fused"][i])[: a.k]}
            gd.append(bool(isgold[i, arms["dense"]].any()))
            gf.append(bool(isgold[i, arms["fused"]].any()))
            for arm, cols in arms.items():
                cands = [ids[int(order[i, c])] for c in cols]
                sig = (it["query"], frozenset(cands))
                qid = f"{name}#{i}#{arm}"
                meta_by_qid[qid] = {"split": name, "arm": arm, "query": it["query"],
                                    "gold": [g for g in it["gold_tool_ids"] if g in pos],
                                    "has_gold": bool(isgold[i, cols].any()), "item": i}
                if sig in seen:
                    meta_by_qid[qid]["alias_of"] = seen[sig]   # same set, do not pay twice
                    continue
                seen[sig] = qid
                cl = list(cands)
                random.Random(SEED + i).shuffle(cl)            # position must not be a cue
                work.append({"qid": qid, "query": it["query"], "cands": cl})
        left[name] = {"n": len(items), "R@k_dense": float(np.mean(gd)),
                      "R@k_fused": float(np.mean(gf)),
                      "rescued": int(sum(f and not d for d, f in zip(gd, gf))),
                      "lost": int(sum(d and not f for d, f in zip(gd, gf)))}

    # pooled left factor over the generated splits, with a paired test
    GENP = [n for n in store if n.startswith(("standard_", "splitD_"))]
    gd = [meta_by_qid[f"{n}#{i}#dense"]["has_gold"] for n in GENP for i in range(left[n]["n"])]
    gf = [meta_by_qid[f"{n}#{i}#fused"]["has_gold"] for n in GENP for i in range(left[n]["n"])]
    b = sum(f and not d for d, f in zip(gd, gf))
    c = sum(d and not f for d, f in zip(gd, gf))
    lo, hi = boot_ci(gd, gf)
    pooled_left = {"n": len(gd), "R@k_dense": float(np.mean(gd)), "R@k_fused": float(np.mean(gf)),
                   "d": float(np.mean(gf) - np.mean(gd)), "rescued": b, "lost": c,
                   "mcnemar_p": mcnemar_exact(b, c), "ci95": [lo, hi]}

    print(f"\nstage 2 result (free): pooled over {len(GENP)} generated splits, n={len(gd)}")
    print(f"  gold in top-{a.k}: dense {pooled_left['R@k_dense']:.4f} -> "
          f"fused {pooled_left['R@k_fused']:.4f}  d={pooled_left['d']:+.4f}")
    print(f"  rescued {b}, lost {c}, McNemar p={pooled_left['mcnemar_p']:.4f}, "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]")

    paid = len(work)
    calls = -(-paid // a.batch)
    print(f"\nstage 3 would need {paid} selector items ({len(meta_by_qid)} arm-items, "
          f"{len(meta_by_qid) - paid} deduplicated as identical shortlists)")
    print(f"  ~{calls} calls at --batch {a.batch}")

    if not a.run:
        print("\nDRY RUN. Nothing spent. Re-run with --run to measure the right factor.")
        print(f"Power note: the left factor moves by {pooled_left['d']:+.4f} "
              f"({b} rescued, {c} lost). The selector was measured at +/-6 pts per split "
              f"(§4d, n~48 after truncation), so an end-to-end difference this small needs "
              f"the paired design above to be visible at all.")
        OUT.write_text(json.dumps(
            {"dry_run": True, "k": a.k, "depth": a.depth, "alpha": a.alpha,
             "partition": a.partition, "n": a.n, "matrix": str(MATRIX), "seed": SEED,
             "left_factor": left, "left_factor_pooled": pooled_left,
             "stage3_items": paid, "stage3_calls_est": calls}, indent=2) + "\n")
        print(f"-> {OUT}")
        return 0

    # ── stage 3: the selector ──
    random.Random(SEED).shuffle(work)
    batches = [work[i : i + a.batch] for i in range(0, len(work), a.batch)]
    print(f"\nstage 3  {len(work)} items -> {len(batches)} calls, budget {a.budget_usd}", flush=True)

    answers, t0 = {}, time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(call, build_prompt(bt, by_id)): bt for bt in batches}
        for n, fut in enumerate(as_completed(futs), 1):
            for row in parse(fut.result()):
                if isinstance(row, dict) and "id" in row:
                    answers[str(row["id"])] = row
            if n % 10 == 0 or n == len(futs):
                print(f"  {n}/{len(futs)} calls  {_c['err']} err  {time.time()-t0:.0f}s",
                      file=sys.stderr, flush=True)
            if _c["usd"] >= a.budget_usd:
                print(f"  BUDGET CAP HIT at {_c['usd']:.2f}; remaining work abandoned. "
                      f"Compare final n against what you asked for.", file=sys.stderr)
                break

    # ── score: factored, per arm ──
    res = {}
    for arm in ("dense", "fused"):
        per = {}
        for name in store:
            rows = []
            for i in range(left[name]["n"]):
                m = meta_by_qid[f"{name}#{i}#{arm}"]
                ans = answers.get(m.get("alias_of", f"{name}#{i}#{arm}"))
                if ans is None:
                    continue
                choice = str(ans.get("choice", "none"))
                rows.append({"has_gold": m["has_gold"],
                             "correct": choice in m["gold"],
                             "none": choice == "none",
                             "tied": bool(ans.get("tied"))})
            if not rows:
                continue
            ng = [r for r in rows if r["has_gold"]]
            per[name] = {
                "answered": len(rows),
                "end_to_end": float(np.mean([r["correct"] for r in rows])),
                "gold_in_shortlist": float(np.mean([r["has_gold"] for r in rows])),
                "conditional": float(np.mean([r["correct"] for r in ng])) if ng else None,
                "none_pct": round(100 * float(np.mean([r["none"] for r in rows])), 1),
                "tied_pct": round(100 * float(np.mean([r["tied"] for r in rows])), 1),
            }
        res[arm] = per

    # paired end-to-end comparison on items both arms answered
    pe = {"dense": [], "fused": []}
    for name in GENP:
        for i in range(left[name]["n"]):
            got = {}
            for arm in ("dense", "fused"):
                m = meta_by_qid[f"{name}#{i}#{arm}"]
                ans = answers.get(m.get("alias_of", f"{name}#{i}#{arm}"))
                if ans is not None:
                    got[arm] = str(ans.get("choice", "none")) in m["gold"]
            if len(got) == 2:
                pe["dense"].append(got["dense"])
                pe["fused"].append(got["fused"])
    paired = None
    if pe["dense"]:
        b2 = sum(f and not d for d, f in zip(pe["dense"], pe["fused"]))
        c2 = sum(d and not f for d, f in zip(pe["dense"], pe["fused"]))
        lo2, hi2 = boot_ci(pe["dense"], pe["fused"])
        paired = {"n": len(pe["dense"]),
                  "end_to_end_dense": float(np.mean(pe["dense"])),
                  "end_to_end_fused": float(np.mean(pe["fused"])),
                  "d": float(np.mean(pe["fused"]) - np.mean(pe["dense"])),
                  "fused_only_correct": b2, "dense_only_correct": c2,
                  "mcnemar_p": mcnemar_exact(b2, c2), "ci95": [lo2, hi2]}
        print(f"\npaired end-to-end over the generated splits (n={paired['n']}):")
        print(f"  dense {paired['end_to_end_dense']:.4f} -> fused {paired['end_to_end_fused']:.4f}"
              f"  d={paired['d']:+.4f}  McNemar p={paired['mcnemar_p']:.4f}"
              f"  CI [{lo2:+.4f}, {hi2:+.4f}]")

    OUT.write_text(json.dumps(
        {"dry_run": False, "k": a.k, "depth": a.depth, "alpha": a.alpha,
         "partition": a.partition, "n": a.n, "matrix": str(MATRIX), "seed": SEED,
         "reranker": RERANKER, "selector_calls": _c["calls"], "errors": _c["err"],
         "cost_usd": round(_c["usd"], 4), "budget_hit": _c["usd"] >= a.budget_usd,
         "left_factor": left, "left_factor_pooled": pooled_left,
         "arms": res, "paired_end_to_end": paired}, indent=2) + "\n")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
