"""Round-trip validity check (design §4.2) -- and the first real S2 measurement.

The question: generated splits score R@1 0.13-0.18 under dense retrieval vs 0.35 for
MetaTool. Is that legitimate difficulty, or intents that don't pin down their gold tool?

Test: give a model the query plus a shortlist that is GUARANTEED to contain gold, and
ask it to choose. This is oracle-retrieval selection (design arm-infinity), so:
  high accuracy  -> intents are valid; dense retrieval is the weak stage (fix S1)
  low accuracy   -> intents under-specify their tool; labels are noisy (regenerate)

Distinguishing these two is the difference between "invest in the retriever" and
"throw away the dataset", so it runs before any arm is built.
"""

import argparse, json, random, re, subprocess, sys, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

CORPUS, GEN, OUT = Path("data/corpus"), Path("data/generated"), Path("data/eval")
_lock = threading.Lock()
_c = {"usd": 0.0, "calls": 0, "err": 0}


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
    """One call scores several items; each carries its own candidate list."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-split", type=int, default=75)
    ap.add_argument("--k", type=int, default=20, help="shortlist size (gold forced in)")
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--budget-usd", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    cat = [json.loads(l) for l in (CORPUS / "catalog.jsonl").read_text().splitlines() if l.strip()]
    by_id = {t["tool_id"]: t for t in cat}
    ids = [t["tool_id"] for t in cat]
    E = np.load("data/eval/tool_embeddings.npy")
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    sources = {
        "splitD_separable": (GEN / "confusion_L2.jsonl", lambda r: r.get("separable")),
        "splitD_inseparable": (GEN / "confusion_L2.jsonl", lambda r: not r.get("separable")),
        "standard_L2": (GEN / "standard_L2.jsonl", None),
        "standard_L3": (GEN / "standard_L3.jsonl", None),
        "metatool_test": (CORPUS / "test_A.jsonl", None),
        "splitE_abstain": (GEN / "abstain_L2.jsonl", None),
    }

    work, meta = [], {}
    for name, (path, filt) in sources.items():
        if not path.exists():
            continue
        rs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if filt:
            rs = [r for r in rs if filt(r)]
        rng.shuffle(rs)
        rs = rs[: a.n_per_split]
        Q = st.encode([r["query"] for r in rs], normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        order = np.argsort(-(Q @ E.T), axis=1)
        for n, r in enumerate(rs):
            gold = set(r["gold_tool_ids"])
            top = [ids[j] for j in order[n, : a.k]]
            if gold:  # force gold into the shortlist -> oracle retrieval
                cands = [c for c in top if c not in gold][: a.k - len(gold)] + list(gold)
            else:  # abstention: no gold exists, shortlist is all distractors
                cands = top[: a.k]
            rng.shuffle(cands)
            qid = f"{name}#{n}"
            meta[qid] = {"split": name, "gold": gold, "query": r["query"]}
            work.append({"qid": qid, "query": r["query"], "cands": cands})

    rng.shuffle(work)
    batches = [work[i : i + a.batch] for i in range(0, len(work), a.batch)]
    print(f"{len(work)} items -> {len(batches)} calls (k={a.k})", file=sys.stderr)

    answers = {}

    def run(b):
        if _c["usd"] >= a.budget_usd:
            return {}
        rows = parse(call(build_prompt(b, by_id)))
        got = {}
        want = {x["qid"] for x in b}
        for r in rows:
            if isinstance(r, dict) and r.get("id") in want:
                got[r["id"]] = r
        return got

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(run, b) for b in batches]
        for n, f in enumerate(as_completed(futs), 1):
            answers.update(f.result())
            if n % 10 == 0 or n == len(futs):
                print(f"  {n}/{len(futs)} calls  ${_c['usd']:.2f}  {_c['err']} err  {time.time()-t0:.0f}s",
                      file=sys.stderr)

    agg = defaultdict(lambda: {"n": 0, "correct": 0, "none": 0, "tied": 0, "missing": 0})
    for qid, m in meta.items():
        s = agg[m["split"]]
        s["n"] += 1
        r = answers.get(qid)
        if r is None:
            s["missing"] += 1
            continue
        ch = (r.get("choice") or "").strip()
        if r.get("tied"):
            s["tied"] += 1
        if ch == "none" or not ch:
            s["none"] += 1
            if not m["gold"]:
                s["correct"] += 1  # correct abstention
        elif ch in m["gold"]:
            s["correct"] += 1

    print(f"\n{'split':<22}{'n':>5}{'answered':>10}{'acc':>8}{'none%':>8}{'tied%':>8}")
    res = {}
    for k, s in sorted(agg.items()):
        ans = s["n"] - s["missing"]
        acc = s["correct"] / max(ans, 1)
        res[k] = {**s, "answered": ans, "accuracy": round(acc, 4),
                  "none_pct": round(100 * s["none"] / max(ans, 1), 1),
                  "tied_pct": round(100 * s["tied"] / max(ans, 1), 1)}
        print(f"{k:<22}{s['n']:>5}{ans:>10}{acc:>8.3f}{res[k]['none_pct']:>8.1f}{res[k]['tied_pct']:>8.1f}")
    out = {"k": a.k, "cost_usd": round(_c["usd"], 4), "calls": _c["calls"], "errors": _c["err"],
           "elapsed_s": round(time.time() - t0), "splits": res}
    (OUT / "roundtrip_check.json").write_text(json.dumps(out, indent=2))
    print(f"\ncost ${_c['usd']:.2f}  calls {_c['calls']}  errors {_c['err']}")


if __name__ == "__main__":
    main()
