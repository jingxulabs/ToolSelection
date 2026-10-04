"""Generate user intents over the APIs.guru catalog (design §4.2).

Modes
-----
standard  : one natural intent per tool, difficulty L2-L4. Splits A/B.
confusion : split D. Gives the model a tool AND its nearest confusable neighbour
            and asks for a request only the target satisfies. Crucially the model
            may return separable=false -- those pairs are genuinely ambiguous and
            become set-valued gold (variant F) instead of being scored as errors.
            This is what keeps generated labels honest on near-misses.
abstain   : split E. In-domain intents that NO catalog tool can satisfy.

Cost note: the `claude` CLI pays ~27k cached prompt tokens per call. A cold call runs
roughly 7.5x a warm one (<5 min TTL). Run continuously -- do not trickle.
"""

import argparse, json, os, random, re, subprocess, sys, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

CORPUS = Path("data/corpus")
OUT = Path("data/generated")
_lock = threading.Lock()
_cost = {"usd": 0.0, "calls": 0, "errors": 0}

LEVELS = {
    "L2": "natural everyday phrasing, states the goal plainly",
    "L3": "indirect - describes the underlying problem or situation, never the action",
    "L4": "includes distracting context or an unrelated aside before the real ask",
}


def call_claude(prompt, timeout=300):
    try:
        p = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json"],
            capture_output=True, text=True, timeout=timeout,
        )
        d = json.loads(p.stdout)
    except Exception as e:
        with _lock:
            _cost["errors"] += 1
        return None
    with _lock:
        _cost["usd"] += d.get("total_cost_usd") or 0.0
        _cost["calls"] += 1
        if d.get("is_error"):
            _cost["errors"] += 1
    return None if d.get("is_error") else d.get("result", "")


def parse_json_array(txt):
    if not txt:
        return []
    m = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S)
    if m:
        txt = m.group(1)
    i, j = txt.find("["), txt.rfind("]")
    if i < 0 or j < 0:
        return []
    try:
        out = json.loads(txt[i : j + 1])
        return out if isinstance(out, list) else []
    except json.JSONDecodeError:
        return []


def fmt_tool(t, with_params=True):
    p = ", ".join(list(t["parameters"]["properties"])[:8])
    s = f"  tool_id: {t['tool_id']}\n  name: {t['name']}\n  service: {t['api_title'][:60]}\n  desc: {(t['description'] or t.get('summary') or '(no description)')[:260]}"
    if with_params and p:
        s += f"\n  params: {p}"
    return s


def prompt_standard(batch, level):
    head = [
        f"For each API operation below write ONE realistic user request it satisfies.",
        f"Style: {LEVELS[level]}.",
        "Hard rules: speak as an end user with a goal; NEVER name the API, operation, service or provider;",
        "never reuse its parameter names; include plausible concrete values where a real user would.",
        "If an operation is too vague to write a distinguishing request for, set \"skip\": true.",
        'Output ONLY a JSON array of {"tool_id":..., "query":..., "skip":false}. No prose.',
        "",
    ]
    return "\n".join(head + [fmt_tool(t) for t in batch])


def prompt_confusion(pairs):
    head = [
        "Each item below is a PAIR of similar API operations: a TARGET and a NEIGHBOUR.",
        "Write a user request that the TARGET satisfies and the NEIGHBOUR does NOT.",
        "Speak as an end user; never name the API, operation, service or parameters.",
        "",
        "IMPORTANT: if the two operations are genuinely interchangeable for any request a real",
        'user would make, set "separable": false and still give your best query. Do not invent a',
        "spurious distinction. Honesty here matters more than volume.",
        'Output ONLY a JSON array of {"tool_id":<TARGET id>,"query":...,"separable":true|false,'
        '"distinction":"<=12 words on what separates them, or why not"}. No prose.',
        "",
    ]
    body = []
    for tgt, nb, sim in pairs:
        body.append(f"- TARGET:\n{fmt_tool(tgt)}\n  NEIGHBOUR (sim={sim:.2f}):\n{fmt_tool(nb)}\n")
    return "\n".join(head + body)


def prompt_abstain(batch):
    head = [
        "Below are real API operations from a catalog, shown so you know what the catalog COVERS.",
        "Write user requests in these same domains that the catalog CANNOT satisfy -- adjacent asks",
        "that plausibly sound in-scope but need a capability none of these operations provide.",
        "Rules: realistic phrasing; stay in the same subject area; do NOT be absurd or off-topic",
        "(no 'what is the meaning of life'). The point is near-miss out-of-scope, not obvious nonsense.",
        f"Write {len(batch)} requests.",
        'Output ONLY a JSON array of {"query":...,"why_unsupported":"<=12 words"}. No prose.',
        "",
    ]
    return "\n".join(head + [fmt_tool(t, with_params=False) for t in batch])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["standard", "confusion", "abstain"], required=True)
    ap.add_argument("--n-tools", type=int, default=60)
    ap.add_argument("--batch", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--level", default="L2", choices=list(LEVELS))
    ap.add_argument("--min-sim", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget-usd", type=float, default=5.0, help="hard stop")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    catalog = [json.loads(l) for l in (CORPUS / "catalog.jsonl").read_text().splitlines() if l.strip()]
    guru = [t for t in catalog if t["source"] == "apis.guru"]
    by_id = {t["tool_id"]: t for t in catalog}

    # ---- build work units
    if a.mode == "confusion":
        import numpy as np

        E = np.load("data/eval/tool_embeddings.npy")
        pos = {t["tool_id"]: i for i, t in enumerate(catalog)}
        gi = [pos[t["tool_id"]] for t in guru]
        S = E[gi] @ E[gi].T
        np.fill_diagonal(S, -1)
        nn = S.argmax(axis=1)
        cands = [
            (guru[i], guru[int(nn[i])], float(S[i, nn[i]]))
            for i in range(len(guru))
            if S[i, nn[i]] >= a.min_sim
        ]
        rng.shuffle(cands)
        cands = cands[: a.n_tools]
        units = [cands[i : i + a.batch] for i in range(0, len(cands), a.batch)]
        print(f"confusion: {len(cands)} pairs >= {a.min_sim} -> {len(units)} calls", file=sys.stderr)
    else:
        pool = [t for t in guru if t["doc_quality"] >= 2] if a.mode == "standard" else guru
        rng.shuffle(pool)
        pool = pool[: a.n_tools]
        units = [pool[i : i + a.batch] for i in range(0, len(pool), a.batch)]
        print(f"{a.mode}: {len(pool)} tools -> {len(units)} calls", file=sys.stderr)

    builder = {"standard": lambda u: prompt_standard(u, a.level), "confusion": prompt_confusion,
               "abstain": prompt_abstain}[a.mode]

    def work(unit):
        if _cost["usd"] >= a.budget_usd:
            return []
        txt = call_claude(builder(unit))
        rows = parse_json_array(txt)
        ids = {(x[0]["tool_id"] if a.mode == "confusion" else x["tool_id"]) for x in unit} if a.mode != "abstain" else None
        out = []
        for r in rows:
            if not isinstance(r, dict) or not (r.get("query") or "").strip():
                continue
            if a.mode == "abstain":
                out.append({"query": r["query"].strip(), "gold_tool_ids": [],
                            "task_variant": "unsupported", "difficulty": "L3",
                            "why_unsupported": r.get("why_unsupported", ""), "source": "gen_abstain"})
                continue
            tid = r.get("tool_id")
            if tid not in ids or r.get("skip"):
                continue
            sep = r.get("separable", True)
            out.append({
                "query": r["query"].strip(),
                "gold_tool_ids": [tid],
                "task_variant": "unambiguous" if sep else "ambiguous",
                "difficulty": a.level if a.mode == "standard" else "L5",
                "separable": bool(sep),
                "distinction": r.get("distinction", ""),
                "source": f"gen_{a.mode}",
            })
        return out

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, u) for u in units]
        for n, f in enumerate(as_completed(futs), 1):
            results.extend(f.result())
            if n % 5 == 0 or n == len(futs):
                print(f"  {n}/{len(futs)} calls  {len(results)} intents  ${_cost['usd']:.2f}  "
                      f"{_cost['errors']} err  {time.time()-t0:.0f}s", file=sys.stderr)

    for i, r in enumerate(results):
        r["intent_id"] = f"{a.mode}_{a.level}_{i}"
        r["provenance"] = "claude-opus-5-generated"
    path = Path(a.out) if a.out else OUT / f"{a.mode}_{a.level}.jsonl"
    with path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    stats = {
        "mode": a.mode, "level": a.level, "intents": len(results), "calls": _cost["calls"],
        "errors": _cost["errors"], "cost_usd": round(_cost["usd"], 4),
        "cost_per_intent": round(_cost["usd"] / max(len(results), 1), 5),
        "elapsed_s": round(time.time() - t0),
        "out": str(path),
    }
    if a.mode == "confusion":
        sep = sum(1 for r in results if r.get("separable"))
        stats["separable"] = sep
        stats["inseparable"] = len(results) - sep
        stats["inseparable_pct"] = round(100 * (len(results) - sep) / max(len(results), 1), 1)
    print(json.dumps(stats, indent=2))
    (OUT / f"{a.mode}_{a.level}_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
