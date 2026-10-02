"""Unify labeled intents + a large distractor catalog, then cut the splits.

MetaTool supplies 20.6k human-plausible (query -> tool) pairs over 199 tools, but
199 candidates is too small to make retrieval load-bearing. We pool the APIs.guru
catalog in as distractors so selection happens over 3.5k+ tools (design §4.0).

Splits follow §5: tools are held out *before* intents, so split C measures real
generalization to catalog additions rather than memorized tool identities.
"""

import csv, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

CAT = Path("data/catalog")
META = Path("data/raw/MetaTool/dataset")
OUT = Path("data/corpus")
SEED = 13
HELDOUT_TOOL_FRAC = 0.20
DEV_FRAC, TEST_FRAC = 0.12, 0.20


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def load_metatool():
    """199 plugin tools + their labeled queries."""
    des = json.loads((META / "plugin_des.json").read_text())
    info = {}
    try:
        raw = json.loads((META / "plugin_info.json").read_text())
        items = raw if isinstance(raw, list) else list(raw.values())
        for it in items:
            if isinstance(it, dict):
                key = norm(it.get("name_for_model") or it.get("name") or it.get("nameForModel") or "")
                if key:
                    info[key] = it
    except Exception:
        pass

    tools, by_norm = [], {}
    for name, desc in des.items():
        extra = info.get(norm(name), {})
        t = {
            "tool_id": f"metatool::{norm(name)}",
            "name": name,
            "description": (desc or "").strip(),
            "summary": (extra.get("description_for_human") or "")[:300],
            "parameters": {"type": "object", "properties": {}, "required": []},
            "returns": [],
            "auth_scope": [],
            "side_effects": "read",
            "domain": "plugin",
            "categories": ["plugin"],
            "provider": "openai_plugin_store",
            "api_title": name,
            "http": {},
            "tags": [],
            "deprecated": False,
            "source": "metatool",
            "spec_url": "",
        }
        d, p = t["description"], t["parameters"]["properties"]
        t["doc_quality"] = (1 if len(d) >= 20 else 0) + (1 if len(d) >= 120 else 0)
        tools.append(t)
        by_norm[norm(name)] = t["tool_id"]

    intents, unmatched = [], 0
    with (META / "data" / "all_clean_data.csv").open() as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            gold = by_norm.get(norm(row.get("Tool")))
            if not gold:
                unmatched += 1
                continue
            q = (row.get("Query") or "").strip()
            if len(q) < 8:
                continue
            intents.append(
                {
                    "intent_id": f"mt_{i}",
                    "query": q,
                    "gold_tool_ids": [gold],
                    "task_variant": "unambiguous",
                    "difficulty": "L2",
                    "source": "metatool_single",
                    "provenance": "chatgpt_gpt4_generated",
                }
            )

    # multi-tool queries -> ambiguous/multi variant
    try:
        multi = json.loads((META / "data" / "multi_tool_query_golden.json").read_text())
        items = multi if isinstance(multi, list) else list(multi.values())
        for j, it in enumerate(items):
            if not isinstance(it, dict):
                continue
            q = (it.get("Query") or it.get("query") or "").strip()
            raw_tools = it.get("Tool") or it.get("tools") or it.get("golden") or []
            if isinstance(raw_tools, str):
                raw_tools = [raw_tools]
            gold = [by_norm[norm(t)] for t in raw_tools if norm(t) in by_norm]
            if q and len(gold) >= 2:
                intents.append(
                    {
                        "intent_id": f"mtm_{j}",
                        "query": q,
                        "gold_tool_ids": gold,
                        "task_variant": "multi",
                        "difficulty": "L5",
                        "source": "metatool_multi",
                        "provenance": "chatgpt_gpt4_generated",
                    }
                )
    except Exception as e:
        print(f"  (multi-tool load skipped: {type(e).__name__})", file=sys.stderr)

    return tools, intents, unmatched


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    guru = [json.loads(l) for l in (CAT / "catalog.jsonl").read_text().splitlines() if l.strip()]
    mt_tools, intents, unmatched = load_metatool()
    catalog = mt_tools + guru
    print(f"catalog: {len(mt_tools)} metatool + {len(guru)} apis.guru = {len(catalog)} tools", file=sys.stderr)
    print(f"intents: {len(intents)} ({unmatched} unmatched gold dropped)", file=sys.stderr)

    # --- dedup near-identical queries BEFORE splitting (design §5: paraphrase leakage)
    seen, deduped = set(), []
    for it in intents:
        k = re.sub(r"[^a-z0-9 ]+", "", it["query"].lower())
        k = " ".join(sorted(k.split()))
        if k in seen:
            continue
        seen.add(k)
        deduped.append(it)
    print(f"  after dedup: {len(deduped)} (-{len(intents)-len(deduped)})", file=sys.stderr)
    intents = deduped

    # --- split by TOOL first (split C = unseen tools)
    labeled_tools = sorted({g for it in intents for g in it["gold_tool_ids"]})
    rng.shuffle(labeled_tools)
    n_hold = int(len(labeled_tools) * HELDOUT_TOOL_FRAC)
    heldout_tools = set(labeled_tools[:n_hold])
    print(f"  held-out tools: {len(heldout_tools)}/{len(labeled_tools)}", file=sys.stderr)

    splits = defaultdict(list)
    for it in intents:
        it["uses_heldout_tool"] = any(g in heldout_tools for g in it["gold_tool_ids"])
        if it["uses_heldout_tool"]:
            splits["C_unseen_tools"].append(it)
        elif it["task_variant"] == "multi":
            splits["F_ambiguous_multi"].append(it)
        else:
            r = rng.random()
            splits["test_A" if r < TEST_FRAC else "dev_A" if r < TEST_FRAC + DEV_FRAC else "train"].append(it)

    # --- write
    with (OUT / "catalog.jsonl").open("w") as fh:
        for t in catalog:
            t["is_heldout"] = t["tool_id"] in heldout_tools
            fh.write(json.dumps(t) + "\n")
    for name, items in splits.items():
        with (OUT / f"{name}.jsonl").open("w") as fh:
            for it in items:
                fh.write(json.dumps(it) + "\n")

    stats = {
        "catalog_total": len(catalog),
        "catalog_by_source": dict(Counter(t["source"] for t in catalog)),
        "labeled_tools": len(labeled_tools),
        "heldout_tools": len(heldout_tools),
        "splits": {k: len(v) for k, v in sorted(splits.items())},
        "intents_total": sum(len(v) for v in splits.values()),
        "distractor_ratio": round(len(catalog) / max(len(labeled_tools), 1), 1),
        "seed": SEED,
        "notes": [
            "Splits A/C only. Splits D (confusion), E (abstention), G (drift) are built by "
            "make_hard_splits.py once embeddings exist.",
            "All MetaTool queries are LLM-generated; human validation of 500 still required (design 4.3).",
        ],
    }
    (OUT / "corpus_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
