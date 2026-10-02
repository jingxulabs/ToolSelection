"""Cut a dev/test partition over data/generated/ -> data/corpus/generated_partition.json.

data/generated/ shipped as one undifferentiated pool, so every config choice made by
looking at it erodes it (EXPERIMENT_DESIGN 5: "freeze test, one evaluation per arm").
This cuts the partition once, by the design's rules:

  1. BY TOOL FIRST, then by intent. A random intent split leaks tool identity.
     Because standard L2/L3/L4 reuse the SAME 1,000 tools, tool-level assignment also
     keeps a tool's three difficulty levels on the same side -- the paired L2/L3/L4
     comparison survives inside each partition.

  2. CONFUSION TWINS TRAVEL TOGETHER. Split D items were generated against a (target,
     nearest-neighbour) pair. If the target lands in dev and its twin in test, the
     discrimination signal leaks across the boundary. The neighbour identity was never
     persisted, so it is reconstructed here exactly as generate_intents.py computed it
     (argmax cosine over APIs.guru rows of the SHIPPED v0 matrix, sim >= --min-sim),
     then union-found so each confusable component is assigned as a unit.

  3. DEDUP ACROSS THE BOUNDARY (cosine > 0.95 on queries). Paraphrase leakage is the
     silent killer in synthetic sets. Dev-side copies are dropped, so test stays intact.

Writes a manifest, not new copies of the data: intent_id -> dev|test|dropped.
Consume it with eval_retrieval.py --partition {dev,test}.
"""

import argparse, json, random
from pathlib import Path

import numpy as np

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

CORPUS = Path("data/corpus")
GEN = Path("data/generated")
OUT = CORPUS / "generated_partition.json"
GEN_FILES = ["standard_L2", "standard_L3", "standard_L4", "confusion_L2", "abstain_L2"]


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-frac", type=float, default=0.33,
                    help="target share of intents in dev (design 4.4: 1.5k dev / 3k test)")
    ap.add_argument("--min-sim", type=float, default=0.82,
                    help="must match the --min-sim split D was generated with")
    ap.add_argument("--dedup-sim", type=float, default=0.95)
    ap.add_argument("--matrix", default="data/eval/tool_embeddings.npy",
                    help="SHIPPED v0 matrix -- the geometry the confusion pairs were cut under")
    ap.add_argument("--dedup-model", default="thenlper/gte-large")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    rng = random.Random(a.seed)
    cat = [json.loads(l) for l in (CORPUS / "catalog.jsonl").read_text().splitlines() if l.strip()]
    pos = {t["tool_id"]: i for i, t in enumerate(cat)}

    items = {}
    for name in GEN_FILES:
        f = GEN / f"{name}.jsonl"
        if f.exists():
            items[name] = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    total = sum(len(v) for v in items.values())
    print(f"{total} generated intents across {len(items)} files")

    # --- 2. reconstruct the confusion pairing and union-find confusable components
    E = np.load(a.matrix)
    if E.shape[0] != len(cat):
        raise SystemExit(f"{a.matrix}: {E.shape[0]} rows != {len(cat)} catalog rows")
    guru = [i for i, t in enumerate(cat) if t["source"] == "apis.guru"]
    G = E[guru]
    S = G @ G.T
    np.fill_diagonal(S, -1.0)
    nn = S.argmax(axis=1)
    dsu = DSU()
    n_pairs = 0
    for k, i in enumerate(guru):
        j = int(nn[k])
        if S[k, j] >= a.min_sim:
            dsu.union(cat[i]["tool_id"], cat[guru[j]]["tool_id"])
            n_pairs += 1
    print(f"confusable pairs >= {a.min_sim}: {n_pairs} -> union-found into components")

    # --- 1. group intents by the component of their gold tool
    by_comp = {}
    anchorless = []
    for name, its in items.items():
        for it in its:
            gold = [g for g in it["gold_tool_ids"] if g in pos]
            if not gold:  # abstention split has no gold tool to anchor on
                anchorless.append((name, it))
                continue
            by_comp.setdefault(dsu.find(gold[0]), []).append((name, it))

    comps = list(by_comp)
    rng.shuffle(comps)
    target = a.dev_frac * (total - len(anchorless))
    part, dev_n = {}, 0
    for c in comps:
        side = "dev" if dev_n < target else "test"
        for name, it in by_comp[c]:
            part[it["intent_id"]] = side
        if side == "dev":
            dev_n += len(by_comp[c])
    # abstention items have no tool anchor -> plain random assignment
    rng.shuffle(anchorless)
    cut = int(a.dev_frac * len(anchorless))
    for i, (name, it) in enumerate(anchorless):
        part[it["intent_id"]] = "dev" if i < cut else "test"

    # --- 3. dedup across the boundary
    flat = [(name, it) for name, its in items.items() for it in its]
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(a.dedup_model, device=a.device)
    Q = m.encode([it["query"] for _, it in flat], batch_size=128,
                 normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    sides = np.array([part[it["intent_id"]] == "dev" for _, it in flat])
    dropped = set()
    SS = Q @ Q.T
    np.fill_diagonal(SS, -1.0)
    hits = np.argwhere(SS > a.dedup_sim)
    for i, j in hits:
        if i >= j or sides[i] == sides[j]:
            continue
        d = i if sides[i] else j  # drop the dev-side copy; test stays intact
        dropped.add(flat[d][1]["intent_id"])
    for iid in dropped:
        part[iid] = "dropped"
    print(f"cross-boundary near-duplicates > {a.dedup_sim}: {len(dropped)} dev-side intents dropped")

    # --- stats
    stats = {}
    for name, its in items.items():
        c = {"dev": 0, "test": 0, "dropped": 0}
        for it in its:
            c[part[it["intent_id"]]] += 1
        stats[name] = c
    print(f"\n{'file':16s} {'dev':>6s} {'test':>6s} {'dropped':>8s}")
    for name, c in stats.items():
        print(f"{name:16s} {c['dev']:6d} {c['test']:6d} {c['dropped']:8d}")
    tot = {k: sum(c[k] for c in stats.values()) for k in ("dev", "test", "dropped")}
    print(f"{'TOTAL':16s} {tot['dev']:6d} {tot['test']:6d} {tot['dropped']:8d}")

    # tool-level disjointness check: a gold tool must not appear on both sides
    side_of = {}
    bad = set()
    for name, its in items.items():
        for it in its:
            s = part[it["intent_id"]]
            if s == "dropped":
                continue
            for g in it["gold_tool_ids"]:
                if side_of.setdefault(g, s) != s:
                    bad.add(g)
    print(f"\ntools appearing on BOTH sides: {len(bad)} (must be 0)")

    Path(a.out).write_text(json.dumps({
        "seed": a.seed, "dev_frac": a.dev_frac, "min_sim": a.min_sim,
        "dedup_sim": a.dedup_sim, "dedup_model": a.dedup_model,
        "pair_matrix": a.matrix, "n_confusable_pairs": n_pairs,
        "stats": stats, "totals": tot, "tools_on_both_sides": sorted(bad),
        "partition": part,
    }, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
