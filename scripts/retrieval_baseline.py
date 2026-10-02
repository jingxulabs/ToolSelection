"""Arm 0/3 retrieval baseline + the distractor-pooling degradation curve.

Two things this establishes before any model is trained:
  1. Does pooling actually make retrieval load-bearing? (recall@k vs pool size)
  2. Which tools are confusable? -> writes the similarity graph used for split D.
"""

import argparse, json, math, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

CORPUS = Path("data/corpus")
OUT = Path("data/eval")
SEED = 13
KS = [1, 5, 10, 25, 50, 100]
POOLS = [199, 500, 1000, 2000, 3551]


def tool_text(t):
    parts = [t["name"], t.get("summary", ""), t["description"]]
    p = t.get("parameters", {}).get("properties", {})
    if p:
        parts.append(" ".join(list(p)[:15]))
    if t.get("tags"):
        parts.append(" ".join(t["tags"]))
    return " ".join(x for x in parts if x).strip()


def tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [tok(d) for d in docs]
        self.n = len(self.docs)
        self.len = np.array([len(d) for d in self.docs], dtype=np.float32)
        self.avg = float(self.len.mean()) or 1.0
        self.tf = [Counter(d) for d in self.docs]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        self.idf = {w: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.post = defaultdict(list)
        for i, c in enumerate(self.tf):
            for w, f in c.items():
                self.post[w].append((i, f))

    def scores(self, q):
        s = np.zeros(self.n, dtype=np.float32)
        for w in tok(q):
            idf = self.idf.get(w)
            if idf is None:
                continue
            for i, f in self.post[w]:
                s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


def recall_at_k(rank_lists, golds, ks):
    out = {}
    for k in ks:
        hits = sum(1 for r, g in zip(rank_lists, golds) if any(x in g for x in r[:k]))
        out[f"recall@{k}"] = round(hits / max(len(golds), 1), 4)
    mrr = 0.0
    for r, g in zip(rank_lists, golds):
        for i, x in enumerate(r[:100], 1):
            if x in g:
                mrr += 1 / i
                break
    out["mrr@100"] = round(mrr / max(len(golds), 1), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test_A")
    ap.add_argument("--n", type=int, default=1500, help="sample size per split")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    catalog = [json.loads(l) for l in (CORPUS / "catalog.jsonl").read_text().splitlines() if l.strip()]
    ids = [t["tool_id"] for t in catalog]
    idx = {t: i for i, t in enumerate(ids)}
    texts = [tool_text(t) for t in catalog]
    labeled = {t["tool_id"] for t in catalog if t["source"] == "metatool"}

    from sentence_transformers import SentenceTransformer

    print(f"embedding {len(texts)} tools ...", file=sys.stderr)
    model = SentenceTransformer(a.model)
    T = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    bm25 = BM25(texts)

    results = {}
    for split in [a.split, "C_unseen_tools"]:
        f = CORPUS / f"{split}.jsonl"
        if not f.exists():
            continue
        items = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if len(items) > a.n:
            items = rng.sample(items, a.n)
        queries = [it["query"] for it in items]
        golds = [set(it["gold_tool_ids"]) for it in items]
        Q = model.encode(queries, batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)

        split_res = {}
        for pool in POOLS:
            # pool = all labeled tools + distractors sampled up to `pool` total
            keep = set(labeled)
            extra = [i for i in ids if i not in labeled]
            rng2 = random.Random(SEED)
            rng2.shuffle(extra)
            keep.update(extra[: max(0, pool - len(labeled))])
            cols = np.array([idx[i] for i in ids if i in keep])
            sub_ids = [ids[c] for c in cols]

            sims = Q @ T[cols].T
            order = np.argsort(-sims, axis=1)[:, :100]
            dense_ranks = [[sub_ids[j] for j in row] for row in order]

            bm_ranks = []
            for q in queries:
                s = bm25.scores(q)[cols]
                bm_ranks.append([sub_ids[j] for j in np.argsort(-s)[:100]])

            split_res[f"pool_{len(sub_ids)}"] = {
                "dense": recall_at_k(dense_ranks, golds, KS),
                "bm25": recall_at_k(bm_ranks, golds, KS),
            }
        results[split] = {"n": len(items), "pools": split_res}

    # --- confusability graph for split D
    print("building confusability graph ...", file=sys.stderr)
    S = T @ T.T
    np.fill_diagonal(S, -1)
    pairs = []
    lab_idx = [idx[i] for i in labeled]
    for i in lab_idx:
        j = int(np.argmax(S[i]))
        if S[i, j] >= 0.75:
            pairs.append({"a": ids[i], "b": ids[j], "sim": round(float(S[i, j]), 4)})
    pairs.sort(key=lambda p: -p["sim"])
    (OUT / "confusion_pairs.json").write_text(json.dumps(pairs, indent=2))

    results["confusion_pairs"] = {
        "n_pairs_over_0.75": len(pairs),
        "n_over_0.85": sum(1 for p in pairs if p["sim"] >= 0.85),
        "top5": pairs[:5],
    }
    (OUT / "retrieval_baseline.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
