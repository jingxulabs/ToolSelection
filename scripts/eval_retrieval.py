"""Evaluate a tool-embedding matrix over every labeled split, MetaTool + generated.

Fills two gaps:
  1. retrieval_baseline.py only reads data/corpus/ splits, so nothing reproduced the
     DATA.md 4c table over data/generated/. This does.
  2. Embedding matrices are row-aligned to the catalog and meaningless without the
     model/text-variant/query-prefix that built them. This reads the .meta.json
     sidecar build_embeddings.py writes and refuses to guess.

  ./.venv/bin/python scripts/eval_retrieval.py \
      --matrix data/eval/tool_embeddings.npy \
      --compare data/eval/tool_embeddings_bge_v3.npy

SELECTION HYGIENE: standard_L2 and splitD_separable were used to pick the encoder and
document variant, so they are dev, not test. Splits are tagged [dev]/[test] below;
quote the [test] rows as the independent result.
"""

import argparse, json
from pathlib import Path

import numpy as np

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

CORPUS = Path("data/corpus")
GEN = Path("data/generated")
OUT = Path("data/eval")
KS = [1, 5, 10, 25, 50, 100]
SEED = 13

# Splits touched while choosing the encoder/text variant -> dev. Everything else is a
# clean confirmation set for this change.
DEV_SPLITS = {"standard_L2", "splitD_separable"}


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


PARTITION = CORPUS / "generated_partition.json"


def splits(partition="all"):
    """name -> list of items with .query and .gold_tool_ids (abstain has empty gold).

    partition='dev'  -> dev_A anchor  + dev-side generated   (iterate here)
    partition='test' -> test_A + C_unseen + test-side generated (one access per arm)
    partition='all'  -> everything, pre-partition behaviour
    """
    part = {}
    if partition != "all":
        if not PARTITION.exists():
            raise SystemExit(f"--partition {partition} needs {PARTITION}; "
                             f"run scripts/split_generated.py first")
        part = json.loads(PARTITION.read_text())["partition"]

    s = {}
    anchors = {"dev": ["dev_A"], "test": ["test_A", "C_unseen_tools"],
               "all": ["test_A", "C_unseen_tools"]}[partition]
    for name in anchors:
        f = CORPUS / f"{name}.jsonl"
        if f.exists():
            s[name] = load_jsonl(f)
    for lvl in ("L2", "L3", "L4"):
        f = GEN / f"standard_{lvl}.jsonl"
        if f.exists():
            s[f"standard_{lvl}"] = load_jsonl(f)
    f = GEN / "confusion_L2.jsonl"
    if f.exists():
        items = load_jsonl(f)
        s["splitD_separable"] = [i for i in items if i.get("separable")]
        s["splitD_inseparable"] = [i for i in items if i.get("separable") is False]
    f = GEN / "abstain_L2.jsonl"
    if f.exists():
        s["splitE_abstain"] = load_jsonl(f)

    if part:
        out = {}
        for k, v in s.items():
            if not k.startswith(("standard", "splitD", "splitE")):
                out[k] = v  # MetaTool anchors carry their own train/dev/test cut
                continue
            kept = [i for i in v if part.get(i.get("intent_id")) == partition]
            if not kept:
                raise SystemExit(f"{k}: 0 items in partition '{partition}' -- stale "
                                 f"{PARTITION}? Re-run scripts/split_generated.py.")
            out[k] = kept
        s = out
    return s


def evaluate(E, model, prefix, split_items, pos, cat, n_max, device):
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model, device=device)
    rng = np.random.default_rng(SEED)
    res = {}
    for name, items in split_items.items():
        if n_max and len(items) > n_max:
            items = [items[i] for i in rng.choice(len(items), n_max, replace=False)]
        Q = m.encode([prefix + it["query"] for it in items], batch_size=128,
                     normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        S = Q @ E.T
        top1 = S.max(axis=1)

        golds = [[pos[g] for g in it["gold_tool_ids"] if g in pos] for it in items]
        if not any(golds):  # abstention split: no gold, only the separability signal
            res[name] = {"n": len(items), "abstain": True,
                         "mean_top1_sim": round(float(top1.mean()), 4)}
            continue

        order = np.argsort(-S, axis=1)
        ranks, thin = [], []
        for i, g in enumerate(golds):
            if not g:
                continue
            r = min(int(np.where(order[i] == j)[0][0]) + 1 for j in g)
            ranks.append(r)
            desc = cat[g[0]].get("description") or ""
            if len(desc.split()) <= 5:
                thin.append(r)
        ranks = np.array(ranks)
        out = {"n": int(len(ranks)), "mean_top1_sim": round(float(top1.mean()), 4)}
        for k in KS:
            out[f"R@{k}"] = round(float((ranks <= k).mean()), 4)
        out["mrr@100"] = round(float(np.mean([1.0 / r if r <= 100 else 0.0 for r in ranks])), 4)
        if thin:
            t = np.array(thin)
            out["thin_n"] = int(len(t))
            out["thin_R@10"] = round(float((t <= 10).mean()), 4)
            out["thin_R@25"] = round(float((t <= 25).mean()), 4)
        res[name] = out
    del m
    return res


def sidecar(matrix, model_override, prefix_override):
    meta_p = Path(matrix).with_suffix(".meta.json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    model = model_override or meta.get("model")
    if not model:
        raise SystemExit(
            f"no {meta_p} and no --model given; cannot know which encoder built {matrix}. "
            f"Rebuild with build_embeddings.py (writes the sidecar) or pass --model.")
    prefix = prefix_override if prefix_override is not None else meta.get("query_prefix", "")
    return model, prefix, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="data/eval/tool_embeddings.npy")
    ap.add_argument("--compare", default=None, help="second matrix; prints deltas vs --matrix")
    ap.add_argument("--model", default=None, help="override the sidecar's model")
    ap.add_argument("--compare-model", default=None)
    ap.add_argument("--query-prefix", default=None)
    ap.add_argument("--compare-query-prefix", default=None)
    ap.add_argument("--n", type=int, default=0, help="cap items per split (0 = all)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--partition", default="all", choices=["all", "dev", "test"],
                    help="dev = iterate freely; test = one access per arm (design 5)")
    ap.add_argument("--out", default=str(OUT / "retrieval_eval.json"))
    a = ap.parse_args()

    cat = [json.loads(l) for l in (CORPUS / "catalog.jsonl").read_text().splitlines() if l.strip()]
    pos = {t["tool_id"]: i for i, t in enumerate(cat)}
    sp = splits(a.partition)

    runs = {}
    for label, mpath, mo, po in (("base", a.matrix, a.model, a.query_prefix),
                                 ("new", a.compare, a.compare_model, a.compare_query_prefix)):
        if not mpath:
            continue
        E = np.load(mpath)
        if E.shape[0] != len(cat):
            raise SystemExit(f"{mpath} has {E.shape[0]} rows but catalog has {len(cat)} "
                             f"-- row alignment is broken, regenerate the matrix.")
        model, prefix, meta = sidecar(mpath, mo, po)
        print(f"[{label}] {mpath}  {E.shape}  model={model}  "
              f"text={meta.get('text_variant','?')}  prefix={prefix!r}")
        runs[label] = {"matrix": mpath, "model": model, "text_variant": meta.get("text_variant"),
                       "results": evaluate(E, model, prefix, sp, pos, cat, a.n, a.device)}

    base, new = runs.get("base"), runs.get("new")
    hdr = f"{'split':22s} {'n':>5s} " + " ".join(f"{'R@'+str(k):>7s}" for k in (1, 10, 25, 100)) + f" {'MRR':>7s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for name in base["results"]:
        b = base["results"][name]
        tag = "[dev] " if name in DEV_SPLITS else "[test]"
        if b.get("abstain"):
            line = f"{tag}{name:16s} {b['n']:5d}   (no gold) mean top-1 sim {b['mean_top1_sim']:.3f}"
            if new:
                line += f" -> {new['results'][name]['mean_top1_sim']:.3f}"
            print(line)
            continue
        print(f"{tag}{name:16s} {b['n']:5d} " +
              " ".join(f"{b['R@'+str(k)]:7.3f}" for k in (1, 10, 25, 100)) + f" {b['mrr@100']:7.3f}")
        if new:
            nw = new["results"][name]
            print(f"{'':22s} {'':5s} " +
                  " ".join(f"{nw['R@'+str(k)]:7.3f}" for k in (1, 10, 25, 100)) + f" {nw['mrr@100']:7.3f}"
                  + "   <- new")
            print(f"{'':22s} {'Δ':>5s} " +
                  " ".join(f"{nw['R@'+str(k)] - b['R@'+str(k)]:+7.3f}" for k in (1, 10, 25, 100))
                  + f" {nw['mrr@100'] - b['mrr@100']:+7.3f}")
            if "thin_R@10" in b:
                print(f"{'':22s} thin n={b['thin_n']:<4d} R@10 {b['thin_R@10']:.3f} -> "
                      f"{nw['thin_R@10']:.3f} ({nw['thin_R@10'] - b['thin_R@10']:+.3f})")
        print()

    Path(a.out).write_text(json.dumps(runs, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
