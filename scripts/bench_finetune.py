"""Wall-clock benchmark for retriever fine-tuning on MPS. Measures, does not estimate.

Answers "how long would a retriever fine-tune actually take on this machine" by running
real MNRL training steps at several (model, batch, seq_len) settings and timing them, then
extrapolating to full epochs over data/corpus/train.jsonl.

Reports the degradation CLAUDE.md warns about: MPS throughput drops ~40% over an hour as
the machine heats up, so a cold per-step time extrapolates optimistically. The --soak
mode measures the drift directly instead of assuming a number.

  ./.venv/bin/python scripts/bench_finetune.py --steps 12
  ./.venv/bin/python scripts/bench_finetune.py --soak 300   # thermal drift over 5 min
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
from eval_retrieval import CORPUS

CONFIGS = [
    ("thenlper/gte-large", 16, 256),
    ("thenlper/gte-large", 32, 256),
    ("thenlper/gte-base", 32, 256),
    ("thenlper/gte-base", 64, 256),
]


def load_pairs(n=4096):
    cat = {t["tool_id"]: t for t in
           (json.loads(l) for l in (CORPUS / "catalog.jsonl").read_text().splitlines() if l.strip())}
    out = []
    for line in (CORPUS / "train.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        it = json.loads(line)
        g = [x for x in it["gold_tool_ids"] if x in cat]
        if g:
            out.append({"anchor": it["query"], "positive": tool_text(cat[g[0]], "v3")})
        if len(out) >= n:
            break
    return out


def bench(model_name, batch, seq, steps, pairs, device):
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers import SentenceTransformerTrainingArguments as TA
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss

    m = SentenceTransformer(model_name, device=device)
    m.max_seq_length = seq
    # MUST cast: transformers 5.x loads gte-* in their native fp16, and timing an fp16
    # "training" run measures something that diverges to all-NaN weights within a few
    # steps -- i.e. not a usable estimate of real (fp32) training cost. The first version
    # of this benchmark did exactly that and under-reported step time by ~4x.
    if next(m.parameters()).dtype != torch.float32:
        m = m.float()
    n_params = sum(p.numel() for p in m.parameters())
    ds = Dataset.from_list(pairs[: batch * (steps + 2)])
    args = TA(output_dir="/tmp/bench_ft", per_device_train_batch_size=batch,
              max_steps=steps, logging_steps=10**6, save_strategy="no",
              report_to=[], dataloader_num_workers=0, learning_rate=2e-5,
              warmup_steps=0, fp16=False, bf16=False)  # device comes from the model
    trainer = SentenceTransformerTrainer(model=m, args=args, train_dataset=ds,
                                         loss=MultipleNegativesRankingLoss(m))
    t0 = time.time()
    trainer.train()
    el = time.time() - t0
    del trainer, m
    if device == "mps":
        torch.mps.empty_cache()
    return {"model": model_name, "params_M": round(n_params / 1e6, 1), "batch": batch,
            "seq": seq, "steps": steps, "total_s": round(el, 1),
            "s_per_step": round(el / steps, 3),
            "pairs_per_s": round(batch * steps / el, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--soak", type=int, default=0,
                    help="seconds of continuous training to measure thermal drift")
    ap.add_argument("--out", default="data/eval/bench_finetune.json")
    a = ap.parse_args()

    pairs = load_pairs()
    n_train = sum(1 for l in (CORPUS / "train.jsonl").read_text().splitlines() if l.strip())
    print(f"{len(pairs)} benchmark pairs loaded; full train set = {n_train} pairs\n")

    rows = []
    for name, batch, seq in CONFIGS:
        try:
            r = bench(name, batch, seq, a.steps, pairs, a.device)
        except Exception as e:
            print(f"  !! {name} b{batch} failed: {type(e).__name__}: {str(e)[:120]}")
            continue
        steps_per_epoch = -(-n_train // batch)
        r["steps_per_epoch"] = steps_per_epoch
        r["epoch_min_cold"] = round(steps_per_epoch * r["s_per_step"] / 60, 1)
        rows.append(r)
        print(f"  {name.split('/')[-1]:12s} {r['params_M']:6.1f}M  b{batch:<3d} seq{seq}  "
              f"{r['s_per_step']:6.3f} s/step  {r['pairs_per_s']:6.1f} pairs/s  "
              f"-> 1 epoch = {r['epoch_min_cold']:5.1f} min ({steps_per_epoch} steps)")

    soak = None
    if a.soak:
        name, batch, seq = CONFIGS[0]
        print(f"\nsoak: {a.soak}s continuous on {name} b{batch} to measure thermal drift")
        first = bench(name, batch, seq, a.steps, pairs, a.device)
        t0, blocks = time.time(), [first["s_per_step"]]
        while time.time() - t0 < a.soak:
            blocks.append(bench(name, batch, seq, a.steps, pairs, a.device)["s_per_step"])
        soak = {"s_per_step_blocks": blocks,
                "drift_pct": round(100 * (blocks[-1] / blocks[0] - 1), 1)}
        print(f"  per-step: {blocks[0]:.3f}s -> {blocks[-1]:.3f}s "
              f"({soak['drift_pct']:+.1f}% over {a.soak}s)")

    out = {"device": a.device, "n_train_pairs": n_train, "configs": rows, "soak": soak}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
