"""Fine-tune the bi-encoder retriever on MetaTool's labeled pairs.

The §12 gate (§4d) dropped arms 4-5 because ~83% of the error mass is retrieval miss, not
selection error. The corollary -- never tested -- is that the RETRIEVER is the thing worth
training. §6-§10 only ever swapped in off-the-shelf encoders; none of them has seen a
single labeled pair from this domain. Off the shelf, gte-large encodes general semantic
similarity, but the relation this task needs is "this intent is SERVED BY this operation",
which is not the same relation.

Contrastive, MultipleNegativesRankingLoss over (anchor, positive, 4 mined negatives):
cosine x 20 scored against every candidate in the batch, cross-entropy to the positive.
Full fine-tune -- gradients reach all encoder weights, and because a bi-encoder shares one
tower between queries and documents, the whole catalog must be re-embedded afterwards.

TWO SETTINGS THAT ARE NOT OPTIONAL:

  batch_sampler=NO_DUPLICATES. train.jsonl has 11,326 pairs over only 160 distinct tools
  (median 46 queries/tool, max 764). Measured: at batch 32, 99.9% of RANDOM batches
  contain two queries sharing a gold tool. MNRL treats every other in-batch item as a
  negative, so default sampling would spend nearly every step pushing apart queries that
  should map to the same tool -- training the wrong objective outright.

  1 epoch, low LR. With 160 distinct positives the dominant risk is not classic overfit
  but LABEL-SPACE COLLAPSE: the space reorganising into 160 attractor basins, which gets
  worse with more epochs and shows up as degraded transfer while training loss keeps
  falling. Training loss cannot detect it; only the held-out tool populations can.

--probe measures the specific hazard that mined negatives create (see
mine_hard_negatives.py): 66% of hard negatives are APIs.guru documents, which are the
population the generated splits require as POSITIVES. If the mean query-to-APIs.guru
cosine falls after training, the document tower has learned "metadata-style text is not
answer-shaped" unconditionally, and transfer will be negative rather than flat.

  ./.venv/bin/python scripts/finetune_retriever.py --base thenlper/gte-base \
      --epochs 1 --out models/gte-base-ft-screen --device mps --probe
"""

import argparse, json, time
from pathlib import Path

import numpy as np

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from eval_retrieval import CORPUS, splits

TRIPLETS = CORPUS / "train_triplets.jsonl"
SEED = 13


def probe(model, base_matrix, device, n_per_split=150):
    """Mean cosine from generated-split queries to APIs.guru docs, trained vs baseline.

    Mode-4 detector. Reported as a DELTA against the same quantity under the untrained
    matrix, on identical queries, so it isolates what training did to the geometry rather
    than measuring an absolute that has no reference point.
    """
    from build_embeddings import tool_text
    from eval_retrieval import sidecar

    cat = [json.loads(l) for l in (CORPUS / "catalog.jsonl").read_text().splitlines() if l.strip()]
    guru = np.array([i for i, t in enumerate(cat) if t["source"] == "apis.guru"])
    bmodel, bqpre, bmeta = sidecar(base_matrix, None, None)
    variant = bmeta.get("text_variant", "v3")
    E0 = np.load(base_matrix)

    sp = {k: v for k, v in splits("dev").items() if k.startswith(("standard", "splitD"))}
    rng = np.random.default_rng(SEED)
    qs = []
    for name, items in sp.items():
        if len(items) > n_per_split:
            items = [items[i] for i in rng.choice(len(items), n_per_split, replace=False)]
        qs += [it["query"] for it in items]

    from sentence_transformers import SentenceTransformer

    base = SentenceTransformer(bmodel, device=device)
    Q0 = base.encode([bqpre + q for q in qs], batch_size=128, normalize_embeddings=True,
                     show_progress_bar=False).astype(np.float32)
    del base
    E1 = model.encode([tool_text(t, variant) for t in cat], batch_size=64,
                      normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    Q1 = model.encode(qs, batch_size=128, normalize_embeddings=True,
                      show_progress_bar=False).astype(np.float32)

    out = {}
    for tag, Q, E in (("baseline", Q0, E0), ("finetuned", Q1, E1)):
        S = Q @ E[guru].T
        out[tag] = {"mean_cos_to_apisguru": round(float(S.mean()), 4),
                    "mean_top1_cos": round(float(S.max(axis=1).mean()), 4)}
    out["delta_mean"] = round(out["finetuned"]["mean_cos_to_apisguru"]
                              - out["baseline"]["mean_cos_to_apisguru"], 4)
    out["delta_top1"] = round(out["finetuned"]["mean_top1_cos"]
                              - out["baseline"]["mean_top1_cos"], 4)
    out["n_queries"] = len(qs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="thenlper/gte-base")
    ap.add_argument("--triplets", default=str(TRIPLETS))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--freeze-layers", type=int, default=0,
                    help="freeze the bottom N transformer layers (0 = full fine-tune)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--out", default="models/gte-base-ft-screen")
    ap.add_argument("--base-matrix", default="data/eval/tool_embeddings_gtebase_v3.npy")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--cap-per-positive", type=int, default=0,
                    help="max training rows per distinct gold tool (0 = no cap)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--sampler", default="no_duplicates",
                    choices=["no_duplicates", "default"],
                    help="no_duplicates avoids same-gold-in-batch but must dedupe EVERY "
                         "column; with explicit negatives that can be the bottleneck")
    ap.add_argument("--n-neg", type=int, default=0,
                    help="keep only the first N negative columns (0 = all present)")
    a = ap.parse_args()

    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
    from sentence_transformers import SentenceTransformerTrainingArguments as TA
    from sentence_transformers.sentence_transformer.losses import MultipleNegativesRankingLoss
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers

    rows = [json.loads(l) for l in Path(a.triplets).read_text().splitlines() if l.strip()]

    # Capping rows per gold tool does two things at once. Statistically it stops one tool
    # (764 of 11,326 queries) from dominating the gradient. Mechanically it is what makes
    # NO_DUPLICATES affordable: that sampler must assemble batches with no repeated
    # positive, and on the raw heavy-tailed distribution it defers most of the epoch in
    # Python, which is why the uncapped run was CPU-bound rather than GPU-bound.
    if a.cap_per_positive:
        rng = np.random.default_rng(SEED)
        by_pos = {}
        for r in rows:
            by_pos.setdefault(r["positive"], []).append(r)
        kept = []
        for p, group in by_pos.items():
            if len(group) > a.cap_per_positive:
                idx = rng.choice(len(group), a.cap_per_positive, replace=False)
                group = [group[i] for i in idx]
            kept += group
        rng.shuffle(kept)
        print(f"cap {a.cap_per_positive}/tool: {len(rows)} -> {len(kept)} rows "
              f"over {len(by_pos)} tools")
        rows = kept

    if a.n_neg:
        keep = ["anchor", "positive"] + [f"negative_{i}" for i in range(1, a.n_neg + 1)]
        rows = [{k: r[k] for k in keep if k in r} for r in rows]

    ds = Dataset.from_list(rows)
    print(f"{len(rows)} triplets, columns={list(ds.column_names)}")
    print(f"distinct positives: {len({r['positive'] for r in rows})}  "
          f"<- the label space the model can learn")

    m = SentenceTransformer(a.base, device=a.device)
    m.max_seq_length = a.seq

    # transformers 5.x loads a checkpoint in its NATIVE dtype, and gte-* ship as fp16.
    # Training fp16 weights with plain AdamW and no gradient scaler overflows within a few
    # steps: grad_norm goes nan, loss prints 0, and every saved tensor is NaN -- while the
    # reported train_loss average still looks plausible, so it fails silently. fp16
    # INFERENCE is fine, which is why the baseline embedding matrices are unaffected.
    # Cast to fp32 and assert it, because this cost a full 17-minute run to find.
    dt = next(m.parameters()).dtype
    if dt != torch.float32:
        print(f"casting model from {dt} to float32 for training")
        m = m.float()
    assert next(m.parameters()).dtype == torch.float32, "model must be fp32 to train"
    if a.freeze_layers:
        enc = m[0].auto_model
        for p in enc.embeddings.parameters():
            p.requires_grad = False
        for layer in enc.encoder.layer[: a.freeze_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"froze embeddings + bottom {a.freeze_layers} layers; "
              f"{n_tr/1e6:.1f}M params trainable")

    args = TA(output_dir=f"/tmp/ft_{Path(a.out).name}", num_train_epochs=a.epochs,
              per_device_train_batch_size=a.batch, learning_rate=a.lr,
              warmup_ratio=a.warmup_ratio,
              batch_sampler=(BatchSamplers.NO_DUPLICATES if a.sampler == "no_duplicates"
                             else BatchSamplers.BATCH_SAMPLER),
              max_steps=a.max_steps, logging_steps=50, save_strategy="no", report_to=[],
              dataloader_num_workers=0, seed=SEED, fp16=False, bf16=False)
    trainer = SentenceTransformerTrainer(model=m, args=args, train_dataset=ds,
                                         loss=MultipleNegativesRankingLoss(m))
    t0 = time.time()
    out = trainer.train()
    el = time.time() - t0
    steps = int(out.global_step)
    seen = steps * a.batch
    print(f"\ntrained {steps} steps in {el/60:.1f} min ({el/max(steps,1):.2f} s/step)")
    print(f"  samples seen: {seen} of {len(rows)} ({seen/len(rows):.1%}) -- "
          f"NO_DUPLICATES drops rows it cannot place in a duplicate-free batch")
    print(f"  final train loss: {out.training_loss:.4f}")

    bad = [n for n, p_ in m.named_parameters() if torch.isnan(p_).any()]
    if bad:
        raise SystemExit(f"DIVERGED: {len(bad)} parameters are NaN (e.g. {bad[0]}). "
                         f"Refusing to save. The reported train_loss average can look "
                         f"plausible even when every weight is NaN -- check dtype first.")
    probe_vec = m.encode(["find academic research papers"], normalize_embeddings=True)
    if np.isnan(probe_vec).any():
        raise SystemExit("DIVERGED: encoder emits NaN embeddings. Refusing to save.")
    print("  finite-weight check ok")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    m.save(a.out)
    meta = {"base": a.base, "triplets": a.triplets, "epochs": a.epochs,
            "batch": a.batch, "lr": a.lr, "seq": a.seq, "seed": SEED,
            "freeze_layers": a.freeze_layers, "steps": steps,
            "samples_seen": seen, "n_triplets": len(rows),
            "train_loss": round(float(out.training_loss), 4),
            "train_minutes": round(el / 60, 2),
            "batch_sampler": a.sampler, "n_neg_used": a.n_neg}

    if a.probe:
        print("\nprobe: mode-4 geometry check (query -> APIs.guru docs)", flush=True)
        meta["probe"] = probe(m, a.base_matrix, a.device)
        p = meta["probe"]
        print(f"  mean cosine to APIs.guru docs: {p['baseline']['mean_cos_to_apisguru']:.4f} "
              f"-> {p['finetuned']['mean_cos_to_apisguru']:.4f}  ({p['delta_mean']:+.4f})")
        print(f"  mean top-1 cosine:             {p['baseline']['mean_top1_cos']:.4f} "
              f"-> {p['finetuned']['mean_top1_cos']:.4f}  ({p['delta_top1']:+.4f})")
        print("  (absolute cosine scale is not comparable across models; the SIGN of the"
              "\n   top-1 delta relative to the mean delta is the signal -- if the mean"
              "\n   falls while top-1 holds, the space sharpened rather than collapsed.)")

    Path(a.out + "/train_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved {a.out}  (+ train_meta.json)")
    print(f"next: ./.venv/bin/python scripts/build_embeddings.py --model {a.out} "
          f"--text v3 --device {a.device} \\\n"
          f"        --out data/eval/tool_embeddings_{Path(a.out).name.replace('-','')}_v3.npy")


if __name__ == "__main__":
    main()
