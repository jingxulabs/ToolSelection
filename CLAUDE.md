# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **research experiment**, not an application. It measures how to improve an agent's
tool/API selection from user intent over a large, open catalog. There is no build, no
test suite, no lint config, and no git repo. Scripts are run by hand, in order, and
their outputs are the deliverable.

Two documents are the source of truth, and they serve different roles:

- **`EXPERIMENT_DESIGN.md`** — the plan: hypotheses (H1–H6), the arm ladder, splits,
  metrics, confounds, statistics, and the §12 pre-registered gating criteria.
- **`DATA.md`** — what was actually acquired and run, with real numbers.
  Where the two disagree, DATA.md is what happened.

**Read `DATA.md` §4c–4d and §6–§9 before proposing work.** The direction has already
changed twice on measurement, and both decisions are settled:

- **Arms 4 (SFT) and 5 (RL) are dropped** (§4d). ~83% of error mass is retrieval miss,
  not selection error, which tripped the §12 gate. Selection given a good shortlist is
  already ~0.93; retrieval R@1 was ~0.15. Proposals to fine-tune a selector are
  re-litigating an evidence-backed decision.
- **The work is now retrieval, and fusion is the live lever** (§6–§8). Best retriever is
  `gte-large` + `v3` document text. Reranking by *replacement* loses to dense; the same
  cross-encoder scores *fused* with dense at α≈0.2 gain +9.4% R@1 — but only at a depth
  whose latency rules it out in practice (§8.4). Do not re-propose replacement reranking.
- **Arm 2 (description rewriting) is screened out as scoped** (§11). Its premise — 86
  no-description and 441 `doc_quality≤1` tools — collapses on contact with the labels:
  those tools are the gold for **3.6% of eval items and 4.4% of the miss mass**, because
  `generate_intents.py` had to read a description to write an intent, so `dq1` tools appear
  as gold at **0.150×** their catalog rate. Headroom ceiling is +0.037 R@10, below the
  +6.0 the free `v3` text already delivered. Don't spend the $7 on it; 73% of items have a
  *well-documented* gold and carry 66% of the misses. **H1 is refuted by headroom**, not
  untested — its +10–25 pts exceeds what either stage can give.
- **BM25 fusion is measured and dead** (§10). Both window and full-catalog variants:
  +0.014 mean MRR at best, pooled R@1 +0.003 at p=0.85, and only one of three fusion
  rules improves at all — failing the both-rules-agree test §8 set for itself. The cause
  is upstream of the rule: BM25's top-100 holds fewer golds than dense's *and* almost no
  golds dense lacks, so the union adds +0.000 on four of six splits. Don't re-propose it,
  and **screen any new signal on union recall headroom before sweeping it.**

## Setup

```bash
python3 -m venv .venv && ./.venv/bin/pip install requests datasets pyyaml truststore sentence-transformers
```

Always invoke `./.venv/bin/python`, and run from the repo root — every script uses
paths relative to cwd.

## Pipeline

Strict order within each stage; each step consumes the previous one's output.
**Stage A costs money** (`claude` calls); stages B and C are local and free.

```bash
# --- A. corpus + intent generation (done; data/generated/ costs ~$34 to rebuild)
./.venv/bin/python scripts/fetch_apisguru.py --n-apis 400 --per-provider 2   # -> data/catalog/
./.venv/bin/python scripts/build_corpus.py                                   # -> data/corpus/ (catalog + splits)
./.venv/bin/python scripts/build_embeddings.py                               # -> data/eval/tool_embeddings.npy
./.venv/bin/python scripts/retrieval_baseline.py --n 1200                    # arm 0/3, pooling curve
./.venv/bin/python scripts/generate_intents.py --mode confusion --n-tools 1124 --min-sim 0.82 --budget-usd 20
./.venv/bin/python scripts/generate_intents.py --mode standard --n-tools 1000 --level L2 --budget-usd 7
./.venv/bin/python scripts/generate_intents.py --mode abstain --n-tools 500 --budget-usd 5
./.venv/bin/python scripts/roundtrip_check.py --n-per-split 75 --k 20 --budget-usd 6

# --- B. cut the dev/test partition ONCE, before any config choice (done)
./.venv/bin/python scripts/split_generated.py        # -> data/corpus/generated_partition.json

# --- C. retrieval sweeps (local, free)
./.venv/bin/python scripts/build_embeddings.py --model thenlper/gte-large --text v3 \
    --device mps --out data/eval/tool_embeddings_gtelarge_v3.npy
./.venv/bin/python scripts/encoder_sweep.py --matrices data/eval/tool_embeddings_*.npy \
    --device mps                                                  # -> data/eval/encoder_sweep.json
./.venv/bin/python scripts/eval_retrieval.py --partition dev \
    --matrix data/eval/tool_embeddings_gtelarge_v3.npy --compare data/eval/tool_embeddings.npy
./.venv/bin/python scripts/rerank_sweep.py --partition dev --n 150 --device mps   # replacement (loses)
./.venv/bin/python scripts/fusion_sweep.py --partition dev --n 150 --device mps   # fusion (wins)
./.venv/bin/python scripts/bm25_fusion.py  --partition dev --n 150 --device mps   # bm25 fusion (loses)
./.venv/bin/python scripts/doc_quality_headroom.py --device mps    # arm 2 headroom screen
./.venv/bin/python scripts/significance_backfill.py               # tests for §7/§8, no recompute
```

`bm25_fusion.py` is cheap (~2 min, no cross-encoder) and is the template for new sweeps:
it imports §8's `fuse`/`recalls`/`reorder` so the arms stay comparable, asserts its own
reorder path is element-wise identical to `fusion_sweep.reorder` (guardrail C), checks its
dense block against `fusion_sweep.json`, and **persists per-item gold ranks** so results
carry McNemar + paired-bootstrap tests rather than bare point estimates. Copy that shape.

`eval_retrieval.py`, `rerank_sweep.py`, `fusion_sweep.py` and `bm25_fusion.py` take
`--partition`;
**`encoder_sweep.py` does not** — it has no partition support, which is precisely the
§6 debt recorded below.

`build_embeddings.py` must be re-run after any change to `data/corpus/catalog.jsonl`.

**Write new matrices to new filenames.** `tool_embeddings.npy` is the *shipped* v0
MiniLM artifact that split D's confusion pairs were defined against — overwriting it
silently redefines the split. `build_embeddings.py` records model, text variant and
query prefix in a `.meta.json` sidecar; `eval_retrieval.sidecar()` reads it so consumers
embed queries the same way the matrix was built (BGE prefixes the query only, E5 both
sides — getting this wrong measures a handicap, not the encoder).

**Use `--partition dev` for every config decision.** `data/generated/` shipped as one
undifferentiated pool; `split_generated.py` cut it 1,507 dev / 3,069 test by tool, with
confusion twins kept together and cross-boundary paraphrases dropped. `--partition test`
is one access per arm. Note the §6 debt: the encoder sweep predates the partition, so
spend the test access on the composed pipeline, not the encoder alone.

**`fusion_sweep.py` caches cross-encoder scores** to
`data/eval/fusion_scores_{partition}_n{n}.npz` (~70 min to build on MPS). Re-sweeping
α, the fusion rule or depth reads the cache and takes seconds — pass `--rescore` only
when the shortlists themselves change. It asserts α=0 reproduces dense bit-exactly and
cross-checks its dense block against `rerank_sweep.json`; if either fails, the harness
is wrong and the numbers are not reportable.

## Things that will bite you

**Scripts calling `claude` spend real money on the user's account.** Get explicit
approval before any run, and state the estimate. Spend so far: $40.39.

**Prompt caching dominates cost.** The CLI loads a ~27k-token harness prompt per call:
**$0.197 cold vs $0.026 warm**, 5-minute TTL. Long runs must be *continuous* — a
trickled run costs ~7×. This is why generation batches many tools per call and uses
several workers.

**`--budget-usd` truncates silently.** When the cap is hit, remaining work units return
empty and the script still exits 0. This already cost real coverage: the round-trip
check hit its $6 cap and 32 of 90 calls returned nothing, leaving ~48 items per split
instead of 75. Always compare the final `n` against what you asked for.

**Only `claude-opus-5` is reachable.** Haiku and every other model return HTTP 403 on
this account, and there are no third-party API keys, so anything needing a second
*generative* model family (ensemble triage, non-circular generation) is blocked until
the user provides a key.

**Local non-generative models do work, and are free.** `sentence-transformers`
bi-encoders and cross-encoders download from the HF Hub and run on `--device mps` —
§6–§8 are built entirely on them at $0. Budget wall-clock, not dollars: ~90 pairs/sec
for `bge-reranker-base` and ~30/sec for `bge-reranker-large`, and throughput degrades
~40% over an hour as the machine heats up, so a full sweep is ~70 min. Run these in the
background and cache the scores.

**Corporate TLS interception breaks Python HTTPS.** certifi fails with
`CERTIFICATE_VERIFY_FAILED`; `truststore.inject_into_ssl()` uses the macOS keychain
instead. Every network-touching script already does this — keep it in new ones. `curl`
is unaffected.

## Architecture

### Distractor pooling — the core idea

No public benchmark tests this problem: they present 1–10 candidate tools per query, so
retrieval is trivially solved and only selection is measured. `data/corpus/catalog.jsonl`
therefore fuses two sources:

- **199 MetaTool plugins** — carry all 20,532 labeled `(query → tool)` pairs
- **3,352 APIs.guru operations** — carry *no* labels; they exist as distractors

Selection happens over all 3,551 at a 17.8:1 distractor ratio. This is what makes
retrieval load-bearing: recall@100 falls from 0.973 at a 199-tool pool to 0.861 at 3,551
(MiniLM-v0, the shipped baseline; the current `gte-large`/v3 stack reaches 0.918 — see
"Retrieval stack" below, and don't quote 0.861 as the live ceiling).

### Retrieval stack — what is actually measured to work

Retrieval is the whole experiment now (§4d), so these are the live numbers. Mean R@1
over the five generated splits, `DATA.md` §6–§10. **The two blocks are not the same
basis** — the encoder sweep predates the partition (the §6 debt) — so compare within a
block, not across it:

| stage | mean R@1 | paired test vs dense | basis |
|---|---:|---|---|
| MiniLM-v0 (shipped baseline) | 0.139 | — | all, n≤1200 |
| `gte-large` + `v3` text — winner of 10 configs | **0.259** | — | all, n≤1200 |
| `gte-large`/v3 dense | 0.253 | — | dev, n≤150 |
| + cross-encoder **fused** at α≈0.2 | 0.277 | **R@1 n.s. (p=0.12)**; MRR +0.023 [+0.007,+0.041] | dev, n≤150 |
| + cross-encoder by **replacement**, depth 100 | 0.214 | **significantly worse** (p=0.005) | dev, n≤150 |
| + cross-encoder by **replacement**, depth 10 | — | n.s. (p=0.57) — neutral, not harmful | dev, n≤150 |
| + BM25 fused, full catalog at α≈0.1 | 0.267 | n.s. (p=0.85) | dev, n≤150 |
| BM25 alone | 0.086 | — | dev, n≤150 |

The encoder plus document text is the big win (+0.120) and the only unambiguous one.
**Everything after it is small enough that the interval matters** (§12): cross-encoder
fusion's "+9.4% R@1" headline does *not* survive a paired test — quote it as an **MRR**
gain, which does, under both fusion rules. Replacement is significantly harmful only at
depth; BM25 fusion is not significant at all. The §8.4 latency objection still rules
fusion out of production regardless.

Three durable lessons:

1. **`v3` document text is nearly free accuracy.** Adding tags, api_title/provider, HTTP
   method and the tokenized path is worth a **mean +6.0 R@10 points**, and it helps
   *every* encoder tried on average — positive in 24 of 25 encoder×split cells (range
   −2.6 to +19.5; the one loss is `gte-large` on `splitD_inseparable`, n=113, and the
   +19.5 outlier is e5, which gains most because it started worst). Most of the gain
   lands on thin-description tools (R@10 0.348 → 0.568 on standard_L2): OpenAPI paths
   carry the discriminating signal that the prose often lacks.
2. **Combine weak signals, never substitute them — but there is a floor.** A reranker
   measured at 0.77–0.90× dense quality *helps* when fused and *hurts* when it replaces;
   the depth effect is the tell (fusion improves with depth, replacement degrades). The
   floor is §10: BM25 at 0.05–0.31× dense, with no coverage dense lacks, gains nothing
   from the same machinery. Fusion reweights information, it does not manufacture it.
3. **Screen a new signal on union recall headroom first, and demand two fusion rules.**
   If `dense top-k ∪ signal top-k` holds no more golds than dense alone, fusion has
   nothing to find — that check is minutes of work and would have predicted §10 up front.
   And §8 only believed its own result because zscore *and* rrf agreed; §10 fails exactly
   that test. Hold new signals to it, including when the single-rule number looks good.

### The two sources are not interchangeable

| | MetaTool | APIs.guru |
|---|---|---|
| Labeled intents | 20,532 | 0 (generated) |
| Tools with parameters | **0%** | 87.1% |
| doc_quality spread | flat (175/199 at level 1) | full 0–4 |
| Pairs >0.85 similarity | **0** | 844 |

MetaTool cannot test H1 (no doc variance), split D (no confusable pairs), or argument
grounding (no parameter schemas). **APIs.guru + generated intents is the primary set**;
MetaTool is retained as the independent anchor, since it is the only labeled data *not*
generated by Claude and so guards against the generation pipeline flattering itself.

### Splits (`data/corpus/`, `data/generated/`)

Tools are held out **before** intents — a random intent split leaks tool identity and
overstates generalization. Near-duplicate queries are deduped across split boundaries
first. `C_unseen_tools` (39/199 tools held out entirely) is the generalization split.

Two *independent* cuts exist, and they are easy to confuse:

- **MetaTool** carries its own `train` / `dev_A` / `test_A` / `C_unseen_tools` files in
  `data/corpus/`, cut by `build_corpus.py`.
- **Generated intents** are not split on disk at all. `data/generated/*.jsonl` is one
  pool, and `generated_partition.json` is a *manifest* (`intent_id → dev|test|dropped`)
  applied at load time by `eval_retrieval.splits(partition)`. 1,507 dev / 3,069 test.

So `--partition dev` means "dev_A as the MetaTool anchor **plus** the dev third of the
generated pool". The same rules as above apply within it — assignment by tool, confusion
twins union-found so a target and its near-neighbour never straddle the boundary, and
paraphrase dedup across the boundary at cosine >0.95.

### `separable: false` — the label-honesty mechanism

Split D generation hands the model a tool *and* its nearest confusable neighbour and
asks for a request only the target satisfies — while explicitly permitting
`separable: false`. **10.1%** come back inseparable ("only HTTP verb differs", "wrapper
vs core endpoint"). These are real OpenAPI artifacts that would otherwise be scored as
model errors when the model was right; they become variant F (set-valued gold).

Independently confirmed: the *selector* reported 32.7% "tied" on these items vs 0.0% on
separable ones. Two unrelated mechanisms agreeing is why no human annotation pass was
needed. Preserve this flag through any downstream processing.

### Implicit coupling to watch

`data/eval/tool_embeddings.npy` is **row-aligned with `data/corpus/catalog.jsonl`** —
row *i* is line *i*. `generate_intents.py --mode confusion` and `roundtrip_check.py`
both index it positionally. Regenerate the catalog without regenerating embeddings and
they read garbage silently, with no error.

**Known inconsistency:** the shipped matrix (and therefore split D) was built *without*
tool tags (`--text v0`), while `retrieval_baseline.py` computes its own embeddings *with*
them (equivalent to `--text v1`). 2,868 tools have tags; mean row cosine 0.986,
confusion-pair overlap 93%. `build_embeddings.py` defaults to `v0` to reproduce the
shipped artifact. Unifying them requires regenerating split D, so it has not been done —
every later sweep sidesteps it by writing new matrices and reading the variant back from
the `.meta.json` sidecar. (`--tags` still exists as a deprecated alias for `--text v1`;
prefer `--text`.)

## Data layout

| Path | Contents | Regenerable |
|---|---|---|
| `data/raw/` | third-party downloads (MetaTool, ToolACE, Glaive) | yes, slow |
| `data/catalog/` | APIs.guru only, pre-merge | yes |
| `data/corpus/` | unified catalog + splits | yes, fast |
| `data/generated/` | LLM-generated intents | **costs ~$34 to rebuild** |
| `data/eval/` | embeddings (+`.meta.json` sidecars), baselines, round-trip, sweeps | yes, ~70 min |

`data/eval/` holds 10 embedding matrices (~108 MB), `fusion_scores_dev_n150.npz` — the
cached cross-encoder scores — and the per-item gold ranks behind the significance tests
(`bm25_fusion_ranks_dev_n150.npz` for §10, `significance_backfill_ranks.npz` for §7–§8).
Everything there is reproducible, but the cross-encoder cache costs ~70 min of local
compute, so **don't delete it casually — it is also the only input `significance_backfill.py`
needs**, and without it §7–§8 lose their intervals. The rank files re-derive in seconds to
~2 min. `tool_embeddings.npy` is the one file that must never be overwritten (see "Write
new matrices to new filenames").

`data/raw/glaive_v2` (243 MB) and `data/raw/toolace` (32 MB) were downloaded as SFT
training data for the now-dropped arms 4–5. Unused; safe to delete.

## Known limitations to state, not paper over

- **Circularity is unmitigated.** Claude Opus 5 generated the primary eval set and is
  the model being evaluated. Partial defenses: MetaTool as a GPT-4-generated anchor, and
  split D's self-reported `separable` flag. A real fix needs a second model family.
- **Difficulty levels are not ordered.** L3 (indirect) is harder than L4 (distractor
  context): R@10 0.394 vs 0.432 under MiniLM-v0, and **confirmed** under `gte-large`/v3
  at 0.564 vs 0.605. Treat L2/L3/L4 as three conditions, not a scale.
- **Round-trip check ran at n≈48/split**, not 75, from budget truncation. 95% CI ≈ ±6
  pts, so differences *among* generated splits are not significant.
- **Significance:** §10–§12 have paired tests; §6 does not. `rerank_sweep.py` and
  `fusion_sweep.py` still persist aggregates only, but `significance_backfill.py` recovers
  per-item ranks from `fusion_scores_dev_n150.npz` with **no recompute**, so §7–§8 are
  covered (§12). §6's encoder sweep is not — it has no score cache and predates the
  partition. **Two of the four claims tested in §12 changed status**, so treat any
  un-tested number in §6 as provisional and attach an interval to anything quoted.
- **The §8 fusion gain is an argmax over 176 configs**, selected on dev. Expect
  regression to the mean on test, and don't carry the exact α into production as if it
  were measured — §12.4 confirms α∈[0.1,0.3] is a plateau whose members are not
  distinguishable from each other.
- **`standard_L4` regresses under fusion** (R@1 −0.047) and is **hostile to the
  cross-encoder in every configuration**, significantly so under replacement (−0.180 R@1,
  p=0.00003 — the only split where replacement is significant at any depth, §12.5). It is
  the *easiest* standard level for dense retrieval, so this is unexplained. Live
  hypothesis: L4's decoy context is attended to by a cross-encoder and averaged away by a
  bi-encoder.
- **The eval set under-samples poorly-documented tools** (§11.5). `dq1` tools are gold at
  0.150× their catalog rate because the generator had to read a description to write an
  intent. Any claim about documentation quality measured here is a *lower* bound on its
  production effect, and this is not fixable by rewriting — only by regenerating intents
  for those tools, which costs money and adds a new circularity.
