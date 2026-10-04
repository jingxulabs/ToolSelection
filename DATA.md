# Data inventory — acquired and verified

**2026-09-27.** Everything below was downloaded and run locally, not just cited.
Reproduce with `scripts/fetch_apisguru.py` → `scripts/build_corpus.py` →
`scripts/retrieval_baseline.py` (venv at `.venv`, deps: requests, datasets,
sentence-transformers, truststore).

> Note: this network does TLS interception. Python needs `truststore.inject_into_ssl()`
> to use the macOS keychain; certifi alone fails with `CERTIFICATE_VERIFY_FAILED`.
> Already handled in the scripts.

---

## 1. Acquired

| Source | Got | License | Role |
|---|---|---|---|
| **APIs.guru** | **3,352 tools** / 378 providers / 32 domains (sampled from 2,529 APIs, 3,992 specs, **108,837 endpoints**) | CC0-ish, open | Catalog backbone + distractor pool |
| **MetaTool** | **199 tools, 20,614 labeled (query → tool) pairs** | MIT | **Primary labeled set** |
| **ToolACE** | 11,300 dialogues, from a 26,507-API pool | Apache-2.0 | SFT/GRPO training |
| **Glaive FC v2** | 112,960 dialogues | Apache-2.0 | SFT bulk |
| **BFCL** | code+data via `pip install bfcl-eval` | Apache-2.0 | External anchor |
| **AppWorld** | `pip install appworld` (457 APIs, 9 apps) | Apache-2.0 (encrypted bundles) | Phase 2 |

**Blocked:** `Salesforce/xlam-function-calling-60k` is **gated** — needs an HF account,
accepted terms, and `HF_TOKEN`. Worth unblocking: 60k records, verified by real
execution, human-checked >95% correct. The single highest-quality training set here.

**Corrections to what I said earlier:** APIs.guru is 2,529 APIs / 108,837 endpoints,
not "~4k specs." And **τ-bench is deprecated** — its own README says the tasks are
outdated; use **τ²/τ³-bench** (`sierra-research/tau2-bench`, MIT), which adds telecom
and banking domains and fixed 75+ broken tasks. Results across the versions are not
comparable.

---

## 2. Built corpus — `data/corpus/`

MetaTool's 199 candidates are too few for retrieval to matter, so the APIs.guru
catalog is pooled in as distractors (design §4.0).

```
catalog.jsonl          3,551 tools  (199 labeled + 3,352 distractors, 17.8:1)
train.jsonl           11,326 intents
dev_A.jsonl            2,006
test_A.jsonl           3,355
C_unseen_tools.jsonl   3,845   ← 39/199 tools held out entirely
```

Tools were split **before** intents, and 82 near-duplicate queries were dropped
across split boundaries first. Catalog records carry `doc_quality` (0–4) and
`side_effects` (read/write/destructive: 1,987/1,146/219).

---

## 3. Result: distractor pooling works

Dense = all-MiniLM-L6-v2, n=1,200 per split. **No training yet — this is arm 3.**

**test_A**

| Pool | dense R@1 | R@10 | R@25 | R@100 | bm25 R@1 | R@10 |
|---:|---:|---:|---:|---:|---:|---:|
| 199 | 0.540 | 0.815 | 0.889 | **0.973** | 0.296 | 0.549 |
| 500 | 0.508 | 0.792 | 0.860 | 0.946 | 0.273 | 0.504 |
| 1000 | 0.483 | 0.766 | 0.830 | 0.921 | 0.261 | 0.467 |
| 2000 | 0.454 | 0.731 | 0.797 | 0.892 | 0.247 | 0.422 |
| **3551** | **0.442** | **0.708** | 0.771 | **0.861** | 0.241 | 0.391 |

Four things follow:

1. **Pooling does what it was supposed to.** At 199 candidates recall@100 = 0.973 —
   retrieval is effectively solved and the arm is untestable. At 3,551 it's 0.861, so
   **13.9% of gold tools are unreachable regardless of how good the selector is.**
   That's the E1 ceiling, and it only exists because of pooling.
2. **Dense beats BM25 by ~20 points at R@1** (0.442 vs 0.241). Lexical retrieval isn't
   viable here — users don't phrase intents in API vocabulary. Keep BM25 as arm 0 only.
3. **Gating check (design §12):** at k=10, 29% of golds are already missing. The rule
   says stop the training track if retrieval misses exceed 40% of remaining errors —
   that's plausibly already true at small k. **Run the arm-3 error taxonomy before
   committing to arms 4–5.** Provisionally k≈25 looks like the right operating point.
4. **C_unseen_tools tracks test_A almost exactly** (R@10 0.692 vs 0.708). Correct and
   expected — nothing is trained, so "held out" means nothing yet. Its value is as the
   reference line: if arms 4–5 lift test_A but not this, that's H4 confirmed.

---

## 4. Gap found: split D is not buildable from this data

Nearest-neighbour similarity within each source:

| Source | mean nn-sim | >0.75 | >0.85 | >0.95 |
|---|---:|---:|---:|---:|
| MetaTool (199, **labeled**) | 0.442 | 2 (1.0%) | **0 (0%)** | 0 |
| APIs.guru (3,352, **unlabeled**) | 0.748 | 1,724 (51.4%) | 844 (25.2%) | 242 |

**The labeled tools have no confusion; the confusable tools have no labels.** MetaTool's
plugins are semantically well-separated — zero pairs above 0.85 — so it cannot support
split D (near-miss discrimination) at all. Since near-miss discrimination is exactly
what arms 4–5 are supposed to fix, **the current data cannot test the main hypothesis.**

This is the concrete empirical justification for the §4.2 generation pipeline: point it
at the 844 APIs.guru tools with a >0.85 neighbour, and it produces the confusion set
that no public benchmark provides.

---

## 4b. Split D pilot — generated, validated

`scripts/generate_intents.py --mode confusion`. 60 intents over tool pairs with
>0.85 similarity. **55s, 0 errors.**

The generator is asked for a request the TARGET satisfies and its NEIGHBOUR does
not — and is explicitly permitted to answer `separable: false`. **6.7% came back
inseparable**, with reasons like *"identical function and parameters; only HTTP verb
differs"* and *"wrapper versus core endpoint invisible to users"*. Those are real
OpenAPI artifacts that would otherwise have been scored as model errors when the
model was right. They become variant F (set-valued gold) instead.

This is the answer to the label-validity problem: the generator self-reports
ambiguity at generation time, so no human annotation pass is needed to find it.

**Difficulty check (dense retrieval, full 3,551 pool):**

| | R@1 | R@10 | R@100 |
|---|---:|---:|---:|
| Split D separable | **0.179** | 0.661 | 0.911 |
| MetaTool test_A | 0.330 | 0.667 | 0.872 |

R@1 is half MetaTool's while R@10/R@100 match. The gold tool is retrieved into the
neighbourhood but ranks below its twin — near-miss discrimination isolated from
retrieval. **MetaTool cannot produce this signal at all** (zero pairs >0.85).

Prompt caching is load-bearing for confusion mode (two tool specs per item): a cold
call costs roughly 7.5× a warm one, so runs must be continuous.

## 4c. Full generation — complete

`scripts/generate_intents.py`, 347 calls, **0 errors**. This is the one stage that
calls a paid API; everything from §6 on runs locally.

| Set | n |
|---|---:|
| Split D (confusion, sim≥0.82) | 1,120 |
| Standard L2 | 987 |
| Standard L3 | 996 |
| Standard L4 | 982 |
| Split E (abstention) | 500 |
| **Total** | **4,585** |

Split D used sim≥0.82 (1,124 pairs available) rather than 0.85 (844) to approach the
1,500 target; dropping to 0.78 would have diluted the confusion signal. Final: 1,120.
Standard used the **same 1,000 tools at all three levels**, so L2/L3/L4 is a paired
difficulty comparison rather than one confounded with tool identity.

**Retrieval difficulty (dense, full 3,551 pool):**

| Split | n | R@1 | R@10 | R@100 | top-1 sim |
|---|---:|---:|---:|---:|---:|
| Split D separable | 1,007 | 0.180 | 0.544 | 0.819 | 0.474 |
| Split D inseparable | 113 | 0.088 | 0.602 | 0.814 | 0.477 |
| Standard L2 | 987 | 0.164 | 0.473 | 0.763 | 0.445 |
| Standard L3 | 996 | 0.131 | 0.394 | 0.706 | 0.432 |
| Standard L4 | 982 | 0.130 | 0.432 | 0.751 | 0.436 |
| Split E abstention | 500 | — | — | — | **0.479** |
| MetaTool test_A (ref) | 800 | 0.349 | 0.674 | 0.879 | 0.402 |

- **Split E is hard by construction.** Out-of-scope queries have *higher* top-1
  similarity (0.479) than real in-scope ones (0.445) or MetaTool (0.402). No
  similarity threshold can separate them, which is what makes the split worth having.
- **Difficulty ladder is not monotonic.** L3 (indirect) is harder than L4 (distractor
  context): R@10 0.394 vs 0.432. Treat L2/L3/L4 as three conditions, not a scale.

**Circularity warning:** MetaTool (GPT-4-generated), ToolACE, Glaive and xLAM are all
LLM-generated. Generate split D with a *different* model family from whatever you
evaluate, or the numbers are inflated for free (design §4.2).

## 4d. Round-trip validity check — and the finding that changes the plan

`scripts/roundtrip_check.py`. Gold forced into a k=20 shortlist (oracle
retrieval) to separate "hard but valid" from "noisy labels".

| Split | answered | accuracy | tied% |
|---|---:|---:|---:|
| standard L2 | 48 | **0.958** | 2.1 |
| standard L3 | 45 | **0.911** | 0.0 |
| split D separable | 52 | **0.904** | 0.0 |
| MetaTool (ref) | 49 | 0.878 | 2.0 |
| split D inseparable | 52 | 0.731 | **32.7** |
| split E abstention | 44 | 0.864 | 86.4% answered "none" |

**1. The generated intents are valid.** With gold in the shortlist, recovery is
90–96% — *above* MetaTool's 87.8%. The low dense R@1 is therefore difficulty, not
label noise. No regeneration needed.

**2. Retrieval is the bottleneck, not selection.** Composing the stages: retrieval
R@25 ≈ 0.65, selection given a good shortlist ≈ 0.93 → end-to-end ≈ 0.60. Of the
error mass, **~83% is retrieval miss and ~7% is selection error.**

> **§12 gate triggered.** The rule was: stop the training track if retrieval misses
> exceed 40% of remaining errors. It is at ~83%. **Arms 4 and 5 are not justified.**
> A fine-tuned selector cannot fix a retriever that misses a third of golds. Redirect
> to S1: better embeddings, hybrid retrieval, rerankers, and the arm-2 rewrite (which
> improves the *retrieval* text, not just the selection prompt).

**3. The ambiguity flag is independently confirmed.** Items the *generator* marked
`separable: false` came back **32.7% "tied"** from the *selector*, vs 0.0% for
separable items. Two unrelated mechanisms agree, so the 10.1% rate is real. Those
items are variant F (set-valued gold), not model errors.

**Limitation:** a budget cap truncated the run — 32 of 90 calls returned empty,
leaving ~48 answered items per split instead of 75. 95% CI ≈ ±6 pts, so differences
*among* generated splits are not significant. The selection-vs-retrieval gap
(0.93 vs 0.15) is far too large to be affected.

## 6. Retrieval sweep — encoder and document text (2026-09-30)

The S1 redirect from §4d, executed. **All local models, no API spend.**
`scripts/build_embeddings.py` gained multi-encoder support with correct asymmetric
prefixes (BGE prefixes the query only; E5 prefixes *both* sides — scoring E5 without
`passage: ` on documents measures a handicap, not the encoder) and four document
constructions. `scripts/encoder_sweep.py` → `data/eval/encoder_sweep.json`, 10 configs.

Document variants: v0 name+summary+description+param names (the shipped matrix) →
v1 +tags → v2 +api_title/provider → v3 +HTTP method +humanized path.

**R@10, n≤1200 per split:**

| config | test_A | std_L2 | std_L3 | std_L4 | splitD_sep |
|---|---:|---:|---:|---:|---:|
| MiniLM-L6-v2 / v0 (shipped baseline) | 0.725 | 0.473 | 0.394 | 0.432 | 0.544 |
| MiniLM-L6-v2 / v3 | 0.693 | 0.525 | 0.433 | 0.503 | 0.613 |
| e5-base-v2 / v3 | 0.536 | 0.491 | 0.404 | 0.399 | 0.615 |
| bge-base-en-v1.5 / v3 | 0.640 | 0.611 | 0.484 | 0.494 | 0.708 |
| bge-large-en-v1.5 / v3 | 0.680 | 0.605 | 0.497 | 0.511 | 0.723 |
| **gte-large / v3** | **0.759** | **0.644** | **0.564** | **0.605** | **0.737** |

1. **gte-large/v3 wins on every split**, and the two factors are independent: the v3
   document text is worth a **mean +6.0 pts R@10** and lifts every encoder tried on
   average — positive in 24 of 25 encoder×split cells, range −2.6 to +19.5. The spread
   is wide and inversely related to baseline strength: e5-base-v2 gains most (+13.4
   mean) because it started worst, bge-large least (+3.4). The sole negative cell is
   gte-large on splitD_inseparable (n=113). Encoder size alone is not the lever —
   bge-large ≈ bge-base.
2. **The gain is concentrated where the losses were.** Thin-description tools (gold
   desc ≤5 words) go from R@10 0.348 → 0.568 on standard_L2. Adding the HTTP method
   and tokenized path gives discriminating signal to tools whose prose carries none.
3. **e5-base-v2 lost to MiniLM** even with its prefixes handled correctly. Genuine, not
   a harness artifact.
4. **Retrieval is materially better but still the bottleneck.** R@25 on the generated
   splits went 0.52–0.67 → 0.68–0.82. The §4d error decomposition shifts but does not
   invert: selection given a good shortlist was ~0.93.

**Methodological debt:** this sweep ran *before* the dev/test partition below was cut,
i.e. on the full generated pool. The choice of gte-large/v3 was therefore made while
looking at what is now test. Do not spend the one test access re-confirming the encoder
in isolation — spend it on the final composed pipeline.

### Dev/test partition — `data/corpus/generated_partition.json`

`data/generated/` shipped as one undifferentiated pool, so every config choice made by
looking at it erodes it. `scripts/split_generated.py` cuts it once, by the design's
rules: **by tool first** (a random intent split leaks tool identity; because L2/L3/L4
reuse the same 1,000 tools, tool-level assignment also keeps a tool's three levels on
the same side, so the paired difficulty comparison survives *inside* each partition);
**confusion twins union-found and assigned as a unit** (target in dev with its
neighbour in test would leak the discrimination signal); **cross-boundary paraphrase
dedup** at cosine >0.95.

1,507 dev / 3,069 test / 9 dropped. `tools_on_both_sides: []`. It writes a manifest
(`intent_id → dev|test|dropped`), not new copies.

## 7. Reranking does not work here — measured, negative

`scripts/rerank_sweep.py --partition dev --n 150 --device mps`, 157,800 query-document
pairs over two cross-encoders. **~70 min, local.** → `data/eval/rerank_sweep.json`.

The motivating argument was sound on its face: with gte-large/v3, R@100 is 0.83–0.95
while R@10 is 0.56–0.76, so 20–30 points of gold is already retrieved but ranked too
low to reach the selector, and converting recall into precision is a cross-encoder's
entire job. **It does not happen.**

**R@1 / R@10, dev partition, bge-reranker-large (the better of the two):**

| split | dense R@1 | →top-10 | →top-100 | dense R@10 | →top-25 | →top-100 |
|---|---:|---:|---:|---:|---:|---:|
| dev_A | 0.567 | 0.513 | 0.507 | 0.813 | 0.820 | 0.733 |
| standard_L2 | 0.280 | 0.280 | 0.240 | 0.653 | 0.573 | 0.553 |
| standard_L3 | 0.207 | 0.213 | 0.153 | 0.553 | 0.500 | 0.453 |
| standard_L4 | 0.280 | 0.140 | 0.100 | 0.600 | 0.540 | 0.460 |
| splitD_separable | 0.267 | **0.333** | 0.293 | 0.767 | 0.720 | 0.633 |

1. **Reranking is net harmful, and the more influence it has the worse it gets.**
   Across all 48 (split × reranker × depth) configs, reranked MRR beats dense in 5 and
   loses in 43; on the individual recall cells that reranking can actually move, 16 beat
   dense and 137 lose. Depth-100 is the worst config in 11 of 12 split×reranker pairs.
   bge-reranker-base decays strictly monotonically with depth on all six splits, dense
   being the maximum; bge-reranker-large is monotone on standard_L2/L3/L4 and dense is
   its maximum on 4 of 6 splits (the exceptions are both split D variants, see point 3).
2. **It is a weaker ranker, not a broken one.** The control that settles this: if gold
   is in the top-100 (prob = R@100) and that list were shuffled uniformly, its rank
   would be uniform on 1..100, so expected R@k = R@100 × k/100. A sign or indexing bug
   would push gold *below* that line. Observed, at depth 100 with bge-reranker-large:

   | split | random R@10 | reranked R@10 | dense R@10 |
   |---|---:|---:|---:|
   | standard_L2 | 0.085 | 0.553 | 0.653 |
   | standard_L3 | 0.083 | 0.453 | 0.553 |
   | standard_L4 | 0.083 | 0.460 | 0.600 |
   | splitD_separable | 0.093 | 0.633 | 0.767 |

   The reranker lands **5.5–7.8× above random but at 0.77–0.90× dense** on every split.
   It has substantial real signal — just less than gte-large/v3. That rules out a sign
   error, an indexing bug, and a prefix mistake, and makes this a finding about relative
   ranking quality. Two further controls: the reorder code satisfies all 36 mechanical
   invariants exactly (reordering the top-d cannot move gold across the rank-k boundary
   for k ≥ d, so depth-10 must leave R@10 and R@25 bit-identical to dense — it does),
   and the dense rows reproduce §6 (L2 R@10 0.653 dev vs 0.644 all-pool).
3. **Cause: document distribution — inferred, not measured.** BGE rerankers are trained on
   prose passages (MS MARCO and similar). Our documents are concatenated API metadata —
   `name summary description paramnames tags api_title provider GET v1 timezone area
   location region txt`. Cross-encoders attend jointly over query and document and are
   far more sensitive to that mismatch than bi-encoders, which is exactly the asymmetry
   observed: the same v3 text that *helps* every bi-encoder is what the cross-encoder
   chokes on.
4. **The one positive is where theory predicts it.** bge-reranker-large at top-10 on
   splitD_separable: R@1 0.267 → 0.333. Near-miss discrimination between two
   near-identical tools is the one place joint attention should beat independent
   embedding, and it is the only split with a real gain. But n=150 (95% CI ≈ ±7.5 pts),
   it is absent for the base model, and it does not generalize to the standard splits.
   Treat as a hypothesis, not a result.
5. **Latency would have disqualified it anyway.** bge-reranker-large costs 3.3–5.0
   s/query at depth 100 on MPS. Even a winning reranker at that cost is not deployable
   in an interactive agent loop.

**Reading — superseded by §8, read both.** The §4d redirect listed "a cross-encoder
reranker over the top-100" as having "real headroom to exploit." What §7 actually
rejects is **reranking by replacement**, which is the only thing it tested: inside the
top-d it sorted purely by cross-encoder score and discarded the dense ordering. §8
fuses the two scores instead and the conclusion reverses. §7's measurements stand; its
framing as a verdict on *reranking* was too broad.

## 8. Score fusion — the §7 verdict reverses

`scripts/fusion_sweep.py --partition dev --n 150 --device mps`. **~70 min, local.**
→ `data/eval/fusion_sweep.json`, scores cached to `fusion_scores_dev_n150.npz` so the
α × depth × rule sweep re-runs in seconds.

§7.2 established the reranker is a *weaker* ranker, not a broken one (5.5–7.8× above
random). Replacement therefore throws away a stronger signal to adopt a weaker one.
Fusion keeps both, with α=0 reducing to dense and α=1 to pure rerank:

```
zscore  fused = (1-α)·z(dense) + α·z(rerank)              per-query z over the window
rrf     fused = (1-α)/(60+rank_dense) + α/(60+rank_rerank)  rank-based, scale-free
```

**Fixed config applied uniformly — bge-reranker-large / zscore / full depth / α=0.2:**

| split | n | R@1 dense → fused | R@10 dense → fused | MRR dense → fused |
|---|---:|---:|---:|---:|
| dev_A (MetaTool anchor) | 150 | 0.567 → 0.580 **+.013** | 0.813 → 0.847 **+.033** | 0.650 → 0.663 |
| standard_L2 | 150 | 0.280 → 0.313 **+.033** | 0.653 → 0.667 **+.013** | 0.401 → 0.430 |
| standard_L3 | 150 | 0.207 → 0.253 **+.047** | 0.553 → 0.573 **+.020** | 0.328 → 0.369 |
| standard_L4 | 150 | 0.280 → 0.233 **−.047** | 0.600 → 0.620 **+.020** | 0.371 → 0.354 |
| splitD_separable | 150 | 0.267 → 0.327 **+.060** | 0.767 → 0.780 **+.013** | 0.451 → 0.494 |
| splitD_inseparable | 39 | 0.231 → 0.256 **+.026** | 0.692 → 0.744 **+.051** | 0.409 → 0.425 |

Generated-split means: **R@1 0.253 → 0.277 (+9.4%)**, R@10 0.653 → 0.677 (+3.6%),
MRR 0.392 → 0.414 (+5.7%). Pure replacement at α=1 is −0.039 R@1 — i.e. the same
cross-encoder scores either help or hurt depending only on how they are combined.

**1. The depth effect inverts, which is the mechanism confirmed.** Mean MRR over the
generated splits (dense = 0.392):

| fuse/rerank window | α=0.2 (fusion) | α=1.0 (replacement) |
|---|---:|---:|
| top-10 | 0.400 | 0.381 |
| top-25 | 0.404 | 0.359 |
| top-50 | 0.409 | 0.348 |
| top-100 | **0.414** | 0.332 |

Replacement decays monotonically with depth and is below dense everywhere; fusion
*rises* monotonically with depth and is above dense everywhere. Under replacement more
depth means more of a good ordering destroyed; under fusion it means more candidates
the reranker can contribute evidence about. §7's depth decay was an artifact of the
combination rule, not a property of the reranker.

**2. Both fusion rules agree, so it is not a normalisation artifact.** zscore and rrf
share no scale assumptions — one standardises, the other discards magnitude entirely —
yet both peak at α≈0.2 with nearly identical gains (0.414 vs 0.411 mean MRR). The
optimum is a broad plateau, not a knife-edge: the top 8 of 176 configs fall within
0.0035 MRR.

**3. α≈0.2 is where the §7.2 ratio predicts.** The reranker measured at 0.77–0.90× dense
quality, so the optimum should weight dense heavily but not exclusively. It does.

**4. The latency objection (§7.5) survives, and this is the practical catch.** The gain
scales with depth, and so does the cost: full-depth fusion buys +0.022 MRR at 3.3–5.0
s/query, while top-10 fusion — cheap enough to deploy — buys only +0.008. There is no
configuration that is both worthwhile and fast. **Fusion overturns the scientific claim
in §7 without overturning the engineering one.**

**Limits.** α and the rule were selected on dev, which is dev's purpose, but the fixed
config above is still the argmax of a 176-config sweep, so expect regression to the mean
on test. n=150/split (39 for inseparable) and the script stores aggregates, not per-item
ranks, so no paired significance test is possible; the evidence is consistency —
5/6 splits improve on R@1 and MRR, 6/6 on R@10. **standard_L4 regresses** (R@1 −0.047),
unexplained, and it was also the split where replacement did worst.

## 9. Dispositions — what each proposal became

Kept for the item numbers that §10–§12 refer back to. Detail lives in the section
that settled each one; nothing here is a plan.

1. **Hybrid dense+BM25 fusion** — ran, negative. §10.
2. **Arm 2, description rewriting** — screened out, do not run as scoped. §11.
3. **k-sweep for the operating point** — **not run.** Selection held ~0.93 at k=20
   (§4d) and retrieval is now at R@25 0.68–0.82, so k≈25 remains a provisional
   choice, not a measured one. This is the open item that matters most.
4. **One test-partition run, on the composed pipeline only** — not run. If it is:
   encoder + fusion + chosen k together, not the encoder alone (§6 methodological
   debt), and not the §8 argmax config without expecting regression to the mean.
5. **Persist per-item ranks** — done, §10 natively and §7–§8 back-filled in §12.
   The tests changed the status of two of the four claims they touched. **Rule: no
   number from §6–§12 gets quoted or carried into a test run without an interval.**
6. **Why `standard_L4` regresses under fusion** — open, with a lead. §12.5 shows it
   is hostile to the cross-encoder in *every* configuration and significantly so
   (replacement −0.180 R@1, p=0.00003), the only split where replacement is
   significant at any depth. Hypothesis: L4's decoy context is attended to by a
   cross-encoder and averaged away by a bi-encoder.
7. **Screen new retrieval signals on union recall headroom before sweeping them**
   (§10.7). Minutes of work, and it would have predicted §10's negative in advance.
8. **Not pursued:** reranking by replacement (§7), BM25 fusion (§10), arm 2 as
   scoped (§11). Arms 4–5 remain dropped (§4d).
9. **H1 scored, not left open** (§11.3). Its +10–25 pt prediction exceeds the
   headroom available in either stage: refuted by headroom.

Everything in §6–§12 ran on local models.

## 10. BM25 × dense fusion — measured, negative (2026-10-02)

`scripts/bm25_fusion.py --partition dev --n 150 --device mps`. **~2 min, local.**
→ `data/eval/bm25_fusion.json` + `bm25_fusion_ranks_dev_n150.npz` (per-item ranks).

§9 item 1 promoted this as "the cheapest high-value run": §8 showed the fusion *rule* is
what pays, BM25 is the other weak decorrelated signal on hand, and it has none of the
§8.4 latency problem. **It does not pay.** Two variants were run, because the §8 harness
fuses only inside the dense top-100 and that window silently caps what BM25 can do:

- **(a) window fusion** — inside the dense top-100, `reorder` logic identical to §8
  (asserted element-wise against `fusion_sweep.reorder`, guardrail C). Ceiling is dense
  R@100: fusion can only *reorder* golds dense already found.
- **(b) full-catalog fusion** — both signals scored over all 3,551 tools, fused before the
  top-k cut. BM25 is sparse and instant so this is affordable; a cross-encoder could never
  do it, which is why §8 never tested it. This variant can *rescue* a gold dense missed
  entirely, which per §4d is ~83% of the error mass.

(a) is the control that makes (b) interpretable. Sampling is byte-identical to §8 (same
SEED, split order, rng draw order), and the dense block reproduces `fusion_sweep.json`
on all six splits.

**BM25 alone, and the rescue headroom that motivated (b):**

| split | n | bm25 R@1 | bm25 R@10 | bm25 R@100 | dense R@100 | **union R@100** |
|---|---:|---:|---:|---:|---:|---:|
| dev_A | 150 | 0.280 | 0.420 | 0.593 | 0.940 | 0.940 |
| standard_L2 | 150 | 0.087 | 0.227 | 0.460 | 0.853 | **0.893** |
| standard_L3 | 150 | 0.013 | 0.093 | 0.220 | 0.827 | 0.827 |
| standard_L4 | 150 | 0.040 | 0.133 | 0.340 | 0.827 | 0.833 |
| splitD_separable | 150 | 0.087 | 0.200 | 0.433 | 0.933 | 0.933 |
| splitD_inseparable | 39 | 0.205 | 0.333 | 0.615 | 0.949 | 0.949 |

**1. There is no rescue headroom, so (b)'s premise is empirically absent.** BM25's entire
top-100 holds fewer golds than dense's (R@100 0.22–0.62 vs 0.83–0.95), and the ones it
holds are nearly all *already* in dense's top-100: the union adds **+0.000 on four of six
splits**, +0.040 at best. Full-catalog fusion was the variant that could in principle beat
the window ceiling, and the data says there is essentially nothing out there to reach.

**2. The gains are small and not significant.** Best uniform config per variant, chosen on
mean MRR over the generated splits (not per-split argmax):

| variant | config | mean MRR (gen) | vs dense 0.392 |
|---|---|---:|---:|
| (a) window | `zscore / depth100 / α=0.3` | 0.401 | +0.009 |
| (b) full-catalog | `zscore / α=0.1` | 0.406 | **+0.014** |
| — for comparison, §8 cross-encoder | `zscore / depth100 / α=0.2` | 0.414 | +0.022 |

Full-catalog R@1 by split: L2 +0.007, L3 −0.013, L4 +0.000, splitD_sep +0.000,
splitD_insep +0.077, dev_A −0.027. **Pooled over the generated splits: R@1 +0.003, 95%
bootstrap CI [−0.013, +0.019], McNemar 14/12 discordant, p = 0.85.** This is the first
section with paired significance tests (§9 item 5, now implemented), and the first thing
they do is deny a result that the point estimate alone would have let through.

**3. It fails §8's own standard of evidence.** §8 argued its gain was real because zscore
and rrf — which share no scale assumptions — agreed within 0.003 MRR and both peaked at
α≈0.2. Here only zscore improves; **both rrf variants decline monotonically from dense at
every α** (best α=0.1: −0.017 rrf, −0.018 rrf_tie). Under the criterion §8 set for itself,
a one-rule-only gain is a normalisation artifact.

`rrf_tie` exists to make that disagreement fair. BM25 is sparse — only 0.595–0.773 of the
catalog scores nonzero — and `fusion_sweep.rank_of` breaks ties positionally, so every
zero-scoring tool got a distinct rank in *catalog order*, feeding arbitrary ordering into
rrf as evidence. `rank_ties` gives tied scores the worst rank in their group, which is what
a zero BM25 score means. The fix was warranted and **changed the result by 0.001**. So
rrf's failure is a genuine disagreement, not a harness artifact.

**4. α moves exactly where §8.3 predicts, which is the one thing that replicates.** §8
found α≈0.2 optimal for a signal measured at 0.77–0.90× dense. BM25 runs at 0.05–0.31×
dense R@1 on the generated splits, and the optimum drops to α=0.1. The relative-quality
rule governing fusion weight holds; BM25 simply lands below the useful floor.

**5. Latency was never the obstacle, and that is the sting.** BM25 costs **3.1–7.8
ms/query** over the full catalog versus 3,300–5,000 ms for the cross-encoder. This was the
candidate that would have been deployable. The §8.4 dilemma therefore hardens rather than
resolves: the only signal that helps is the one too slow to ship, and the one fast enough
to ship does not help.

**6. One suggestive cell, under-powered.** `splitD_inseparable` gains +0.077 R@1 and +0.052
MRR under both variants — the same split where §8's cross-encoder did best, and the only
split where BM25 alone is nearly competitive with dense (R@1 0.205 vs 0.231). Plausible
mechanism: inseparable twins differ in literal path/verb tokens that lexical matching sees
and embeddings average away. But n=39, McNemar 3/0, p=0.25. Hypothesis, not a result.

**7. The §8 lesson needs a floor.** *(see also §11–§12, which apply this screen to arm 2
and put significance tests on §7–§8)* "Combine weak signals, never substitute them" is still
right about the *rule*, but it is not unconditional in the *signal*: fusion cannot
manufacture information that is not there. A 0.8× signal fused; a 0.1–0.3× signal with no
complementary coverage does not. Future candidates (tags, popularity, graph priors) should
be screened on **union recall headroom first** — it costs minutes, and here it would have
predicted the negative result before any sweep was run.

## 11. Arm 2 headroom screen — descriptions are not where the error mass is (2026-10-02)

`scripts/doc_quality_headroom.py --device mps`. **~1 min, local.**
→ `data/eval/doc_quality_headroom.json`. Full dev partition, all items (not n=150),
`gte-large`/v3.

§10.7's rule, applied before spending: screen a lever on the error mass it can physically
reach. Arm 2 rewrites tool descriptions, so the question is whether retrieval failures
actually sit on poorly-documented tools. §9 item 2 justified arm 2 on catalog counts — "86
tools have no usable description, 441 score ≤1 on doc_quality" — which was never checked
against whether those tools are ever the *gold*.

**Pooled generated splits, dev, n=1,346. Bucketed by the gold tool's `doc_quality`:**

| bucket | n | share of items | R@1 | R@10 | R@25 | share of miss@10 |
|---|---:|---:|---:|---:|---:|---:|
| dq0 (no description) | 14 | 0.010 | 0.214 | 0.643 | 0.786 | 0.010 |
| dq1 (poor) | 35 | 0.026 | 0.257 | 0.514 | 0.686 | 0.034 |
| dq2 (fair) | 311 | 0.231 | 0.222 | 0.521 | 0.627 | **0.297** |
| dq3–4 (good) | 986 | **0.733** | 0.285 | 0.664 | 0.767 | **0.659** |
| overall | 1,346 | 1.000 | 0.269 | 0.627 | 0.733 | — |

**1. The tools arm 2 was proposed for are almost never the gold.** `dq≤1` is 3.6% of
items and carries **4.4% of the miss@10 mass**. Meanwhile 73.3% of items have a
*well-documented* gold, and those carry 65.9% of the misses. Retrieval is failing mostly
on tools whose descriptions are already fine, which no rewriting reaches.

**2. Headroom upper bound: +0.016 R@1 / +0.037 R@10** (pooled; if every weak bucket
retrieved as well as dq3–4). By split, +R@10 runs 0.029–0.097. This is a *loose ceiling*
in a known direction — doc_quality correlates with obscurity, and that part is intrinsic
difficulty — so the realisable gain is some fraction of it. For scale, the v3 document
text already delivered a **measured +6.0 R@10** (§6) for free, which is larger than arm
2's entire ceiling.

**3. H1 as pre-registered is unreachable.** H1 predicted +10–25 pts from description
rewriting. Retrieval affords at most +3.7 R@10, and §4d measured selection-given-a-good-
shortlist at ~0.93, leaving ≤7 pts there. Neither stage has 10–25 points to give. H1
should be scored as refuted-by-headroom rather than left open.

**4. The real soft spot is `dq2`, not `dq≤1`** — 23.1% of items, R@10 0.521 vs 0.664 for
dq3–4, carrying 29.7% of the miss mass. Same for description length: thin (≤5 words) golds
are 25.9% of items at R@10 0.509 vs 0.668 rich, carrying **34.1%** of misses. If arm 2 runs
at all it should target thin/fair documentation, not the empty-description tail.

**5. But the benchmark systematically under-samples the tools arm 2 would help**, so the
screen understates arm 2's production value. Representation as gold vs share of catalog:

| bucket | catalog share | gold share | ratio |
|---|---:|---:|---:|
| dq0 | 0.0245 | 0.0104 | 0.425 |
| dq1 | 0.1735 | 0.0260 | **0.150** |
| dq2 | 0.1870 | 0.2311 | 1.236 |
| dq3–4 | 0.6150 | 0.7325 | 1.191 |

`dq1` tools appear as gold at **0.150× their catalog rate** — a 6.7× under-sampling. The
mechanism is mechanical: `generate_intents.py` had to *read* a tool's documentation to
write an intent for it, so tools with no usable description were largely skipped. This is
a property of the eval set, not of the world. Rewriting cannot fix it; only regenerating
intents for those tools would, which costs money and introduces a fresh circularity (the
rewriter would be authoring the documentation the intent is then derived from).

**6. `dev_A` headroom is unmeasurable, not zero.** All 199 MetaTool tools are
`doc_quality ≤ 2` (175 at 1), so there is no well-documented reference bucket. The script
returns `null` and prints `n/a`; an earlier version printed `+0.0000`, which would have
been read as "no headroom on the anchor". This also means the only non-circular anchor
available cannot validate arm 2's effect size — just its direction.

**Rewrite scope, by label-independent criteria** (selecting tools by "is gold for some
eval intent" would leak labels into the index): `doc_quality≤1` 703 tools (mean 6.5 desc
words); `desc_words≤5` 1,079 (mean 3.0); `no_description` 144.

**Verdict: do not run arm 2 as scoped in §9 item 2.** The premise does not
hold on this benchmark. If it is run, retarget it at thin/`dq2` documentation and
pre-register the expectation as ≤+0.037 R@10, not H1's +10–25.

## 12. Significance tests back-filled onto §7 and §8 (2026-10-02)

`scripts/significance_backfill.py`. **Seconds, no recompute** — it re-reads
`fusion_scores_dev_n150.npz`, which holds the per-query × per-candidate score matrices and
is therefore strictly more informative than the aggregates §7/§8 persisted.
→ `data/eval/significance_backfill.json` + `significance_backfill_ranks.npz` (534 per-item
rank vectors). Closes §9 item 5.

McNemar exact (two-sided binomial on discordant pairs) for R@1, plus a 10,000-sample
paired bootstrap CI over items for R@1 and MRR. All comparisons are paired on identical
queries. **Guardrail: 30 derived aggregate blocks reproduce the persisted ones**, including
the check that α=1 zscore fusion reproduces `rerank_sweep.json` depth-by-depth — which is
what makes these tests apply to §7's claim and not only §8's.

**Pooled over the generated splits (n=639):**

| claim | ΔR@1 | R@1 95% CI | McNemar p | ΔMRR | MRR 95% CI |
|---|---:|---|---:|---:|---|
| §8 fusion, zscore depth100 α=0.2 | +0.024 | [−0.005, +0.052] | 0.124 | +0.023 | **[+0.007, +0.041]** |
| §8 fusion, rrf depth100 α=0.2 | +0.017 | [−0.005, +0.039] | 0.161 | +0.017 | **[+0.003, +0.031]** |
| §7 replacement, depth100 α=1 | **−0.055** | **[−0.092, −0.017]** | **0.005** | −0.072 | **[−0.103, −0.042]** |
| §7 replacement, depth10 α=1 | −0.013 | [−0.052, +0.027] | 0.573 | −0.015 | [−0.043, +0.012] |

**1. §8's headline R@1 claim does not survive the test; its MRR claim does.** The reported
"+9.4% R@1" (0.253 → 0.277) is +0.024 pooled at **p = 0.124**, and *no individual split*
reaches p<0.05 on R@1 — the best are splitD_separable (+0.060, p=0.136) and standard_L3
(+0.047, p=0.144). The MRR gain is significant pooled, under **both** fusion rules, with
CIs excluding zero. **§8 should be restated in MRR terms.** The effect is real but it is a
ranking-quality effect, not a demonstrated top-1 effect at n=150.

**2. §8's "both rules agree" argument gets stronger, not weaker.** zscore and rrf now
agree on *significance* as well as magnitude for MRR (+0.023 and +0.017, both CIs above
zero) while both fail to establish R@1. The agreement §8 claimed is confirmed; what shifts
is which metric carries it. Contrast §10, where only one of three rules improved at all.

**3. §7's negative conclusion holds at depth, and dissolves at shallow depth.**
Replacement at depth 100 is significantly harmful (R@1 p=0.005, MRR CI [−0.103, −0.042]).
At depth 10 it is **indistinguishable from dense** (p=0.573). §7's monotone-decay story is
intact, but "reranking by replacement hurts" is only established for deep windows; shallow
replacement is neutral, not harmful.

**4. The α plateau is real and α=0.2 is not uniquely optimal.** Pooled, zscore, depth100 —
MRR delta with bootstrap p(Δ≤0): α=0.1 +0.012 (p=0.023), **α=0.2 +0.023 (p=0.002)**, α=0.3
+0.017 (p=0.047), α=0.4 +0.004 (p=0.373), and significantly negative from α≥0.7. So
α∈[0.1,0.3] is a genuine plateau whose members are not distinguishable from each other —
§8's "broad plateau, not a knife-edge" is confirmed, and §8's warning not to carry the
exact α into production is reinforced.

**5. `standard_L4` is hostile to the cross-encoder in every configuration, and now
significantly so** — replacement depth100 **−0.180 R@1, p=0.00003**; replacement depth10
−0.140, p=0.0003; fusion −0.047, p=0.092. It is the only split where replacement is
significant at any depth. L4 is the *easiest* standard level for dense retrieval (§4c),
which makes this the sharpest available clue for §9 item 6: the distractor-context queries
are not merely unhelped by joint attention, they are actively mis-ranked by it. A
hypothesis worth one targeted look: L4 queries carry decoy context that a cross-encoder
attends to and a bi-encoder averages away.

**6. Methodological consequence.** Two of the four claims tested here changed status on
contact with a paired test that cost seconds and no recompute. §10 already showed the same
thing in the other direction. **Any number from §6–§10 that is going to be quoted, or
carried into the test-partition run, should carry an interval.**

## 13. Pre-publication redaction (2026-10-02)

`scripts/redact_secrets.py`. **14 strings removed** before the repo was prepared for
GitHub → `data/eval/redaction_report.json` (cumulative across runs).

| class | n | files | real? |
|---|---:|---|---|
| AWS access key ID | 2 | `corpus/catalog.jsonl`, `catalog/catalog.jsonl` | **real, third-party** — arrived inside SYNQ.fm's public OpenAPI spec as an S3 upload example |
| JWT / base64 blob | 4 | catalog ×2, `standard_L2`, `standard_L4` | catalog ones real (the S3 POST policy); generated ones fictional |
| Stripe-style key | 4 | `standard_L2`, `standard_L4` | fictional, generator-invented |
| Slack token | 2 | `standard_L4` | fictional (obviously synthetic: a Slack user-token prefix followed by `-8821-adminaudit`) |
| Bearer literal | 1 | `standard_L4` | fictional |
| Home-directory path | 1 | `standard_L4` | fictional — not a credential; see below |

The home-directory path (`standard_L4_951`) was a fabricated local file path whose directory
chain carried an invented surname next to a synthetic "veteran client / medical evidence"
request. It refers to nobody, but it *reads* like PHI and a reviewer skimming a public repo
cannot tell that at a glance. Collapsed to its basename (`./medical-evidence.pdf`): the
filename is what the task is about — validating a PDF before upload to a VA.gov endpoint —
so the gold label and the difficulty are untouched. This rule is scoped to
`data/generated/` only: the catalogs' own `/home/foo/...` strings are generic documentation
placeholders in a third-party spec, and rewriting catalog text would alter the retriever's
input for no privacy benefit.

An AWS access key *ID* is the non-secret half of a keypair and was already public in that
spec, so this was never a breach. It was redacted because it belongs to a third party and
because GitHub secret scanning flags the pattern on push. The fictional tokens were redacted
for the same mechanical reason: scanners match on **prefix shape**, so a fake `xoxp-` string
generates the same alert as a real one. Replacements deliberately break the matchable prefix.

**Scientifically free.** For generated intents the label is the *tool*, never the token
string, so the redacted text supports an identical task. Row counts are unchanged and every
line still parses: catalog 3,551 / 3,352, generated 500 / 1,120 / 987 / 996 / 982 — all
matching §2 and §4c. The single affected catalog tool (`synq_fm_upload`, doc_quality 4)
retains a 60-word description.

**One consequence, recorded not hidden.** `data/eval/*.npy` were built from the
*pre-redaction* catalog, so one row's document text no longer matches the matrix that
encoded it, and `catalog_sha256` in the `.meta.json` sidecars no longer matches. The `.npy`
files are gitignored, so a fresh clone regenerates them from the redacted catalog and is
self-consistent; only exact reproduction of the numbers above is affected, by **one row of
3,551**. Do *not* regenerate `data/eval/tool_embeddings.npy` to "fix" this — split D's
confusion pairs are defined against that exact matrix.
