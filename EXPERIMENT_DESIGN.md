# Improving Agent Tool/API Selection from User Intent

**Experiment design v0.1 — 2026-09-27**

Status: draft for review. Open questions flagged with ❓ throughout; see §16.

---

## 1. Problem formalization

Given a user intent `u` (natural language, optionally with conversation history `h`)
and a tool catalog `C = {t_1 ... t_N}` where `N ≥ 200` and `C` changes over time,
produce a tool call `(t*, args*)` — or correctly abstain.

Decompose into three sub-decisions, because they fail for different reasons and
need separate measurement:

| Stage | Decision | Failure mode |
|---|---|---|
| **S1 Retrieve** | shortlist `C_k ⊂ C`, `|C_k| = k` | right tool not in shortlist |
| **S2 Select** | pick `t* ∈ C_k ∪ {∅}` | plausible-but-wrong neighbor; hallucinated name; failure to abstain |
| **S3 Ground** | fill `args*` from `u` | schema-invalid; wrong values; hallucinated params |

**Scope decision:** phase 1 is single-call. Multi-call sequencing is phase 2 (§13).
Rationale: you cannot diagnose sequencing errors until single-call selection is
solid, and mixing them makes every metric ambiguous.

### Task variants to measure separately

- **Unambiguous** — exactly one correct tool.
- **Ambiguous** — several tools are defensible; gold is a *set*. Score with set membership, not exact match.
- **Underspecified** — correct behavior is to ask a clarifying question.
- **Unsupported** — no tool applies; correct behavior is abstention.

Collapsing these into one accuracy number is the most common way tool-selection
evals mislead. Report them broken out, always.

---

## 2. Hypotheses

Pre-register these with predicted direction and rough effect size. Predictions are
deliberately committed *before* running — the point is to notice when you're
surprised.

| ID | Hypothesis | Predicted |
|---|---|---|
| **H1** | Rewriting tool descriptions/schemas closes a large share of the error gap with no model changes | +10–25 pts tool-selection accuracy over arm 1 |
| **H2** | Retrieval shortlisting improves precision beyond what fits-in-context prompting achieves, and there is an interior optimum `k` | inverted-U in `k`; best `k` ≈ 10–25 |
| **H3** | An explicit intent-classification layer adds little *once* retrieval is good | < 3 pts over arm 3; may be *negative* via error cascade |
| **H4** | SFT on (intent → call) pairs beats prompting for in-distribution intents but degrades on held-out tools | +8–15 pts seen-tools; **−5 to −15 pts unseen-tools** vs arm 3 |
| **H5** | GRPO over shortlists adds most of its value on abstention and argument grounding, not raw tool-match | +1–4 pts tool-match; **+10–20 pts abstention F1** over arm 4 |
| **H6** | A well-prompted frontier model (arm 2/3) is competitive with a trained small model at materially different cost/latency | the actual business question |

H3 and H4 are written to be *falsifiable and unflattering*. If every hypothesis
predicts your preferred method wins, the design is decorative.

---

## 3. Experimental arms

Cumulative ladder — each arm inherits everything below it. Measure each arm
against the one immediately beneath, not against arm 1.

| Arm | Condition | Marginal question |
|---|---|---|
| **0** | Random / BM25-over-descriptions | Floor |
| **1** | Base model, catalog as-written, all tools in context (or truncated) | True starting point |
| **2** | + rewritten descriptions, params, negative examples | How much is just bad docs? |
| **3** | + embedding retrieval, top-k shortlist | Does narrowing help? At what k? |
| **4** | + SFT on (intent → call) pairs | What does supervised fitting buy? |
| **5** | + GRPO over shortlists | What does RL add beyond SFT? |
| **5-alt** | + DPO on preference pairs | Half the benefit for a tenth the effort? |
| **6** *(optional)* | Online contextual bandit in production | Deployable variant — see §14 ❓ |
| **∞** | **Oracle retrieval** (gold tool always in shortlist) | Ceiling: isolates S2+S3 from S1 |
| **∞∞** | **Human expert** on a 200-item subsample | Absolute ceiling + label-noise estimate |

The two oracle arms are not optional. Without arm ∞ you cannot tell whether a
failure is retrieval or selection, which is exactly the diagnosis that gates
whether arm 5 is worth building.

### Critical design choice: RL over shortlists, not over tool identities

Arm 5 trains the model to discriminate among *candidate tool documents presented
in context*. Tool identities never enter the weights. This preserves zero-shot
generalization to catalog changes — the whole point in an open catalog — and it
makes arms 3 and 5 compose rather than compete.

Train with the shortlist in the prompt, varying `k` and candidate ordering per
rollout so the policy can't latch onto position.

---

## 4. Data

### 4.0 Sources

> **Status: acquired and validated — see `DATA.md`.** 3,551-tool corpus built
> (199 labeled MetaTool + 3,352 APIs.guru distractors), 20,532 labeled intents,
> splits cut, arm-3 retrieval baseline run. Two corrections to the table below,
> both confirmed by fetching: APIs.guru is 2,529 APIs / **108,837 endpoints**, and
> **τ-bench is deprecated** — use τ²/τ³-bench (`sierra-research/tau2-bench`).
> One gap found: MetaTool has **zero** tool pairs above 0.85 similarity, so split D
> is not buildable from it and must be generated over APIs.guru (DATA.md §4).

> **No public benchmark tests this problem.** Nearly all of them present 1–10
> candidate tools per query, i.e. S1 is trivially solved and only S2/S3 are
> measured. Used as-is they would silently delete the retrieval arm.

**Distractor pooling.** Take gold `(intent, tool)` pairs from any source, pool all
tools from all sources into a single 1,000+ tool index, and re-run. Retrieval
becomes load-bearing with zero new labels. Report at pool sizes {50, 200, 1000,
full}; the degradation curve is a result in itself.

| Tier | Source | Use | Caveat |
|---|---|---|---|
| **Catalog** | Own catalog | primary, if it exists | — |
| | APIs.guru (~4k OpenAPI specs) | sample 200–500 tools | real doc-quality variance — good, it's what makes H1 measurable |
| | MCP registries (modelcontextprotocol, Smithery, Glama) | currency; real churn for split G | moving target |
| **Train** | xLAM-function-calling-60k (APIGen) | SFT + GRPO bulk | LLM-generated; APIGen ≈ our §4.2 pipeline |
| | Glaive function-calling v2 (~113k) | SFT volume | lower quality |
| | ToolACE | API breadth | synthetic |
| **Eval** | BFCL v3/v4 | incl. *relevance detection* → split E; AST matching | contaminated |
| | MetaTool (~199 tools) | closest public match: whether-to-use + which | small |
| | AppWorld (457 APIs, stateful) | most realistic; best phase-2 fit | heavier harness |
| | τ-bench / τ²-bench | multi-turn, user simulator, end-to-end | only 2 domains |
| | ToolBench / RapidAPI (16k+) | catalog scale, distractor pool | many dead endpoints; ChatGPT-generated instructions — **not ground truth** |
| | API-Bank, ToolSandbox, Seal-Tools | supplementary | — |

**Recommended default (no labeled traffic):** APIs.guru catalog + own
reverse-generated intents (§4.2) as the *primary* set — the only data matching your
domain and provably uncontaminated — with BFCL + MetaTool + AppWorld as
external-validity anchors, distractor pooling applied throughout.

**Three reading rules:**
1. Public benchmarks predate current pretraining. Quote *relative ordering only*,
   never an absolute number.
2. ToolBench is a catalog and distractor source, not a label source.
3. xLAM/ToolACE are LLM-generated — training on them and evaluating a same-family
   model inherits the generator's fingerprint. Keep one human-validated slice (§4.3).

*Sizes and licences above are from recall and need verification before commitment.*

### 4.1 Catalog normalization

Before any generation: build a canonical `catalog.jsonl`, one record per tool.

```
tool_id, name, description, parameters (JSON Schema), returns,
auth_scope, side_effects (read|write|destructive),
domain, aliases, deprecated_at, version
```

Two things to record during normalization, because they become findings:
- **Doc quality score** per tool (has description? param docs? examples? ≥ 20 words?).
  Expect this to correlate hard with per-tool accuracy — that's H1's mechanism.
- **Confusability graph**: cosine similarity between tool doc embeddings. Any pair
  above ~0.85 is a confusion candidate and should be over-sampled into the eval set.

### 4.2 Intent generation (assuming no labeled traffic ❓)

Reverse-generation pipeline:

1. **Generate.** For each tool, generate intents across a difficulty ladder:
   - L1 near-paraphrase of the description
   - L2 natural user phrasing, no tool vocabulary
   - L3 indirect/goal-stated ("I need to know why my invoice bounced")
   - L4 with distractor context (mentions entities relevant to a *different* tool)
   - L5 multi-hop or ambiguous between confusable pairs
2. **Round-trip filter.** Have a *different* model family recover the tool from the
   intent alone, given the full catalog.
   - recovered by everything → too easy, downsample L1
   - recovered by nothing → either genuinely hard (keep, mark L5) or the intent is
     broken/ambiguous (drop). Disambiguate by human review.
3. **Hard negative mining.** For each intent, store the top-5 non-gold retrieved
   tools. These become DPO rejected candidates and the adversarial eval slice.
4. **Negative/abstention set.** Generate intents deliberately *outside* catalog
   coverage — ~10% of the dataset. Without this, abstention is unmeasurable and
   every model scores 100% by never abstaining.
5. **Ambiguity set.** For high-similarity tool pairs, generate intents that are
   genuinely satisfiable by either. Gold = the set. ~5%.

> ⚠️ **Circularity control.** Never use the same model family to generate intents
> and to evaluate. Generated intents carry the generator's lexical fingerprint, and
> a same-family evaluatee scores several points high for free. Generate with model
> family A, evaluate B and C, and report a cross-generator sanity check on a
> 500-item subsample generated by B.

### 4.3 Human validation

Not optional — it sets your measurement ceiling.

- Sample **500** items stratified by difficulty level and by tool.
- **2 annotators independent + 1 adjudicator.** Report Cohen's κ.
- If κ < 0.7, the task definition is unclear, not the annotators — rewrite the
  guidelines before generating more data.
- Output: **label noise rate**. If it's 8%, then 92% is your effective ceiling and
  a difference between two arms both scoring ~90% is meaningless. State this number
  in every results table footer.

### 4.4 Volumes

| Purpose | Size |
|---|---|
| SFT train | 5k–10k |
| GRPO prompts | 8k–20k (× 8–16 rollouts) |
| DPO pairs | 5k–15k |
| Dev | 1.5k |
| Test (frozen, single-use per arm) | 3k |
| Human-validated subsample | 500 |

Coverage floor: ≥ 50 intents/tool, ≥ 100 for tools in high-similarity clusters.

### 4.5 Public benchmark anchoring

Run arms 1–5 on **BFCL v3**, **τ-bench**, and **API-Bank** as an external-validity
check. Do not train on them. Purpose is not to top a leaderboard — it's to detect
that your synthetic pipeline produced a distribution where everything looks great
and nothing transfers. A large gap between your internal set and public benchmarks
is a finding about your data, not about the models.

Assume public benchmarks are partly contaminated in pretraining. Treat them as a
lower bound on relative ordering, not an absolute measure.

---

## 5. Splits — the thing most likely to make your results wrong

| Split | Construction | What it answers |
|---|---|---|
| **A. Seen tools / seen intent types** | random | fitting sanity check |
| **B. Seen tools / novel phrasings** | held-out L2–L4 paraphrases | language robustness |
| **C. Unseen tools** | **20% of tools held out entirely** from all training | **generalization — the real number** |
| **D. Confusion pairs** | intents over similarity > 0.85 tool pairs | discrimination |
| **E. Abstention** | out-of-catalog intents | over-triggering |
| **F. Ambiguous** | set-valued gold | reasonable-choice handling |
| **G. Catalog drift** | mutate descriptions / rename / add 20 new tools post-training | production realism |

Rules:
- Split **by tool first**, then by intent. A random intent split leaks tool identity
  into training and will overstate arms 4–5 substantially.
- Dedup near-duplicate intents *across* splits (embedding cosine > 0.95) before
  freezing. Paraphrase leakage is the silent killer in synthetic datasets.
- Freeze test. One evaluation per arm. Iterate on dev only. Log every test-set
  access with a timestamp and arm ID.

Split C is the headline. If arms 4–5 win on A but lose on C, the honest conclusion
is "training helps in-distribution and hurts under catalog change," and for an open
catalog that's a reason not to ship it.

---

## 6. Metrics

### Retrieval (S1)
`recall@k` for k ∈ {1,5,10,25,50,100} · `MRR` · `nDCG@k` · shortlist token cost

### Selection (S2)
- Tool accuracy (exact) — primary
- Tool-family accuracy (partial credit for right family, wrong member)
- Hallucination rate (name not in catalog)
- **Abstention precision / recall / F1** — the arm-5 headline
- Over-triggering rate on split E

### Grounding (S3)
- Schema validity (binary, free to compute)
- Exact arg-set match
- Per-parameter F1 with value normalization
- Required-param omission rate
- Hallucinated-param rate

### End-to-end (phase 2)
Task success · turns to success · redundant calls · **irreversible wrong actions**
(weight these separately — a wrong `read` and a wrong `delete` are not the same error)

### Calibration & selective prediction
- ECE
- **Risk–coverage curve + AURC.** More decision-relevant than raw accuracy: the
  operational question is "at what confidence threshold can we auto-execute vs.
  route to a human," and AURC answers it directly.

### Operational
p50/p95 latency · cost per query · prompt+completion tokens · training compute

### Robustness
- Paraphrase invariance (variance across 5 paraphrases of the same intent)
- **Position-bias delta**: same shortlist, gold at position 1 vs. position k
- Catalog-drift degradation (split G)

---

## 7. Ablations

| Ablation | Sweep | Isolates |
|---|---|---|
| Shortlist size | k ∈ {1,5,10,25,50,all} | H2's inverted-U |
| Description components | name → +desc → +params → +examples → +negative examples | which part of the rewrite did the work |
| Candidate ordering | gold-first / gold-last / randomized | position bias |
| Retriever | BM25 / dense / hybrid / reranked | retrieval contribution |
| Reward components | drop each term from §8 | reward design |
| Rollouts per prompt | 4 / 8 / 16 | GRPO compute scaling |
| Model scale | small / mid / frontier | H6 |

The description-component ablation is the highest-value one in the whole design and
costs almost nothing. "Negative examples in descriptions" is frequently the single
largest lever and is unglamorous enough that people skip measuring it.

---

## 8. RL specifics (arm 5)

**Algorithm:** GRPO — group-relative, no value model. PPO's critic is infrastructure
you don't need at this scale. RLOO is an acceptable substitute.

**Reward:**

```
r =  w1 · tool_match            # 1.0 exact, 0.3 right family
   + w2 · schema_valid          # binary
   + w3 · arg_F1                # per-param, value-normalized
   + w4 · abstain_correct       # on split-E-style negatives
   − p1 · hallucinated_tool     # large
   − p2 · redundant_calls
```

Start `w = [1.0, 0.2, 0.5, 1.0]`, `p = [2.0, 0.3]`. Ablate per §7.

**Reward-hacking probes** — run all three every checkpoint:
1. **Prior collapse.** Evaluate on rare-gold-tool inputs. If output entropy over
   tool choice drops sharply, the policy is farming `arg_F1` by emitting the
   highest-prior tool. This is the predicted failure for this reward.
2. **Abstention gaming.** If `w4` is too high the policy abstains everywhere.
   Track abstention *precision* alongside recall, always paired.
3. **Schema-only degenerate.** A minimal always-valid call can score on `w2` alone.
   Check that `schema_valid` gains track `tool_match` gains.

**KL control.** Reference = the arm-4 SFT checkpoint. Track KL per step; a sudden
drop in generation diversity is the early warning for all three probes above.

**Checkpoint selection on dev split C (unseen tools), not split A.** Selecting on
in-distribution dev performance is how you ship a memorizer.

---

## 9. Statistical analysis

This is where tool-calling evals are usually weakest. Commit to it up front.

- **Seeds.** n ≥ 3 per trained arm; report mean ± std. A single-seed result on a
  3k test set is not a result — seed variance on RL runs routinely exceeds the
  effect sizes being claimed.
- **Paired tests.** All arms run on identical test items → use paired methods.
  **McNemar** for paired binary (tool correct y/n); **paired bootstrap** (10k
  resamples) for continuous metrics and all CIs.
- **Report intervals, not points.** Every headline number gets a 95% CI.
- **Multiple comparisons.** ~6 arms × ~7 splits. Apply **Holm–Bonferroni** within
  each metric family and say so. Designate **split C tool-accuracy** as the single
  primary endpoint; everything else is secondary/exploratory and labeled as such.
- **Power.** At n = 3000 and baseline ~70%, you have ~80% power to detect ~3.5 pts
  paired. Differences below ~3 pts are noise at this size — decide now whether
  that resolution is enough, because it determines test set size.
- **Effect sizes.** Report absolute deltas and error-reduction rate. p-values alone
  answer no decision you actually face.

---

## 10. Confounds to control explicitly

| Confound | Control |
|---|---|
| **Generator = evaluatee** | cross-family generation; cross-generator subsample (§4.2) |
| **Position bias** | randomize candidate order; measure the delta (§7) |
| **Prompt-length confound** — arm 3's gain may be "less distraction," not retrieval | control arm: random k tools of the same token budget |
| **Benchmark contamination** | anchor on internal set; treat public as relative-only |
| **Label noise ceiling** | §4.3; footer every table with the noise rate |
| **Test-set erosion** | frozen test, logged single access per arm |
| **Catalog version skew** | pin a catalog snapshot hash into every run's metadata |
| **LLM-as-judge bias** (if used for ambiguous/partial credit) | different family from evaluatee; validate judge against the 500 human labels and report judge–human agreement |
| **Difficulty imbalance across splits** | stratify by L1–L5; report per-level breakdown |

The prompt-length control is easy to forget and can flip H2's interpretation
entirely.

---

## 11. Error taxonomy

Hand-code ≥ 200 errors per arm against a fixed scheme. This is what actually gates
the next arm — the aggregate numbers tell you *whether* you improved, the taxonomy
tells you *what to build next*.

```
E1  retrieval miss            — gold not in shortlist
E2  wrong tool, right family  — near-miss discrimination
E3  wrong tool, wrong family  — intent misread
E4  hallucinated tool
E5  should have abstained     — over-triggering
E6  should have acted         — over-abstaining
E7  should have clarified
E8  arg: missing required
E9  arg: wrong value / bad grounding
E10 arg: schema violation
E11 gold is wrong / ambiguous — label noise, not model error
```

Track the E11 rate separately and subtract it before comparing arms.

---

## 12. Gating criteria (pre-registered stop rules)

Decide these *now*, so a disappointing arm produces a decision instead of a search
for a better framing.

- **After arm 2:** if error gap closes ≥ 50%, that's the headline. Continue, but
  reframe the program as "docs + retrieval," not "training."
- **After arm 3:** if E1 (retrieval miss) > 40% of remaining errors → **stop the
  training track.** Invest in the retriever. RL cannot fix a tool that isn't in
  the shortlist.
- **After arm 3:** if E2 (near-miss) > 35% of remaining errors → training is
  justified; proceed to arm 4.
- **After arm 4:** if split C (unseen tools) *regresses* vs. arm 3 by > 5 pts →
  do not proceed to arm 5 without first fixing generalization. RL on a memorizing
  base makes it worse, not better.
- **After arm 5-alt (DPO):** if DPO captures ≥ 70% of the projected GRPO gain,
  seriously question whether GRPO's infrastructure cost is justified.
- **Any arm:** if the gain is within the label-noise band (§4.3), report it as
  "no measurable difference." Do not narrate it as a win.

---

## 13. Phase 2 — multi-call / agentic

Only after phase 1 stabilizes.

Adds: a **mock execution layer** (deterministic stubs for all N tools, with seeded
failure injection), episode-level reward, credit assignment across turns, and three
new metrics — sequencing accuracy, stop-decision accuracy, recovery-from-error rate.

Reward shifts from per-call to outcome-based, which is where reward hacking gets
much harder to detect. Keep the per-call verifiable reward as a shaped auxiliary
term rather than discarding it.

---

## 14. ❓ Open: offline RL vs. online learning

Still unresolved from our discussion. Arm 6 (contextual bandit over shortlists,
learning from production outcomes) shares almost no infrastructure with arm 5:

| | Arm 5 (GRPO, offline) | Arm 6 (bandit, online) |
|---|---|---|
| Data | labeled (intent → call) pairs | instrumented production outcomes |
| Infra | rollout server, training cluster | logging, propensity scores, off-policy eval |
| Feedback latency | none | days |
| Risk | reward hacking | exploration harms real users |
| Deployability | medium | high |
| Publishability | high | low |

If the goal is a shippable improvement, arm 6 may dominate. If it's a research
result, arm 5. **This is the biggest unresolved fork in the design.**

---

## 15. Infrastructure & artifacts

**Build:**
- Eval harness — arm-agnostic, one config per arm, deterministic seeding
- Mock API layer — stubs for all N tools (needed for phase 2; useful for arg validation now)
- Trace logger — full prompt/response/shortlist/reward per item, queryable
- Versioning — catalog snapshot hash + prompt hash + data hash + code SHA in every run record

**Produce:**
- `catalog.jsonl`, `intents.jsonl` (with difficulty + provenance), split manifests
- Dataset card — generation method, generator model, validation protocol, κ, noise rate, known limitations
- Pre-registration doc (§2, §9, §12) — timestamped before arm 4 runs
- Per-arm results tables with CIs and noise-rate footers
- Error taxonomy counts per arm
- Reproduction scripts

**Rough budget:** arms 0–3 are days–weeks and cheap. Arm 4 is ~1 week. Arm 5 is the
dominant cost — 3–6 weeks including infra, and the majority of total compute. Human
validation is ~40–60 annotator-hours. Budget the gating criteria (§12) as real
off-ramps, not formalities: there is a genuine chance the honest answer arrives at
arm 3.

---

## 16. Open questions

1. ❓ **Data source** — §4.0 assumes *own catalog, no labeled traffic* and makes the
   synthetic pipeline the backbone. If real traffic exists, it should displace the
   synthetic set as primary and §4.2 becomes coverage-filling only. **Confirm which.**
2. ❓ **Offline RL vs. online bandit** (§14).
3. ❓ **How open is "open catalog"** — 200 versioned internal endpoints, or a
   marketplace that changes weekly? Changes how heavily split G is weighted.
4. ❓ **Success definition** — selection accuracy, end-to-end task completion, or an
   operational target (wrong-call rate in production)?
5. ❓ **Model access** — can you train weights, or is this prompt/retrieval-only plus
   a hosted fine-tune API? Arms 4–5 assume weight access.
6. ❓ **Irreversible actions in catalog** — if yes, asymmetric error costs need to be
   in the metrics from the start, not retrofitted.
