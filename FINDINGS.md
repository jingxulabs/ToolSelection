# Tool Selection from User Intent: What Worked, What Didn't

*Findings summary. Numbers trace to `DATA.md`; the plan they were measured against is
`EXPERIMENT_DESIGN.md`. Total cost: **$40.39**, almost all of it one-time data generation.*

## 1. Goal

When an agent has thousands of candidate tools, which stage actually fails — finding the
right tool, or choosing among a short list?

```
user intent → RETRIEVER (3,551 tools → ~25) → SELECTOR (25 → 1) → tool call
```

Public benchmarks present 1–10 candidates, so retrieval is trivially solved and **only
selection gets measured.** Production agents face thousands. We built the missing condition.

## 2. Data

| Source | Size | Role |
|---|---|---|
| APIs.guru | 3,352 real API operations | Realistic catalog, **no labels** |
| MetaTool | 199 tools, 20,532 labeled pairs | The labels (GPT-4-made, so independent of Claude) |
| Generated | 4,585 intents, $33.94 | Labels for the APIs.guru tools |

Merged into **3,551 tools — 199 labeled, 3,352 distractors (17.8:1).** One number justifies
the construction: recall@100 falls **0.973 → 0.861** going from 199 to 3,551 candidates. At
199, retrieval is solved and unmeasurable. At 3,551, **~14% of correct tools are unreachable
regardless of selector quality.**

Generated intents cover direct/indirect/distractor-context requests, near-identical
**confusion pairs**, and **unanswerable** queries where refusing is correct.

## 3. Design

Three choices that earned their keep:

- **Tools held out before intents** (39 of 199 entirely). Splitting intents randomly leaks
  tool identity and overstates generalization.
- **A pre-registered stop rule:** *if retrieval misses exceed 40% of errors, abandon the
  model-training track.* The most consequential sentence in the design.
- **Self-reported ambiguity.** The generator could answer `separable: false` — *"identical
  function and parameters; only the HTTP verb differs."* **10.1%** did. The selector
  independently flagged **32.7%** of those as "tied" vs **0.0%** of separable ones. Two
  unrelated mechanisms agreeing removed the need for a human annotation pass.

## 4. Results

| Intervention | Effect | Verdict |
|---|---|---|
| **Better encoder + richer document text** | **R@1 0.139 → 0.259** | ✅ The only large win |
| Fine-tune the *selector* (SFT/RL) | n/a — gate tripped | ❌ Dropped before building |
| Reranking by **replacement** | R@1 0.253 → 0.214, p=0.005 | ❌ Significantly harmful |
| Same scores **fused** at 20% weight | MRR +0.023 [+0.007,+0.041]; R@1 p=**0.124** | ⚠️ MRR real, R@1 not; 5 s/query |
| Add keyword (BM25) search | R@1 +0.003, p=**0.85** | ❌ Null |
| Rewrite tool descriptions | ceiling +0.037 R@10 | ❌ Screened out, unspent |
| Fine-tune the *retriever* | — | ⚠️ No result (fp16 NaN bug + thermal throttling) |

The decomposition that redirected everything: **~83% of errors are retrieval failures, ~7%
selection.** The selector is ~0.93 accurate given a good shortlist; the retriever's R@1 was
~0.15.

## 5. Five insights

**① The field is optimizing the stage that isn't broken.** Because benchmarks hand over ~10
candidates, published work tunes selection. Our pooled catalog showed the failure is
overwhelmingly retrieval. Any agent with a large tool catalog should verify which stage is
failing before optimizing either — force the right answer into the shortlist and see whether
accuracy jumps.

**② How you combine signals matters more than which signals you have.** Identical
cross-encoder scores: **fused → +9% R@1, substituted → −15%.** The tell is the depth curve —
fusion improves as you go deeper, replacement degrades. A signal measured at 0.8× your main
signal's quality is still worth keeping, just not worth trusting alone.

**③ But fusion has a floor: it reweights information, it cannot manufacture it.** BM25 fused
gave nothing — its top-100 held *fewer* correct tools than dense's and almost none dense
missed. **The union added zero correct tools on 4 of 6 splits.** Hence a cheap screening
test: before tuning any new signal, check whether `dense ∪ new_signal` reaches errors dense
alone can't. Minutes of work; it would have predicted both negative results in advance.

**④ Plausible mechanism plus catalog statistics is not evidence.** Three interventions were
justified that way and all three died against *where the error mass actually is.* Description
rewriting is the clean case — "441 badly-documented tools" sounded compelling, but those tools
are the correct answer for only **3.6% of queries and 4.4% of failures.** 73% of queries have
a *well-documented* correct tool and carry 66% of failures.

A subtler trap, generalizable to any LLM-generated eval set: badly-documented tools appear as
correct answers at **0.150×** their catalog rate — because the generator had to *read* a
description to write a request for it. **The benchmark structurally under-samples the very
tools the fix would help.**

**⑤ Point estimates in this regime are noise, and intervals are nearly free.** Paired tests
run off cached scores in seconds and **changed the status of 2 of 4 claims — including our
own headline.** The "+9.4% R@1" from fusion does not survive (p=0.124); the MRR gain does.

**Bonus paradoxes.** Difficulty isn't ordered — "indirect" requests are *harder* than
"distractor context" ones (R@10 0.564 vs 0.605). And you cannot threshold for "no tool
applies": unanswerable queries score **higher** top-1 similarity (0.479) than real ones
(0.445), so refusal needs a different mechanism entirely.

**The engineering bind.** The only intervention that helped costs 5 s/query and can't ship.
The only one fast enough to ship (3 ms) doesn't help. The deployable frontier is already at
its limit.

## 6. Next steps

1. **Fix document text before touching the model.** Index the HTTP method and the URL path
   split into words — `worldtimeapi` has endpoints differing *only* by path depth, invisible
   in prose. Worth **+6.0 R@10** on average, positive in 24 of 25 cases, and biggest on thin
   descriptions (**0.348 → 0.568**). Model size barely mattered; `bge-large ≈ bge-base`.
2. Run the k-sweep for the operating point; spend the single held-out test access on the full
   composed pipeline.
3. Retry the retriever fine-tune on cooler or rented hardware — the one genuinely untested
   lever.
4. Record H1 ("+10–25 points from better descriptions") as **refuted by headroom**, not
   untested.
5. Open question: `standard_L4` is hostile to the cross-encoder in *every* configuration
   (−0.180 R@1, p=0.00003) despite being the easiest split for dense retrieval.

**Takeaway:** retrieval went **0.139 → 0.259** from better embeddings and richer document
text; **nothing since has added a defensible point** — but we now know exactly where the
remaining errors live.

## 7. Comparison with JEV

> **TODO — not yet written.** Pending a grounded description of JEV. To make the comparison
> rigorous rather than a feature table, the four facts needed are:
>
> 1. **Which stage** it addresses — narrowing thousands of candidates, choosing among a few,
>    or both?
> 2. **Joint or independent scoring?** The pivotal property. Independent scoring precomputes
>    all 3,551 tool vectors offline (3 ms/query here); joint scoring must run per
>    candidate per query (5 s/query here) and that alone decided our outcome.
> 3. **What supervision does it need?** We had labels for only 160 of 3,551 tools, and the
>    real test was transfer to tools never seen in training.
> 4. **Can it abstain natively?** Similarity thresholds provably cannot here (see §5).
>
> Axes to compare on, with the bar from this work: error location (83% retrieval / 7%
> selection — anything improving only selection caps at ~7%); scale cost (3,551 candidates
> per query); unseen tools (catalogs change; 39/199 held out); near-duplicates (10.1% are
> genuinely inseparable, so the right answer is a *set*); evidence standard (at n=150 point
> estimates are noise).
