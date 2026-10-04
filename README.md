# Tool Selection from User Intent

A research experiment measuring how an agent picks the right tool/API from natural-language
intent over a **large, open catalog** — and, more pointedly, *which stage of that pipeline is
actually broken*.

```
user intent → RETRIEVER (3,551 tools → ~25) → SELECTOR (25 → 1) → tool call
```

**Headline result: ~83% of end-to-end errors are retrieval failures, ~7% are selection
failures.** The selector is ~0.93 accurate when handed a shortlist containing the right
tool; the retriever's R@1 started at ~0.15. Most published work tunes the selector, because
most benchmarks present 1–10 candidates and so make retrieval trivially easy.

Retrieval R@1 went **0.139 → 0.259** from a better encoder plus richer document text.
Nothing tried after that produced a statistically defensible gain — see
[`FINDINGS.md`](FINDINGS.md) for what failed and why, which is most of the value here.

## Why the catalog is built the way it is

No public benchmark tests this problem: they offer a handful of candidates per query. So the
catalog fuses two sources:

| | count | role |
|---|---:|---|
| MetaTool plugins | 199 | carry all 20,532 labeled `(query → tool)` pairs |
| APIs.guru operations | 3,352 | carry **no** labels — they exist as distractors |

Selection happens over all **3,551 at a 17.8:1 distractor ratio.** One number shows why that
matters: recall@100 falls from **0.973** at a 199-tool pool to **0.861** at 3,551. At 199,
retrieval is solved and unmeasurable. At 3,551, ~14% of correct tools are unreachable no
matter how good the selector is.

## Repository layout

| path | contents |
|---|---|
| `EXPERIMENT_DESIGN.md` | the plan: hypotheses H1–H6, arms, splits, metrics, pre-registered gating criteria |
| `DATA.md` | **what actually happened** — every run, with real numbers. Where the two disagree, this one is authoritative |
| `FINDINGS.md` | publication summary: what helped, what didn't, and the transferable lessons |
| `scripts/` | the pipeline, run by hand in order (see below) |
| `data/corpus/` | unified catalog + splits |
| `data/generated/` | 4,585 LLM-generated intents (rebuilding needs paid API calls) |
| `data/eval/` | sweep results, metrics, per-item ranks |

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install requests datasets pyyaml truststore sentence-transformers accelerate
```

Always invoke `./.venv/bin/python` and run from the repo root — every script uses paths
relative to cwd.

> If your network does TLS interception, `certifi` fails with `CERTIFICATE_VERIFY_FAILED`.
> Every network-touching script calls `truststore.inject_into_ssl()` to use the OS trust
> store instead. Keep that in any new script.

## Reproducing

Embedding matrices (`data/eval/*.npy`, ~108 MB) and the raw third-party downloads are **not
in the repo**. Regenerate them:

```bash
# catalog + splits (fast, free)
./.venv/bin/python scripts/fetch_apisguru.py --n-apis 400 --per-provider 2
./.venv/bin/python scripts/build_corpus.py

# the winning retrieval config (local, free, ~2 min on Apple Silicon)
./.venv/bin/python scripts/build_embeddings.py --model thenlper/gte-large --text v3 \
    --device mps --out data/eval/tool_embeddings_gtelarge_v3.npy
./.venv/bin/python scripts/eval_retrieval.py --partition dev \
    --matrix data/eval/tool_embeddings_gtelarge_v3.npy
```

Stage A (intent generation) costs real money; stages B and C are local and free. The full
ordered pipeline, and the gotchas that will bite you, are documented in `CLAUDE.md`.

### Two things to know before changing anything

- **`data/eval/tool_embeddings.npy` is row-aligned with `data/corpus/catalog.jsonl`** — row
  *i* is line *i*. Regenerate the catalog without regenerating embeddings and consumers read
  garbage **silently, with no error**.
- **Write new matrices to new filenames.** `tool_embeddings.npy` is the shipped v0 artifact
  that the confusion split was defined against; overwriting it silently redefines the split.

## Known limitations

Stated rather than papered over — the longer list is in `DATA.md`:

- **Circularity is unmitigated.** Claude Opus 5 generated the primary eval set and is the
  model evaluated on it. Partial defenses: MetaTool as a GPT-4-generated anchor, and the
  generator's self-reported `separable` flag. A real fix needs a second model family.
- **Difficulty levels are not ordered.** "Indirect" (L3) is harder than "distractor context"
  (L4). Treat L2/L3/L4 as three conditions, not a scale.
- **The eval set under-samples poorly-documented tools** at 0.150× their catalog rate,
  because the generator had to read a description to write an intent for it.
- **No significance tests in the encoder sweep.** Later sections have paired McNemar +
  bootstrap tests; two of the four claims tested changed status as a result.

## Data provenance and attribution

- **APIs.guru** — OpenAPI specs, openly licensed. <https://apis.guru/>
- **MetaTool** — MIT. Labeled `(query → tool)` pairs. <https://github.com/HowieHwong/MetaTool>
- **Generated intents** — produced for this experiment with Claude Opus 5; released under
  this repository's license.

`data/corpus/` and `data/catalog/` contain data **derived** from the sources above. Credential-shaped
strings that arrived inside third-party OpenAPI specs (and fictional ones invented by the
generator) have been redacted — see `scripts/redact_secrets.py` and
`data/eval/redaction_report.json`.

## License

Code and generated data: see [`LICENSE`](LICENSE). Third-party data retains its original
license as listed above.
