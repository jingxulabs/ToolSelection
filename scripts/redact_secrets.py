"""Redact credential-shaped strings from the publishable data -> in-place, idempotent.

Run before publishing. Two distinct classes get redacted for two different reasons:

  1. REAL third-party identifiers that arrived inside APIs.guru OpenAPI specs. The
     notable one is an AWS access key ID in SYNQ.fm's upload example. An access key ID
     is the non-secret half of a keypair and is already public in that spec, so this is
     not a breach -- but it belongs to someone else, and GitHub secret scanning flags it
     on push (alerts, possibly push protection).

  2. FICTIONAL tokens that Claude invented inside generated intents ("my admin token is
     xoxp-8821-adminaudit"). Harmless in substance, but scanners match on PREFIX SHAPE,
     so fakes generate the same alert noise as the real thing.

Scientifically free: for the generated intents the label is the TOOL, never the token
string, so the redacted text supports the identical task. For the catalog it rewrites a
few characters inside one tool's description.

KNOWN CONSEQUENCE, recorded rather than hidden: data/eval/*.npy were built from the
PRE-redaction catalog, so one row's document text no longer matches the matrix that
encoded it, and the `catalog_sha256` in the .meta.json sidecars will not match either.
The .npy files are gitignored, so anyone cloning regenerates them from the redacted
catalog and gets a self-consistent artifact; only exact reproduction of our published
numbers is affected, by one row out of 3,551. Do NOT regenerate
data/eval/tool_embeddings.npy to "fix" this -- split D's confusion pairs are defined
against that exact matrix (see CLAUDE.md).

  ./.venv/bin/python scripts/redact_secrets.py --dry-run
  ./.venv/bin/python scripts/redact_secrets.py
"""

import argparse, json, re
from pathlib import Path

# Replacements deliberately BREAK the scanner-matchable prefix (no surviving "xoxp-" or
# "AKIA" literal), otherwise the alert fires on the placeholder too.
PATTERNS = [
    ("aws_access_key_id", re.compile(r"AKIA[0-9A-Z]{16}"), "<REDACTED_AWS_KEY_ID>"),
    ("slack_token", re.compile(r"xox[baprs]-[A-Za-z0-9-]{4,}"), "<REDACTED_SLACK_TOKEN>"),
    ("stripe_style_key", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{10,}"),
     "<REDACTED_API_KEY>"),
    ("google_api_key", re.compile(r"AIza[0-9A-Za-z_-]{30,}"), "<REDACTED_GOOGLE_KEY>"),
    ("bearer_literal", re.compile(r"([Bb]earer\s+)[A-Za-z0-9._~+/=-]{20,}"),
     r"\1<REDACTED>"),
    ("jwt_or_b64_blob", re.compile(r"eyJ[A-Za-z0-9_/+=-]{24,}"), "<REDACTED_JWT>"),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
     "<REDACTED_PRIVATE_KEY>"),
    # Home-directory paths in generated intents. Not credentials, but the directory chain
    # can carry an invented personal name next to sensitive-sounding context -- e.g. a
    # synthetic "veteran client" request naming a surname and a medical-evidence PDF. The
    # content is entirely LLM-fabricated and refers to nobody, but it READS like PHI, and a
    # reviewer skimming a public repo cannot tell that at a glance. Collapse to the
    # basename: the filename is what the task is about (validate a PDF before upload), the
    # path to it is not, so the label and difficulty are untouched.
    ("home_dir_path", re.compile(r"(?:/Users/|/home/|[A-Za-z]:\\\\Users\\\\)"
                                r"[A-Za-z0-9._\\/-]*?([A-Za-z0-9._-]+\.[A-Za-z0-9]{1,5})\b"),
     r"./\1"),
]

# Rules that must NOT touch the catalogs. The catalog's text is the retriever's input, so
# rewriting it changes the experiment; and the only home-dir paths in there are generic
# documentation placeholders (`/home/foo/foobot.aiml` in the foobot AIML spec) with no
# privacy value. Credential rules still apply everywhere.
GENERATED_ONLY = {"home_dir_path"}
CATALOGS = {"data/corpus/catalog.jsonl", "data/catalog/catalog.jsonl"}

TARGETS = [
    "data/corpus/catalog.jsonl",
    "data/catalog/catalog.jsonl",
    "data/generated/standard_L2.jsonl",
    "data/generated/standard_L3.jsonl",
    "data/generated/standard_L4.jsonl",
    "data/generated/confusion_L2.jsonl",
    "data/generated/abstain_L2.jsonl",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--report", default="data/eval/redaction_report.json")
    a = ap.parse_args()

    report, grand = {}, 0
    for rel in a.targets:
        p = Path(rel)
        if not p.exists():
            print(f"  skip (missing): {rel}")
            continue
        orig = p.read_text()
        text, counts = orig, {}
        for name, pat, repl in PATTERNS:
            if name in GENERATED_ONLY and rel in CATALOGS:
                continue
            found = pat.findall(text)
            if not found:
                continue
            counts[name] = len(found)
            text = pat.sub(repl, text)
        if not counts:
            print(f"  clean: {rel}")
            continue

        # every line must still parse as JSON -- a redaction that corrupts the corpus is
        # worse than the thing it redacts
        for i, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    raise SystemExit(f"ABORT: {rel} line {i} is not valid JSON after "
                                     f"redaction ({e}). No files written.")
        n = sum(counts.values())
        grand += n
        report[rel] = counts
        print(f"  {'WOULD REDACT' if a.dry_run else 'REDACTED'} {n:3d} in {rel}: {counts}")
        if not a.dry_run:
            p.write_text(text)

    print(f"\n{'would redact' if a.dry_run else 'redacted'} {grand} occurrence(s) total")
    if not a.dry_run and report:
        # MERGE, never overwrite. The script is idempotent, so a second run finds only
        # whatever rule was added since the first -- overwriting would silently erase the
        # record of everything already redacted, which is the one thing this file is for.
        rp = Path(a.report)
        prev = json.loads(rp.read_text()) if rp.exists() else {}
        merged = prev.get("by_file", {})
        for f, counts in report.items():
            dst = merged.setdefault(f, {})
            for k, v in counts.items():
                dst[k] = dst.get(k, 0) + v
        rp.write_text(json.dumps(
            {"note": "credential-shaped strings and home-directory paths removed before "
                     "publication; see scripts/redact_secrets.py for rationale and the "
                     "embedding-staleness consequence. Cumulative across runs.",
             "by_file": merged,
             "total": sum(v for c in merged.values() for v in c.values())}, indent=2))
        print(f"wrote {rp} (cumulative)")


if __name__ == "__main__":
    main()
