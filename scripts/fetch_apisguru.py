"""Build a normalized tool catalog from the APIs.guru OpenAPI directory.

Each OpenAPI *operation* becomes one tool record (catalog.jsonl), matching the
schema in EXPERIMENT_DESIGN.md §4.1. Specs are sampled across providers and
categories so the catalog has domain spread rather than 300 Adyen endpoints.
"""

import argparse, json, random, re, sys, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

try:  # corporate TLS interception: use the OS trust store, not certifi
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

LIST_URL = "https://api.apis.guru/v2/list.json"
OUT = Path("data/catalog")
WRITE_VERBS = {"post", "put", "patch", "delete"}
DESTRUCTIVE_VERBS = {"delete"}


def get_json(url, timeout=30):
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "tool-selection-research"})
    r.raise_for_status()
    return r.json()


def pick_specs(listing, n_apis, per_provider, seed):
    """Sample API specs with a per-provider cap so no vendor dominates."""
    rng = random.Random(seed)
    by_provider = defaultdict(list)
    for api_id, entry in listing.items():
        ver = entry.get("versions", {}).get(entry.get("preferred"))
        if not ver or not ver.get("swaggerUrl"):
            continue
        info = ver.get("info", {})
        by_provider[info.get("x-providerName", api_id.split(":")[0])].append(
            {
                "api_id": api_id,
                "url": ver["swaggerUrl"],
                "provider": info.get("x-providerName", ""),
                "title": info.get("title", ""),
                "categories": info.get("x-apisguru-categories", []),
            }
        )

    pool = []
    for provider, specs in by_provider.items():
        rng.shuffle(specs)
        pool.extend(specs[:per_provider])
    rng.shuffle(pool)
    return pool[:n_apis]


def resolve(node, root, depth=0):
    """Shallow $ref resolution. Depth-capped; OpenAPI schemas are often cyclic."""
    if depth > 6 or not isinstance(node, dict):
        return node if not isinstance(node, list) else [resolve(x, root, depth + 1) for x in node]
    if "$ref" in node and isinstance(node["$ref"], str) and node["$ref"].startswith("#/"):
        target = root
        for part in node["$ref"][2:].split("/"):
            target = target.get(part.replace("~1", "/").replace("~0", "~"), {}) if isinstance(target, dict) else {}
        return resolve(target, root, depth + 1)
    return {k: resolve(v, root, depth + 1) for k, v in node.items()}


def params_to_schema(op, path_item, root):
    """Flatten OpenAPI params + requestBody into a single JSON Schema object."""
    props, required = {}, []
    for p in (path_item.get("parameters", []) or []) + (op.get("parameters", []) or []):
        p = resolve(p, root)
        name = p.get("name")
        if not name:
            continue
        sch = resolve(p.get("schema", {}), root) or {"type": p.get("type", "string")}
        props[name] = {
            "type": sch.get("type", "string"),
            "description": (p.get("description") or "")[:300],
            "in": p.get("in", "query"),
        }
        if p.get("required"):
            required.append(name)

    body = resolve(op.get("requestBody", {}), root)
    content = body.get("content", {}) if isinstance(body, dict) else {}
    js = next((v for k, v in content.items() if "json" in k.lower()), {})
    bsch = resolve(js.get("schema", {}), root)
    if isinstance(bsch, dict) and bsch.get("properties"):
        for name, sub in list(bsch["properties"].items())[:40]:
            sub = sub if isinstance(sub, dict) else {}
            props[name] = {
                "type": sub.get("type", "string"),
                "description": (sub.get("description") or "")[:300],
                "in": "body",
            }
        required.extend([r for r in (bsch.get("required") or []) if isinstance(r, str)])

    return {"type": "object", "properties": props, "required": sorted(set(required))}


def slug(*parts):
    s = "_".join(p for p in parts if p)
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s).strip("_").lower()
    return re.sub(r"_+", "_", s)[:120]


def extract(spec, meta, max_ops):
    """Turn one OpenAPI spec into tool records."""
    out = []
    base = (spec.get("info") or {}).get("title") or meta["title"]
    for path, item in list((spec.get("paths") or {}).items()):
        if not isinstance(item, dict):
            continue
        for verb, op in item.items():
            if verb.lower() not in {"get", "post", "put", "patch", "delete"} or not isinstance(op, dict):
                continue
            desc = (op.get("description") or op.get("summary") or "").strip()
            summary = (op.get("summary") or "").strip()
            name = op.get("operationId") or slug(verb, path)
            schema = params_to_schema(op, item, spec)
            side = (
                "destructive" if verb.lower() in DESTRUCTIVE_VERBS
                else "write" if verb.lower() in WRITE_VERBS
                else "read"
            )
            out.append(
                {
                    "tool_id": slug(meta["api_id"], name),
                    "name": name,
                    "description": desc[:1200],
                    "summary": summary[:300],
                    "parameters": schema,
                    "returns": sorted((op.get("responses") or {}).keys())[:6],
                    "auth_scope": sorted({k for s in (op.get("security") or []) for k in s})[:5],
                    "side_effects": side,
                    "domain": meta["categories"][0] if meta["categories"] else "uncategorized",
                    "categories": meta["categories"],
                    "provider": meta["provider"],
                    "api_title": base,
                    "http": {"method": verb.upper(), "path": path},
                    "tags": (op.get("tags") or [])[:5],
                    "deprecated": bool(op.get("deprecated")),
                    "source": "apis.guru",
                    "spec_url": meta["url"],
                }
            )
            if len(out) >= max_ops:
                return out
    return out


def doc_quality(t):
    """Score 0-4. Expected to correlate with per-tool accuracy (design H1)."""
    d, p = t["description"], t["parameters"]["properties"]
    s = 0
    s += 1 if len(d) >= 20 else 0
    s += 1 if len(d) >= 120 else 0
    s += 1 if p else 0
    s += 1 if p and sum(1 for v in p.values() if len(v.get("description", "")) >= 10) / max(len(p), 1) >= 0.5 else 0
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-apis", type=int, default=400, help="specs to download")
    ap.add_argument("--per-provider", type=int, default=2, help="cap specs per vendor")
    ap.add_argument("--max-ops-per-api", type=int, default=12)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print("fetching directory listing ...", file=sys.stderr)
    listing = get_json(LIST_URL, timeout=120)
    print(f"  {len(listing)} APIs in directory", file=sys.stderr)

    specs = pick_specs(listing, a.n_apis, a.per_provider, a.seed)
    print(f"  sampled {len(specs)} specs across {len({s['provider'] for s in specs})} providers", file=sys.stderr)

    tools, failed = [], 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(get_json, s["url"], 40): s for s in specs}
        for i, f in enumerate(as_completed(futs), 1):
            meta = futs[f]
            try:
                tools.extend(extract(f.result(), meta, a.max_ops_per_api))
            except Exception:
                failed += 1
            if i % 50 == 0:
                print(f"  {i}/{len(specs)} specs, {len(tools)} tools, {failed} failed", file=sys.stderr)

    seen, uniq = set(), []
    for t in tools:
        if t["tool_id"] in seen:
            continue
        seen.add(t["tool_id"])
        t["doc_quality"] = doc_quality(t)
        uniq.append(t)

    path = OUT / "catalog.jsonl"
    with path.open("w") as fh:
        for t in uniq:
            fh.write(json.dumps(t) + "\n")

    by_dom = defaultdict(int)
    by_q = defaultdict(int)
    by_side = defaultdict(int)
    for t in uniq:
        by_dom[t["domain"]] += 1
        by_q[t["doc_quality"]] += 1
        by_side[t["side_effects"]] += 1

    stats = {
        "tools": len(uniq),
        "specs_downloaded": len(specs) - failed,
        "specs_failed": failed,
        "providers": len({t["provider"] for t in uniq}),
        "domains": dict(sorted(by_dom.items(), key=lambda x: -x[1])),
        "doc_quality_hist": dict(sorted(by_q.items())),
        "side_effects": dict(by_side),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": a.seed,
    }
    (OUT / "catalog_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
