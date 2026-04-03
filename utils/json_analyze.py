#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick analyzer for DARPA TC E5 (CDM20) JSONL logs.

E5 JSON exports are commonly line-delimited JSON (JSONL), where each line is:
{
  "datum": {"com.bbn.tc.schema.avro.cdm20.Subject": {...}},
  "CDMVersion": "20",
  "source": "..."
}

This script:
- counts record types (Host/Subject/Event/...)
- lists key paths per record type (configurable depth)
- finds timestamp-like keys (contains "timestamp"/"time" or endswith "Nanos")
- optionally writes a JSON report

Usage:
  python e3_jsonl_quick_analyze.py --input 1.json
  python e3_jsonl_quick_analyze.py --input ta1-theia-...json --max-lines 200000 --report report.json
"""
import argparse
import json
import os
import re
from collections import Counter, defaultdict

TS_RE = re.compile(r"(timestamp|time)", re.IGNORECASE)

def is_ts_key(k: str) -> bool:
    return bool(TS_RE.search(k)) or k.lower().endswith("nanos")

def flatten_keys(obj, prefix="", depth=2):
    """Collect dotted key paths up to a certain depth."""
    out = set()
    if depth <= 0:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            out.add(key)
            out |= flatten_keys(v, key, depth=depth-1)
    elif isinstance(obj, list):
        out.add(prefix + "[]")
        if obj:
            out |= flatten_keys(obj[0], prefix + "[]", depth=depth-1)
    return out

def short_type_name(full: str) -> str:
    return full.split(".")[-1] if "." in full else full

def truncate(obj, max_str=120, max_items=30):
    if isinstance(obj, str):
        return obj if len(obj) <= max_str else obj[: max_str - 3] + "..."
    if isinstance(obj, dict):
        out = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                out["..."] = f"(+{len(obj)-max_items} more keys)"
                break
            out[k] = truncate(v, max_str=max_str, max_items=max_items)
        return out
    if isinstance(obj, list):
        return [truncate(x, max_str=max_str, max_items=max_items) for x in obj[:5]] + ([] if len(obj)<=5 else ["..."])
    return obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to a JSONL file (E3/CDM export).")
    ap.add_argument("--max-lines", type=int, default=200000, help="Max lines to scan.")
    ap.add_argument("--depth", type=int, default=2, help="Key-path depth to collect per record.")
    ap.add_argument("--samples", type=int, default=1, help="Samples to keep per record type.")
    ap.add_argument("--report", default="", help="Optional JSON report output path.")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input not found: {args.input}")

    total = parsed = bad = 0
    type_counter = Counter()
    type_keys = defaultdict(set)
    type_ts_keys = defaultdict(set)
    type_samples = defaultdict(list)
    version_counter = Counter()
    source_counter = Counter()

    with open(args.input, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            total += 1
            if total > args.max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                parsed += 1
            except Exception:
                bad += 1
                continue

            if isinstance(rec, dict):
                version_counter[str(rec.get("CDMVersion", ""))] += 1
                source_counter[str(rec.get("source", ""))] += 1

            datum = rec.get("datum", {}) if isinstance(rec, dict) else {}
            if not isinstance(datum, dict) or not datum:
                type_counter["<no_datum>"] += 1
                continue

            full_type = next(iter(datum.keys()))
            payload = datum.get(full_type, {})
            t = short_type_name(full_type)
            type_counter[t] += 1

            if isinstance(payload, dict):
                ks = flatten_keys(payload, depth=args.depth)
                type_keys[t] |= ks
                # top-level ts keys
                for k in payload.keys():
                    if is_ts_key(k):
                        type_ts_keys[t].add(k)
                # nested ts-ish keys (from flattened paths)
                for k in ks:
                    if is_ts_key(k.split(".")[-1]):
                        type_ts_keys[t].add(k)

            if len(type_samples[t]) < args.samples:
                type_samples[t].append(payload)

    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Lines scanned: {total} (parsed={parsed}, bad_json={bad})")
    print(f"CDMVersion counts: {dict(version_counter)}")
    print(f"Top sources (up to 5): {[s for s,_ in source_counter.most_common(5)]}")
    print("-" * 80)
    print("Record types (count):")
    for t, c in type_counter.most_common():
        print(f"  {t:25s} {c}")
    print("-" * 80)

    for t, _ in type_counter.most_common():
        print(f"[{t}] key-paths collected: {len(type_keys[t])} (depth<={args.depth})")
        ts = sorted(type_ts_keys[t])
        print(f"  timestamp-like keys: {ts if ts else '(none found at scanned depth)'}")
        if type_samples[t]:
            print("  sample[0] (truncated):")
            print(json.dumps(truncate(type_samples[t][0]), ensure_ascii=False, indent=2)[:1500])
        print("-" * 80)

    if args.report:
        report = {
            "input": args.input,
            "lines_scanned": total,
            "parsed": parsed,
            "bad_json": bad,
            "cdm_versions": dict(version_counter),
            "sources_top": source_counter.most_common(20),
            "types": {
                t: {
                    "count": int(type_counter[t]),
                    "keys": sorted(type_keys[t]),
                    "timestamp_like_keys": sorted(type_ts_keys[t]),
                    "samples": type_samples[t],
                }
                for t in type_counter.keys()
            },
        }
        with open(args.report, "w", encoding="utf-8") as wf:
            json.dump(report, wf, ensure_ascii=False, indent=2)
        print(f"Wrote report: {args.report}")

if __name__ == "__main__":
    main()
