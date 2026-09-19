#!/usr/bin/env python3
"""Peak VRAM / power draw over a sampler CSV, optionally windowed by UTC timestamps."""
import csv, sys, collections
path = sys.argv[1]
lo = sys.argv[2] if len(sys.argv) > 2 else ""
hi = sys.argv[3] if len(sys.argv) > 3 else "9"
peak = collections.defaultdict(int); draw = collections.defaultdict(float); limit = {}
n = 0
with open(path) as h:
    for row in csv.DictReader(h):
        ts = row["timestamp_utc"]
        if lo and not (lo <= ts <= hi):
            continue
        n += 1
        key = (row["kind"], row["key"])
        peak[key] = max(peak[key], int(row["used_mib"] or 0))
        if row["kind"] == "gpu":
            if row.get("power_draw_w"): draw[key] = max(draw[key], float(row["power_draw_w"]))
            if row.get("power_limit_w"): limit[key] = row["power_limit_w"]
print(f"# samples in window: {n}  ({lo or 'start'} .. {hi if hi!='9' else 'end'})")
for key in sorted(peak):
    extra = f"  peak draw {draw[key]:.1f} W of {limit.get(key,'?')} W" if key in draw else ""
    print(f"  {key[0]} {key[1]}: {peak[key]} MiB ({peak[key]/1024:.2f} GiB){extra}")
