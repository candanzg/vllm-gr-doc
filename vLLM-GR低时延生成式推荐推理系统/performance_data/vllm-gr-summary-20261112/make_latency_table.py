"""End-to-end latency percentiles per test case from the harness's raw batch timings.

Input:  offline_measurements.csv from each run (one row per timed batch;
        48 rows per point = 3 rounds x 16 batches, warmup excluded).
Metric: batch_latency_ms = wall time of one generate() call for B requests,
        from submit to fully materialized tokens/scores (infer_timed in measure.py).
Output: data/gr-latency-168.csv and report-latency-20261112.md.

Percentiles use linear interpolation between order statistics (numpy's default,
statistics.quantiles(method="inclusive")). With 48 samples, p99 lies between the
largest and second-largest sample.
"""

import csv
import os
import statistics
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
# Each mode was measured in two runs: L<=4096 (first run) and L=8192/10240 (rerun in
# fresh engines after the first run's engine died on the 4096 -> 8192 switch).
# Raw batch timings of both runs are read from data/raw/ and concatenated.
RAW = os.path.join(DATA, "raw")
SOURCES = [("U", ["U-lo.csv", "U-hi.csv"]), ("C", ["C-lo.csv", "C-hi.csv"])]
STATS = ("mean", "p50", "p90", "p99")


def percentile(sorted_xs, q):
    """Linear interpolation; q in [0, 100]."""
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = (len(sorted_xs) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def summarize(xs):
    s = sorted(xs)
    return {
        "n": len(s),
        "mean": statistics.fmean(s),
        "p50": percentile(s, 50),
        "p90": percentile(s, 90),
        "p99": percentile(s, 99),
    }


samples = defaultdict(list)
for mode, names in SOURCES:
    for name in names:
        path = os.path.join(RAW, name)
        if not os.path.exists(path):
            print("skip (missing):", name)
            continue
        with open(path, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["mode"] != mode:
                    raise SystemExit(f"{name}: unexpected mode {r['mode']}")
                key = (mode, int(r["length"]), int(r["beam"]), int(r["batch"]))
                samples[key].append(float(r["batch_latency_ms"]))

if not samples:
    raise SystemExit("no measurements found")

rows = []
for key in sorted(samples):
    mode, length, beam, batch = key
    st = summarize(samples[key])
    rows.append(
        {
            "config": f"bw{beam}_in{length}_b{batch}_{mode}",
            "mode": mode,
            "length": length,
            "beam": beam,
            "batch": batch,
            "n": st["n"],
            **{k: round(st[k], 2) for k in STATS},
        }
    )

counts = sorted({r["n"] for r in rows})
if counts != [48]:
    print("WARNING: sample counts per point are", counts, "(expected 48)", file=sys.stderr)


def write_locked_safe(path, text):
    # Excel holds a deny-write handle on Windows; fall back to <name>.new.
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    try:
        os.replace(tmp, path)
        print("wrote", path)
    except PermissionError:
        os.replace(tmp, path + ".new")
        print(f"LOCKED {path} -> wrote {path}.new")


header = ["config", "mode", "length", "beam", "batch", "n", *STATS]
lines = [",".join(header)]
lines += [",".join(str(r[h]) for h in header) for r in rows]
write_locked_safe(os.path.join(DATA, "gr-latency-168.csv"), "\n".join(lines) + "\n")

# Markdown: one table per (mode, batch), rows = beam x length.
by = {(r["mode"], r["length"], r["beam"], r["batch"]): r for r in rows}
modes = [m for m, _ in SOURCES if any(k[0] == m for k in by)]
batches = sorted({k[3] for k in by})
lengths = sorted({k[1] for k in by})
beams = sorted({k[2] for k in by})
LABEL = {"U": "U（不启用受限表）", "C": "C（启用受限表）"}

md = [
    "# vllm-gr 端到端时延（H20-3e，main `bf9db2a`）",
    "",
    "端到端时延 = 一次 `generate()` 调用处理 B 个请求的墙钟时间，从提交到 token 与分数全部取回；"
    "预热批次不计入。每个用例 48 个样本（3 轮 × 16 批）。单位 ms。",
    "",
]
for mode in modes:
    for b in batches:
        md += [f"## {LABEL[mode]}，batch = {b}", ""]
        md += ["| 配置 | mean | p50 | p90 | p99 |", "|---|---:|---:|---:|---:|"]
        for w in beams:
            for length in lengths:
                r = by.get((mode, length, w, b))
                if r is None:
                    md.append(f"| bw{w}_in{length}_b{b}_{mode} | — | — | — | — |")
                    continue
                md.append(
                    f"| {r['config']} | {r['mean']:.2f} | {r['p50']:.2f} | {r['p90']:.2f} | {r['p99']:.2f} |"
                )
        md.append("")
write_locked_safe(os.path.join(HERE, "report-latency-20261112.md"), "\n".join(md))
print("points:", len(rows), "modes:", modes, "batches:", batches)
