#!/usr/bin/env python3
"""Print a summary table from Google Benchmark JSON output (read from stdin)."""

from __future__ import annotations

import json
import sys


def format_time(ns: float) -> str:
    if ns >= 1_000_000_000:
        return f"{ns / 1_000_000_000:.2f} s"
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.2f} us"
    return f"{ns:.2f} ns"


def main() -> None:
    data = json.load(sys.stdin)
    benchmarks = data.get("benchmarks", [])

    # Keep only _mean entries (skip _median duplicates when aggregates_only is set)
    entries = [b for b in benchmarks if b["name"].endswith("_mean")]

    if not entries:
        print("(no benchmark results)")
        return

    # Column widths
    max_name = max(len(b["name"].removesuffix("_mean")) for b in entries)
    name_w = max(max_name, 4)

    print(f"{'name':<{name_w}}  {'iters':>7}  {'real_time':>10}  {'cpu_time':>10}")
    print(f"{'-' * name_w}  {'-' * 7}  {'-' * 10}  {'-' * 10}")

    for b in entries:
        name = b["name"].removesuffix("_mean")
        iters = b.get("iterations", 0)
        real = format_time(b["real_time"])
        cpu = format_time(b["cpu_time"])
        print(f"{name:<{name_w}}  {iters:>7}  {real:>10}  {cpu:>10}")


if __name__ == "__main__":
    main()
