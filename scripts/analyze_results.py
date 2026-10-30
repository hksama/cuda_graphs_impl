#!/usr/bin/env python3
"""Aggregate JSONL experiment outputs."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acg.analyze import aggregate_repeats, print_comparison_table

POLICIES = [
    "always_eager",
    "lru_on_first_use",
    "calibration_fixed",
    "adaptive_admission",
]
TRACES = ["recurring.json", "many_rare.json", "shift.json"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory containing *_policy_runN.jsonl (e.g. results/qwen2_5_1_5b)",
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    all_aggs: dict = {}
    for trace in TRACES:
        stem = Path(trace).stem
        for policy in POLICIES:
            paths = [
                args.results_dir / f"{stem}_{policy}_run{i}.jsonl"
                for i in range(args.repeats)
            ]
            if paths[0].exists():
                all_aggs[f"{stem}/{policy}"] = aggregate_repeats(paths)

    print_comparison_table(all_aggs)


if __name__ == "__main__":
    main()
