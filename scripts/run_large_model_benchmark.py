#!/usr/bin/env python3
"""
End-to-end benchmark for 1B–2B-class models: verify → calibrate → experiment.

Usage:
  python scripts/run_large_model_benchmark.py --config configs/qwen2_5_1_5b.yaml
  python scripts/run_large_model_benchmark.py --config configs/smollm2_1_7b.yaml --repeats 1
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIGS = [
    ROOT / "configs" / "qwen2_5_1_5b.yaml",
    ROOT / "configs" / "smollm2_1_7b.yaml",
]


def run_step(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None, help="Single config; default runs both 1B-class configs")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--skip-experiment", action="store_true")
    args = parser.parse_args()

    configs = [args.config] if args.config else CONFIGS
    py = sys.executable

    for cfg in configs:
        print(f"\n========== {cfg.name} ==========")
        if not args.skip_verify:
            run_step([py, "scripts/verify_one_shape.py", "--config", str(cfg)])

        exp_cmd = [py, "scripts/run_experiment.py", "--config", str(cfg), "--generate-traces"]
        if args.repeats is not None:
            exp_cmd.extend(["--repeats", str(args.repeats)])
        if args.skip_experiment:
            exp_cmd.append("--calibrate-only")
        else:
            pass  # full experiment after calibrate inside run_experiment

        if args.skip_experiment:
            run_step(exp_cmd)
        else:
            # calibrate first (fast fail if OOM)
            run_step(exp_cmd + ["--calibrate-only"])
            run_step(exp_cmd)

        results_sub = None
        import yaml

        with cfg.open() as f:
            results_sub = yaml.safe_load(f).get("results_subdir", "")
        results_dir = ROOT / "results" / results_sub if results_sub else ROOT / "results"
        run_step([py, "scripts/analyze_results.py", "--results-dir", str(results_dir)])


if __name__ == "__main__":
    main()
