#!/usr/bin/env python3
"""Generate traces, calibrate device, run policies, write JSONL results."""

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from acg.device import require_cuda
from acg.analyze import aggregate_repeats, print_comparison_table
from acg.calibration import calibrate_device, load_calibration, save_calibration
from acg.env import collect_env, write_json
from acg.model import EncoderModel
from acg.policies import GraphCache, make_policy
from acg.runner import run_trace, write_jsonl
from acg.traces import generate_all_traces, load_trace

POLICIES = [
    "always_eager",
    "lru_on_first_use",
    "calibration_fixed",
    "adaptive_admission",
]


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_paths(config: dict, args: argparse.Namespace) -> tuple[Path, Path, Path]:
    calibration = args.calibration
    default_cal = ROOT / "calibration" / "rtx5060.json"
    if calibration == default_cal and config.get("calibration_basename"):
        calibration = ROOT / "calibration" / config["calibration_basename"]

    traces_dir = args.traces_dir
    if config.get("traces_subdir"):
        traces_dir = traces_dir / config["traces_subdir"]

    results_dir = args.results_dir
    if config.get("results_subdir"):
        results_dir = results_dir / config["results_subdir"]

    return calibration, traces_dir, results_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--traces-dir", type=Path, default=ROOT / "traces")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--calibration", type=Path, default=ROOT / "calibration" / "rtx5060.json")
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--generate-traces", action="store_true")
    parser.add_argument("--repeats", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    calibration_path, traces_dir, results_dir = resolve_paths(config, args)
    device = require_cuda()
    encoder = EncoderModel.from_config(config, device)

    if args.generate_traces or not (traces_dir / "recurring.json").exists():
        generate_all_traces(
            traces_dir,
            config["batch_sizes"],
            config["seq_lengths"],
            config["trace_length"],
            config["trace_seeds"],
        )
        print(f"Wrote traces to {traces_dir}")

    if args.calibrate_only or not calibration_path.exists():
        print(f"Calibrating device -> {calibration_path}")
        cal = calibrate_device(
            encoder,
            config["batch_sizes"],
            config["seq_lengths"],
            warmup_iters=config["calibration_warmup_iters"],
            timing_iters=config["calibration_timing_iters"],
        )
        save_calibration(calibration_path, cal)
        if args.calibrate_only:
            return

    calibration = load_calibration(calibration_path)
    repeats = args.repeats if args.repeats is not None else config["experiment_repeats"]
    trace_files = ["recurring.json", "many_rare.json", "shift.json"]

    for trace_name in trace_files:
        trace_path = traces_dir / trace_name
        _, requests = load_trace(trace_path)
        for policy_name in POLICIES:
            repeat_paths = []
            for run_i in range(repeats):
                policy = make_policy(policy_name, config)
                cache = GraphCache(
                    max_size=config["max_graph_cache"],
                    mem_budget_mb=config["mem_budget_mb"],
                )
                records, summary = run_trace(
                    encoder,
                    policy,
                    requests,
                    cache,
                    calibration,
                    config,
                    correctness_sample_every=config["correctness_sample_every"],
                )
                out_path = results_dir / f"{trace_path.stem}_{policy_name}_run{run_i}.jsonl"
                env = collect_env(args.config, trace_path)
                meta = {
                    "env": env,
                    "summary": summary,
                    "trace": trace_name,
                    "run": run_i,
                }
                write_jsonl(out_path, records, meta)
                repeat_paths.append(out_path)
                print(
                    f"{trace_name} {policy_name} run{run_i}: "
                    f"workload={summary['total_workload_s']:.3f}s captures={summary['capture_count']}"
                )
    all_aggs: dict = {}
    for trace_name in trace_files:
        for policy_name in POLICIES:
            repeat_paths = [
                results_dir / f"{Path(trace_name).stem}_{policy_name}_run{i}.jsonl"
                for i in range(repeats)
            ]
            if repeat_paths[0].exists():
                all_aggs[f"{Path(trace_name).stem}/{policy_name}"] = aggregate_repeats(repeat_paths)

    if all_aggs:
        print("\n=== Aggregate comparison (mean±std) ===")
        print_comparison_table(all_aggs)

    env_path = results_dir / "environment.json"
    write_json(env_path, collect_env(args.config))
    print(f"\nWrote {env_path}")


if __name__ == "__main__":
    main()
