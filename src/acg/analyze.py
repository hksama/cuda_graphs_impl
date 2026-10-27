from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


def load_jsonl(path: Path) -> tuple[dict, List[dict]]:
    meta = {}
    records = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("type") == "meta":
                meta = obj
            elif obj.get("type") == "request":
                records.append(obj)
    return meta, records


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(round((p / 100.0) * (len(sorted_v) - 1)))
    return float(sorted_v[idx])


def summarize_run(meta: dict, records: List[dict]) -> Dict[str, Any]:
    lats = [r["latency_s"] for r in records]
    correct = sum(1 for r in records if r.get("correct", True))
    return {
        "total_workload_s": meta.get("summary", {}).get("total_workload_s", sum(lats)),
        "p50_latency_s": percentile(lats, 50),
        "p95_latency_s": percentile(lats, 95),
        "capture_count": meta.get("summary", {}).get("capture_count", 0),
        "cache_mem_mb": meta.get("summary", {}).get("cache_mem_mb", 0),
        "correctness_rate": correct / len(records) if records else 1.0,
    }


def aggregate_repeats(paths: List[Path]) -> Dict[str, Any]:
    runs = [summarize_run(meta, recs) for meta, recs in (load_jsonl(p) for p in paths)]
    if not runs:
        return {}
    keys = runs[0].keys()
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [r[k] for r in runs]
        out[k] = {
            "mean": statistics.mean(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }
    return out


def print_comparison_table(groups: Dict[str, Dict[str, Any]]) -> None:
    headers = ["policy", "total_s", "p50_s", "p95_s", "captures", "cache_mb", "correct"]
    print("\t".join(headers))
    for name, agg in sorted(groups.items()):
        print(
            "\t".join(
                [
                    name,
                    f"{agg['total_workload_s']['mean']:.4f}±{agg['total_workload_s']['stdev']:.4f}",
                    f"{agg['p50_latency_s']['mean']:.6f}",
                    f"{agg['p95_latency_s']['mean']:.6f}",
                    f"{agg['capture_count']['mean']:.1f}",
                    f"{agg['cache_mem_mb']['mean']:.1f}",
                    f"{agg['correctness_rate']['mean']:.4f}",
                ]
            )
        )
