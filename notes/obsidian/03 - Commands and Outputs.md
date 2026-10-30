# Commands to run experiments and get outputs

Related: [[04 - Final Results and Talking Points]] · [[00 - Index - Mentor Pack]]

## Setup (once)

```bash
cd cuda_graph_impl
source .venv/bin/activate
# GPU must work: nvidia-smi
```

## Sanity check (one shape, eager vs graph)

```bash
python scripts/verify_one_shape.py --batch 4 --seq 128
# Large model:
python scripts/verify_one_shape.py --config configs/qwen2_5_1_5b.yaml --batch 1 --seq 64
```

## MiniLM full pipeline (default config)

```bash
# Calibration only → writes calibration/rtx5060.json
python scripts/run_experiment.py --calibrate-only

# Regenerate traces + calibrate + run 3 traces × 4 policies × 3 repeats
python scripts/run_experiment.py --generate-traces

# Faster: single repeat
python scripts/run_experiment.py --repeats 1
```

## Analyze results → table for slides

```bash
python scripts/analyze_results.py
# Custom dir / repeat count:
python scripts/analyze_results.py --results-dir results/qwen2_5_1_5b --repeats 1
```

**Columns:** `policy`, `total_s` (mean±stdev over repeats), `p50_s`, `p95_s`, `captures`, `cache_mb`, `correct`.

## Large model shortcut

```bash
python scripts/run_large_model_benchmark.py --config configs/qwen2_5_1_5b.yaml
python scripts/run_large_model_benchmark.py --config configs/qwen2_5_1_5b.yaml --repeats 1
```

## Per-shape replay savings from calibration (quick script)

```bash
python final_res.py   # reads calibration/rtx5060.json; prints ms saved per hit per shape
```

Or open `calibration/rtx5060.json` / `calibration/rtx5060_qwen2_5_1_5b.json` directly.

## Where files land

| Artifact | Location |
|----------|----------|
| Traces | `traces/` or `traces/qwen2_5_1_5b/` |
| Calibration | `calibration/rtx5060.json`, `rtx5060_qwen2_5_1_5b.json`, … |
| Results | `results/` or `results/qwen2_5_1_5b/` |

## Config knobs worth mentioning to mentor

From `configs/default.yaml` (MiniLM):

- `max_graph_cache: 4`, `mem_budget_mb: 768`  
- `trace_length: 200`, `experiment_repeats: 3`  
- `padding_enabled: false` (baseline) — set `true` for padding ablation  

Qwen config shrinks batch/seq grid and cache (`max_graph_cache: 2`) for 8GB VRAM.

## Repro checklist

1. Same GPU / calibration file for fair policy comparison.  
2. Same trace files (SHA in JSONL meta).  
3. Compare policies against **`always_eager` total_s**, not against each other only.  
4. Report **capture_count** — explains LRU regressions.
