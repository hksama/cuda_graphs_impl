# Adaptive CUDA Graph Admission for Variable-Shape Text Inference

Research prototype: **when** to capture and replay CUDA graphs for a variable-shape text encoder (`sentence-transformers/all-MiniLM-L6-v2` via Hugging Face `AutoModel`), not graph capture mechanics themselves.

## Novelty claim

**Novel:** online **admission** and **route selection** (eager vs graph replay vs optional pad-to-captured-shape) under a **bounded graph-cache** and **GPU memory budget**, using **causal** rolling-window demand estimates and **device-calibrated** capture/replay costs.

**Prior art (acknowledged):** PyTorch `torch.cuda.CUDAGraph` and `torch.compile` CUDA graphs; Torch-TensorRT; vLLM piecewise/full graph capture for LLM serving. This repo studies **variable-shape encoder** workloads and policies for **capture admission**, not kernel fusion or serving-stack integration.

## Setup

```bash
cd cuda_graph_impl
source .venv/bin/activate   # or: uv venv .venv && uv pip install -r requirements.txt
```

Requires one NVIDIA GPU (developed on RTX 5060 Laptop; use a separate calibration file on DGX Spark).

### GPU / WSL troubleshooting

If you see `Found no NVIDIA driver on your system`:

1. In the **same terminal** as Python, run `nvidia-smi`. It must succeed before any script here will work.
2. **WSL2:** install the latest [NVIDIA Windows driver with WSL support](https://developer.nvidia.com/cuda/wsl), reboot Windows, open a new WSL shell.
3. Verify `ls /dev/dxg` exists in WSL.
4. Reinstall PyTorch with CUDA only after `nvidia-smi` works:  
   `uv pip install torch --index-url https://download.pytorch.org/whl/cu130` (match your driver/CUDA).

This error is environmental — not caused by the model config or Qwen weights download (loading reached 100% on CPU before `.cuda()` failed).

## Quick checks

```bash
# One-shape eager vs graph correctness + output ownership
python scripts/verify_one_shape.py --batch 4 --seq 128

# Device calibration only (writes calibration/rtx5060.json)
python scripts/run_experiment.py --calibrate-only

# Full trace experiment (3 traces × 4 policies × 3 repeats)
python scripts/run_experiment.py --generate-traces
```

Results land in `results/`; reproducible traces in `traces/`.

## Configuration

[`configs/default.yaml`](configs/default.yaml): shape grid batch `{1,4,8}` × seq `{32,64,128,256}`, graph cache size 4, memory budget, policy hyperparameters, `padding_enabled` (default `false`).

## Policies

| ID | Name | Behavior |
|----|------|----------|
| 1 | `always_eager` | No CUDA graphs |
| 2 | `lru_on_first_use` | Capture on miss; LRU eviction (max 4) |
| 3 | `calibration_fixed` | Observe first 40 requests; pre-capture top 3 shapes; no further captures |
| 4 | `adaptive_admission` | Rolling window demand vs calibrated capture cost + margin |

## Layout

- `src/acg/` — model, eager/graph paths, policies, runner, traces, calibration
- `scripts/` — verification and experiment driver
- `calibration/` — per-device JSON (e.g. `rtx5060.json`, `dgx_spark.json`)
- `traces/` — seeded workload traces: recurring, many_rare, shift

## CUDA graph cache eviction

Captured graphs share one CUDA graph memory pool per `GraphCache`. Evicting one graph invalidates siblings, so LRU eviction **flushes the entire cache** and resets the pool before capturing again. This matches PyTorch shared-pool semantics and avoids illegal memory access after partial eviction.

## 1B–2B models (Qwen2.5-1.5B, SmolLM2-1.7B)

On 8GB GPUs use **bf16** and a **smaller shape grid** (see `configs/qwen2_5_1_5b.yaml`, `configs/smollm2_1_7b.yaml`). Each model gets its own traces, calibration, and results subdirs.

```bash
# One model, full pipeline (verify → calibrate → 3 traces × 4 policies)
python scripts/run_large_model_benchmark.py --config configs/qwen2_5_1_5b.yaml

# Faster iteration
python scripts/run_large_model_benchmark.py --config configs/qwen2_5_1_5b.yaml --repeats 1

# Both bundled ~1–2B configs
python scripts/run_large_model_benchmark.py
```

Manual steps:

```bash
python scripts/verify_one_shape.py --config configs/qwen2_5_1_5b.yaml --batch 1 --seq 64
python scripts/run_experiment.py --config configs/qwen2_5_1_5b.yaml --generate-traces --calibrate-only
python scripts/run_experiment.py --config configs/qwen2_5_1_5b.yaml --repeats 1
python scripts/analyze_results.py --results-dir results/qwen2_5_1_5b
```

**Gated 2B models** (e.g. Gemma 2) require Hugging Face access; add a yaml with `model_id` and `trust_remote_code` as needed.

**FP32** on 1.5B+ needs ~6GB+ weights alone—prefer `dtype: bfloat16` on consumer GPUs. Compare savings via `analyze_results.py` on `results/<model_subdir>/` (same as MiniLM: `total_s` vs `always_eager`).

## Padding ablation

Set `padding_enabled: true` in config to allow routing uncaptured shapes to the smallest compatible captured shape when calibrated latency favors padding. Keep `false` for the padding-disabled baseline.
