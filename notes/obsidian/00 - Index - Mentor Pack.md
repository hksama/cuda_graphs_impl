# Adaptive CUDA Graph Admission — Obsidian notes

Use this folder as a **mentor briefing pack**. Copy any file into Obsidian (or open the whole `notes/obsidian/` folder as a vault subfolder).

## Reading order

1. [[01 - CUDA Graphs - What and Why]]
2. [[02 - This Repo - Structure and Policies]]
3. [[03 - Commands and Outputs]]
4. [[04 - Final Results and Talking Points]]

## One-sentence pitch

This project is **not** “how to capture a CUDA graph” — it studies **when** to capture and replay graphs for **variable batch × sequence length** text encoders, under a **small graph cache** and **GPU memory budget**, using **online admission** driven by **device-calibrated** costs.

## Novelty vs prior art

| Prior art | This repo |
|-----------|-----------|
| PyTorch `CUDAGraph`, `torch.compile` graphs | Uses them as a **mechanism** |
| vLLM / serving stacks (LLM decode graphs) | Focus on **encoder-style** forwards, **many shapes**, **bounded cache** |
| “Capture everything” | **Policies** that trade capture cost vs expected replay savings |

## Is this code tied to one model?

**No — it is config-driven for any Hugging Face `AutoModel` encoder-style forward** (`last_hidden_state`, `use_cache=False`). Bundled configs:

- `configs/default.yaml` — `sentence-transformers/all-MiniLM-L6-v2` (small baseline)
- `configs/qwen2_5_1_5b.yaml` — `Qwen/Qwen2.5-1.5B` (bf16, smaller shape grid)
- `configs/smollm2_1_7b.yaml` — SmolLM2-1.7B

You add a YAML with `model_id`, `dtype`, `batch_sizes`, `seq_lengths`, and optional `traces_subdir` / `results_subdir` / `calibration_basename`. **Calibration is per device and per model** — do not reuse MiniLM calibration for Qwen.

**Caveats for “any model”:**

- Needs **NVIDIA GPU + CUDA** (WSL2 OK if `nvidia-smi` works).
- Model must run a **static-shape forward** for graph capture (same batch, seq at capture time).
- Very large models need **bf16** and smaller grids on 8GB GPUs.
- Gated weights (e.g. Gemma 2) need HF access and `trust_remote_code` if required.

## Hardware used in stored results

- **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU  
- **Stack:** PyTorch 2.14+cu130, WSL2 Linux  
- **Correctness:** sampled every 20 requests; stored runs show **100%** match vs eager within configured tolerances
