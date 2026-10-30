# CUDA graphs — what they are and why they save time

Related: [[02 - This Repo - Structure and Policies]] · [[04 - Final Results and Talking Points]]

## The problem (eager PyTorch)

Each forward pass normally goes through:

1. Python / dispatcher overhead  
2. CUDA kernel **launch** per op (attention, matmul, layernorm, …)  
3. Actual GPU work  

For **small** models or **short** sequences, launch and framework overhead can be a large fraction of wall time. Re-running the **same** operator sequence many times still pays that cost every time.

## What a CUDA graph does

A **CUDA graph** records a fixed sequence of GPU work once, then **replays** it with minimal launch overhead.

In this repo (see `src/acg/graph_pool.py`):

1. **Warmup** — several eager forwards on static buffers (CUDA caching, cudnn heuristics).  
2. **Capture** — `with torch.cuda.graph(...):` runs one forward into static tensors.  
3. **Replay** — copy new `input_ids` / `attention_mask` into static buffers, call `graph.replay()`, read output.

Important constraints:

- **Graph topology is fixed** — same ops, same tensor shapes. Here each graph is keyed by `(batch_size, seq_len)`.  
- **Inputs change via in-place copy** into static tensors, not by rebuilding the graph.  
- **Outputs** are cloned after replay so callers do not hold aliases to static memory.

## Where time is saved

Per request, compare:

- **Eager latency** — full forward each time (`eager_latency_p50_s` in calibration JSON).  
- **Replay latency** — copy inputs + replay + clone output (`replay_latency_p50_s`).

**Savings per graph hit** ≈ `eager − replay` (see `calibration/rtx5060.json` or run `final_res.py` on that file).

Example from RTX 5060 + MiniLM calibration (per hit, p50):

| Shape (batch_seq) | Eager ms | Replay ms | Save ms/hit |
|-------------------|----------|-----------|-------------|
| 1_32 | ~1.62 | ~0.72 | ~0.90 |
| 4_128 | (see JSON) | (see JSON) | (see JSON) |

Replay is **not free**: input `copy_`, synchronize, and output `clone` are included in measured replay time.

## Where time is *spent* (capture cost)

First time you use a shape (or after cache flush):

- **Warmup** — often hundreds of ms on first shape, much less afterward.  
- **Capture** — one-time graph record (`capture_latency_p50_s`).  

Policies must ensure: **expected future savings > capture cost** (with margin), or you lose vs always-eager.

## Why variable shapes hurt naive graph use

Real workloads mix `(batch, seq)` pairs. Each distinct pair needs its **own** graph. Naive strategy: **capture on first sight** (Policy 2) → many captures, cache evictions, and on this codebase **full cache flush** on LRU eviction (shared CUDA graph pool — see README).

That is the research question: **admission** — which shapes deserve a graph slot under **max_graph_cache** (e.g. 4) and **mem_budget_mb** (e.g. 768)?

## Mental model diagram

```mermaid
flowchart LR
  subgraph eager [Eager path]
    A[Python forward] --> B[Many kernel launches]
  end
  subgraph graph [Graph path]
    C[Copy inputs to static] --> D[graph.replay]
    D --> E[Clone output]
  end
  F{Shape in cache?}
  F -->|no| G[Capture + warmup cost]
  F -->|yes| graph
  G --> graph
```

## What this repo does *not* claim

- It does not implement vLLM-style decode scheduling or `torch.compile` fusion.  
- It does not remove the need for **per-shape** graphs for variable-length batches.  
- Padding-to-graph routing exists as an **optional ablation** (`padding_enabled: true`), default **off**.
