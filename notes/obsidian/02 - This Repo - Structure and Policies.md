# Repo structure and the four policies

Related: [[01 - CUDA Graphs - What and Why]] · [[03 - Commands and Outputs]]

## Directory map

| Path | Role |
|------|------|
| `src/acg/model.py` | `EncoderModel` — HF `AutoModel` + tokenizer; forward returns `last_hidden_state` |
| `src/acg/graph_pool.py` | Per-shape `CapturedShapeGraph` — capture, replay, stats |
| `src/acg/eager.py` | Timed eager baseline |
| `src/acg/policies.py` | Four routing policies + `GraphCache` |
| `src/acg/calibration.py` | Per-shape eager / replay / capture timings → JSON |
| `src/acg/runner.py` | Simulates a trace request-by-request; logs JSONL |
| `src/acg/traces.py` | Workload generators: recurring, many_rare, shift |
| `src/acg/analyze.py` | Aggregates JSONL → comparison table |
| `configs/*.yaml` | Model, shape grid, policy hyperparameters |
| `calibration/*.json` | Device-specific cost table (generated) |
| `traces/` | Fixed 200-request workloads (seeded) |
| `results/` | `trace_policy_runN.jsonl` experiment outputs |

## Shape key

Every request is `(batch_size, seq_length)` from the config grid, e.g. batch `{1,4,8}` × seq `{32,64,128,256}` for MiniLM.

## Workload traces (200 requests each)

| Trace | Intent |
|-------|--------|
| **recurring** | ~80% mass on a few “hot” shapes — graphs should pay off |
| **many_rare** | Uniform random over grid — many one-off shapes — capture rarely amortizes |
| **shift** | First half hot set A, second half hot set B — tests adaptation / wrong precapture |

Seeds in config (`trace_seeds`) keep traces reproducible.

## Policy 1 — `always_eager`

- **Behavior:** Never capture; every request uses eager forward.  
- **Role:** **Baseline** for total wall time and correctness.  
- **When it wins:** Rare shapes, or when capture + cache churn dominates.

## Policy 2 — `lru_on_first_use`

- **Behavior:** On cache miss, **always capture** (unless unsafe / over budget). LRU eviction when cache full.  
- **Role:** Naive “graphs whenever possible” — shows **capture storm** on many_rare.  
- **Implementation note:** Evicting one graph **clears entire cache** (PyTorch shared graph pool); still captures aggressively → high `capture_count`.

## Policy 3 — `calibration_fixed`

- **Behavior:**  
  - Requests `0 … observe_k−1` (default 40): **eager only** (observe).  
  - Count shape frequencies in that window; **pre-capture top M** shapes (default 3).  
  - Afterward: replay if shape was precaptured; else eager. **No further captures.**  
- **Role:** Offline-ish “profile then freeze” — strong on **recurring** if hot set stable; weak on **shift** when hot set changes after observe window.

Hyperparameters: `policy3_observe_k`, `policy3_precapture_m`.

## Policy 4 — `adaptive_admission` (novel)

- **Behavior:** Online decision per uncached shape:  
  1. Estimate **P(shape)** from last `policy4_window` requests (default 50).  
  2. From calibration: `savings_per_hit = max(0, eager_p50 − replay_p50)`.  
  3. `expected_savings = P(shape) × policy4_horizon × savings_per_hit`.  
  4. **Capture** if `expected_savings > capture_p50 × policy4_margin` (default margin 1.25).  
  5. Respect **mem_budget_mb**; optional padding route if `padding_enabled: true`.  
- **Role:** Causal, bounded cache, device-calibrated — should beat LRU on rare shapes and compete on recurring/shift.

Hyperparameters: `policy4_window`, `policy4_horizon`, `policy4_margin`.

## JSONL result structure (for mentor deep-dive)

Each run file: `{trace}_{policy}_run{i}.jsonl`

- **Line 1 (`type: meta`):** env (GPU, torch, config hash), `summary.total_workload_s`, `capture_count`, `cache_mem_mb`.  
- **Per request (`type: request`):** shape, `path` (`eager` | `graph`), `latency_s`, `capture_s`, `warmup_s`, `reason`, `correct`.

**Primary metric for policies:** `summary.total_workload_s` — end-to-end simulated serving time for 200 requests (includes capture/warmup when they happen).

Secondary: p50/p95 per-request latency, capture count, cache memory, correctness rate.

## Correctness

`src/acg/correctness.py` compares graph vs eager outputs on a sample of requests. Stored experiment runs report **correctness_rate = 1.0** for completed MiniLM and partial Qwen runs.
