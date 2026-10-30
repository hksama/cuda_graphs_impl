# Final results — stats for your mentor

Related: [[03 - Commands and Outputs]] · [[02 - This Repo - Structure and Policies]]

**How to refresh this section:** run `python scripts/analyze_results.py` and recompute % vs eager from the first JSONL meta line in each `*_always_eager_run*.jsonl`.

---

## Executive summary (MiniLM on RTX 5060 — 3 repeats)

**Baseline:** Policy `always_eager` — no graph capture overhead, pure eager forwards.

**Headline findings:**

1. **Naive LRU (`lru_on_first_use`) is much worse than eager** on all three traces — too many captures on a 12-shape grid with cache size 4 and full-pool flush on eviction.  
2. **`calibration_fixed` wins recurring** (~16% faster than eager) by precapturing the right hot shapes after 40 observe steps.  
3. **`adaptive_admission` wins many_rare and shift** vs eager (~9% and ~17%) by **refusing** unprofitable captures; on recurring it is slightly **slower** than eager in these runs (capture choices + variance).  
4. **Correctness:** 1.0000 across aggregated MiniLM runs.

Use this table verbally: *“Total seconds for 200 requests, lower is better.”*

### MiniLM — `total_workload_s` (mean ± stdev, seconds)

| Trace | always_eager | lru_on_first_use | calibration_fixed | adaptive_admission |
|-------|--------------|------------------|-------------------|------------------|
| recurring | 0.859 ± 0.36 | 2.632 ± 0.10 | **0.725 ± 0.01** | 0.934 ± 0.50 |
| many_rare | 0.937 ± 0.04 | 4.783 ± 0.19 | 1.217 ± 0.05 | **0.855 ± 0.02** |
| shift | 1.216 ± 0.03 | 3.110 ± 0.29 | 1.427 ± 0.04 | **1.015 ± 0.00** |

### MiniLM — percent vs eager (total time)

| Trace | lru | calibration_fixed | adaptive |
|-------|-----|-------------------|----------|
| recurring | −206% (worse) | **+15.7%** (better) | −8.7% |
| many_rare | −410% | −29.9% | **+8.8%** |
| shift | −156% | −17.4% | **+16.5%** |

*(Positive % = faster than eager = less total seconds.)*

### MiniLM — capture counts (mean)

| Trace | lru | calib_fixed | adaptive |
|-------|-----|-------------|----------|
| recurring | 89 | 3 | 3 |
| many_rare | 147 | 3 | 3 |
| shift | 59 | 3 | 7 |

**Talking point:** Adaptive keeps captures **low** like calibration_fixed on many_rare but **adds a few** on shift (7) when admission pays off — without LRU’s 59–147 captures.

---

## Qwen2.5-1.5B (partial runs on this machine — 1 repeat)

Results under `results/qwen2_5_1_5b/`. **Shift trace policies incomplete** in stored data (only eager present for shift).

**Per-request work is ~10× heavier** (total_s ~5–15 s vs ~0.9–1.2 s for MiniLM).

| Trace | always_eager | calibration_fixed | adaptive | lru |
|-------|--------------|-------------------|----------|-----|
| recurring | 5.76 s | **4.27 s (+26%)** | 10.87 s (−89%) | 11.59 s |
| many_rare | 5.14 s | **4.89 s (+5%)** | 15.43 s (−200%) | 15.35 s |

**Talking points for Qwen:**

- **Fixed precapture still helps** on recurring / many_rare when hot shapes match calibration grid.  
- **Adaptive + LRU behaved badly** in this partial benchmark — likely **mis-calibrated admission**, **too many captures**, or **bf16 1.5B capture/replay economics** differ from MiniLM; worth re-running with `--repeats 3` after full `run_large_model_benchmark.py`.  
- Always mention **model-specific calibration file**: `calibration/rtx5060_qwen2_5_1_5b.json`.

---

## Per-shape micro-savings (calibration, not policy)

From device calibration (MiniLM, RTX 5060), each **successful graph replay** saves on the order of **~0.3–0.9 ms p50** per hit for smaller shapes — modest per hit, but adds up when **P(shape)** is high (recurring trace).

Run:

```bash
python final_res.py
```

---

## Suggested 60-second mentor close

> We built a simulator for variable-shape encoder inference with four routing policies. CUDA graphs cut per-forward launch overhead when shapes repeat, but capture is expensive. Naive capture-on-miss destroys end-to-end time when shapes are rare. Our **adaptive admission** policy uses a rolling demand estimate and **GPU-calibrated** capture vs replay costs to capture sparingly; on MiniLM it beats eager on **many_rare** and **shift**, while **calibration_fixed** wins **recurring**. The code is **model-agnostic** via Hugging Face AutoModel and YAML configs; we validated on MiniLM and started Qwen2.5-1.5B. All runs log JSONL with total wall time, captures, and correctness — aggregated with `analyze_results.py`.

---

## Anticipated mentor questions

| Question | Short answer |
|----------|--------------|
| What is novel? | **Online admission** under bounded cache + memory budget, causal demand window, calibrated costs — not graph capture itself. |
| Metric? | **`total_workload_s`** for 200 requests (includes capture/warmup). |
| vs torch.compile? | Different layer — we study **multi-shape cache policy**, not compiler fusion. |
| Any model? | Any HF encoder forward we wrap; needs CUDA, per-model calibration, sensible shape grid for VRAM. |
| Why LRU bad? | Many shapes → repeated captures; **shared graph pool** forces **full cache flush** on eviction. |
| Correctness? | Sampled eager vs graph comparison; reported rate 1.0 on completed MiniLM aggregates. |
