# Adaptive Speculative Decoding

Research playground for LLM inference acceleration and paper-style benchmarking.

Implemented methods:
- Baseline decoding
- Speculative Sampling (exact)
- AutoJudge (paper-aligned mining + LogisticRegression)
- Top-K verification baseline (lossy)
- SpecExec (exact, branch-KV reuse)

## Quick Start

```bash
make setup
make check
make test
```

## Most Used Commands

```bash
# List presets
make list-presets

# Paper-style Qwen2.5 sweep (GSM8K)
make paper-eval

# Local Qwen2.5 7B/1.5B sweep (GSM8K + LiveCodeBench)
make local-eval

# Local Llama-3 8B/3B sweep (GSM8K + LiveCodeBench)
bash scripts/run_llama3_8b_3b_eval.sh
```

## Llama 48h Run (Recommended Pattern)

Use unique artifacts for each run (checkpoint + outputs + report prefix).

```bash
DATE_TAG="$(date +%F)-llama-48h-cgrid8"
LOG_PATH="logs/llama3_48h_${DATE_TAG}.log"

CHECKPOINT_PATH="datasets/autojudge_llama3_3b_to_8b_${DATE_TAG}.pt" \
OUT_GSM8K="datasets/results_llama3_8b_3b_gsm8k_${DATE_TAG}.jsonl" \
OUT_LCB="datasets/results_llama3_8b_3b_lcb_${DATE_TAG}.jsonl" \
REPORT_PREFIX="reports/yandex_llama3_8b_3b_${DATE_TAG}" \
MANIFEST_PATH="reports/llama3_8b_3b_run_manifest_${DATE_TAG}.json" \
bash scripts/run_llama3_8b_3b_eval.sh | tee "${LOG_PATH}"
```

## Monitoring Commands

```bash
# Live log
tail -f "${LOG_PATH}"

# Ensure paper-aligned AutoJudge grid was used (must show C-grid 1/8 ... 8/8)
grep -n "\[autojudge-train\] C-grid" "${LOG_PATH}"

# GPU memory/utilization
watch -n 10 'nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader'

# Active GPU processes
watch -n 10 'nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader'

# Growth of output JSONL files
watch -n 60 "wc -l ${OUT_GSM8K} ${OUT_LCB}"
```

## Validate Outputs

```bash
.venv/bin/python scripts/validate_results_jsonl.py --path "${OUT_GSM8K}" --strict
.venv/bin/python scripts/validate_results_jsonl.py --path "${OUT_LCB}" --strict
```

## Key Constraints

- Draft and target must have tokenizer/vocab compatibility.
- AutoJudge C-grid policy is paper-aligned only:
  - `1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0`
  - do not use `1e1` or `1e2` in overrides.
- Reusing the same `--out` enables resume mode by `resume_key`.

## Repo Layout

- `sp_samp/` - core methods and HF adapters
- `benchmarks/` - benchmark runner
- `configs/` - model/method/experiment presets
- `scripts/` - eval orchestration, validation, report generation
- `tests/` - unit tests
- `reports/` - tracked summary artifacts
- `datasets/` - local run outputs (gitignored)
