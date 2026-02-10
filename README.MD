# Speculative Sampling Playground

This project provides:
- Exact speculative sampling (SpS).
- AutoJudge (lossy judge-decoding style method with synthetic labels, no manual annotation).
- A Hugging Face adapter with KV cache and optional quantization.
- A benchmark harness on MT-Bench with JSONL metrics.

**Features**
1. Baseline, speculative, and AutoJudge decoding in one benchmark entrypoint.
2. MT-Bench loader (JSON/JSONL).
3. Benchmark runner with median timing and method-specific metrics.
4. Preset configs for models, methods, and paired experiments.
5. Makefile shortcuts for local and Docker workflows.
6. Docker support for CPU and GPU.

**Getting Started (From Zero)**
1. Install Docker. For GPU runs, install NVIDIA drivers and the NVIDIA Container Toolkit.
2. Get the MT‑Bench dataset file (JSON/JSONL) and place it on disk, for example `/data/mt_bench.jsonl`.
3. Build a CPU image:
```bash
docker build -t sp-samp .
```
4. Run tests (CPU):
```bash
docker run --rm sp-samp
```
5. Run a CPU benchmark (toy models):
```bash
docker run --rm sp-samp \
  python -m benchmarks.bench_speculative \
  --method both \
  --runs 1 \
  --max-samples 5 \
  --max-new-tokens 32 \
  --vocab-size 2048
```
6. Build a GPU image (CUDA example):
```bash
docker build -f Dockerfile.gpu \
  --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
  --build-arg TORCH_VERSION=2.3.1 \
  -t sp-samp-gpu .
```
7. Run a GPU benchmark (HF model, results saved to JSONL):
```bash
docker run --rm --gpus all -v /path/to/mtbench:/data sp-samp-gpu \
  python -m benchmarks.bench_speculative \
  --dataset /data/mt_bench.jsonl \
  --hf-model RedHatAI/gpt-oss-20b \
  --quant 4bit \
  --bnb-compute-dtype bfloat16 \
  --device cuda \
  --use-chat-template \
  --max-samples 50 \
  --max-new-tokens 128 \
  --k 4 \
  --runs 5 \
  --out /data/results.jsonl
```
8. Run all methods in one launch (baseline + speculative + autojudge):
```bash
docker run --rm --gpus all -v /path/to/mtbench:/data sp-samp-gpu \
  python -m benchmarks.bench_speculative \
  --dataset /data/mt_bench.jsonl \
  --hf-model meta-llama/Meta-Llama-3-8B-Instruct \
  --hf-draft-model meta-llama/Meta-Llama-3-8B-Instruct \
  --device cuda \
  --use-chat-template \
  --method all \
  --k 4 \
  --runs 5 \
  --out /data/results_all.jsonl
```
9. Run AutoJudge only with checkpoint reuse:
```bash
docker run --rm --gpus all -v /path/to/mtbench:/data sp-samp-gpu \
  python -m benchmarks.bench_speculative \
  --dataset /data/mt_bench.jsonl \
  --hf-model meta-llama/Meta-Llama-3-8B-Instruct \
  --hf-draft-model meta-llama/Meta-Llama-3-8B-Instruct \
  --device cuda \
  --use-chat-template \
  --method autojudge \
  --autojudge-train-samples 4000 \
  --autojudge-train-steps 400 \
  --autojudge-threshold 0.5 \
  --autojudge-checkpoint /data/autojudge_llama3.pt \
  --out /data/results_autojudge.jsonl
```

**Make Targets**
1. Show all commands:
```bash
make help
```
2. Syntax check:
```bash
make check
```
3. List presets:
```bash
make list-presets
```
4. Validate config logic:
```bash
make validate-configs
```
5. Quick toy benchmark (no HF models):
```bash
make bench-toy OUT=/tmp/bench_toy.jsonl
```
6. Quick HF smoke run (needs `torch` + `transformers`, downloads tiny model):
```bash
make smoke-hf OUT=/tmp/smoke_hf.jsonl
```
7. Run experiment on MT-Bench:
```bash
make bench DATASET=/path/to/mt_bench.jsonl OUT=/path/to/results.jsonl
```
8. Run AutoJudge preset:
```bash
make autojudge DATASET=/path/to/mt_bench.jsonl OUT=/path/to/results_autojudge.jsonl
```
9. Build and run GPU Docker flow:
```bash
make docker-build-gpu
make docker-bench DATASET=/path/to/mt_bench.jsonl OUT=/path/to/results.jsonl
```

**Presets**
- Models: `configs/models.json`
- Methods: `configs/methods.json`
- Experiments (target/draft pairings): `configs/experiments.json`
- Method templates (future AutoJudge/SpecExec): `configs/method_templates.json`

**CLI Runner**
1. List presets:
```bash
python -m sp_samp.cli list-presets --config-dir configs
```
2. Direct method selection:
```bash
python -m benchmarks.bench_speculative \
  --method autojudge \
  --dataset /path/to/mt_bench.jsonl \
  --hf-model meta-llama/Meta-Llama-3-8B-Instruct \
  --hf-draft-model meta-llama/Meta-Llama-3-8B-Instruct
```
3. Run benchmark using presets:
```bash
python -m sp_samp.cli bench \
  --config-dir configs \
  --model-preset gpt_oss_20b_4bit \
  --method-preset speculative_k4 \
  --dataset /path/to/mt_bench.jsonl \
  --out results.jsonl
```
4. Run benchmark using an experiment preset:
```bash
python -m sp_samp.cli bench \
  --config-dir configs \
  --experiment llama3_all_methods \
  --dataset /path/to/mt_bench.jsonl \
  --out results.jsonl
```
5. Run AutoJudge shortcut command:
```bash
python -m sp_samp.cli autojudge \
  --config-dir configs \
  --experiment llama3_target_llama3_autojudge_k4 \
  --dataset /path/to/mt_bench.jsonl \
  --out results_autojudge.jsonl
```

**Metrics Output**
The benchmark writes JSONL records with per-run metrics and a summary record per method. Fields include:
- `tokens_per_sec`
- `acceptance_rate`
- `avg_tokens_per_step`
- `proposed`, `accepted`, `rejections`
- `judge_accept_rate` (AutoJudge only)
- `target_fallback_rate` (AutoJudge only)
- `target_calls_per_token` (AutoJudge only)
- `autojudge_train_samples`, `autojudge_train_loss` (AutoJudge only)

**Project Layout**
- `sp_samp/`: core library, AutoJudge, HF adapter.
- `benchmarks/`: benchmark runner.
- `configs/`: preset configs.
- `tests/`: tests.
- `Dockerfile`, `Dockerfile.gpu`: containers.

**Notes**
- SpecExec is still placeholder.
- Draft and target models must share an identical tokenizer vocabulary mapping for speculative/AutoJudge correctness.
- `make validate-configs` checks config references and tokenizer compatibility in `configs/*.json`.
- Current default experiments are correctness-safe (same tokenizer family) and may show limited speedup when target=draft.
