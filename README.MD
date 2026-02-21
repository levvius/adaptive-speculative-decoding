# Speculative Sampling Playground

This project provides:
- Exact speculative sampling (SpS).
- AutoJudge (paper-aligned judge decoding: Algorithm 1 mining + Logistic Regression classifier).
- Top-K lossy baseline for paper-style comparisons.
- SpecExec (exact target sampling with draft-branch cache prefill and pruning).
- A Hugging Face adapter with KV cache and optional quantization.
- A benchmark harness on MT-Bench with JSONL metrics.

**Features**
1. Baseline, speculative, AutoJudge, Top-K, and SpecExec decoding in one benchmark entrypoint.
2. MT-Bench loader (JSON/JSONL).
3. Benchmark runner with median timing, resume support, and method-specific metrics.
4. Preset configs for models, methods, and paired experiments.
5. Makefile shortcuts for local and Docker workflows.
6. Docker support for CPU and GPU.
7. CI pipeline (GitHub Actions) for checks/tests + benchmark JSONL schema validation.

**Getting Started (From Zero)**
1. Bootstrap dependencies on a clean Ubuntu host (safe mode, does not touch NVIDIA driver):
```bash
bash scripts/install_dependencies.sh
```
Recommended Python version is `3.11` (`.python-version` in repo). Dependencies are pinned in `requirements*.txt` for reproducible runs.
For GPU Python extras (`bitsandbytes`, `accelerate`):
```bash
bash scripts/install_dependencies.sh --gpu
```
For EOL Ubuntu (for example Ubuntu 17), script stops by default. Continue only if you explicitly accept risks:
```bash
bash scripts/install_dependencies.sh --allow-eol-ubuntu
```
2. Install Docker Engine (Ubuntu 24.04 example):
```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo ${UBUNTU_CODENAME:-$VERSION_CODENAME}) stable" | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
```
For GPU runs, keep your existing NVIDIA driver and install NVIDIA Container Toolkit:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
3. Put MT‑Bench dataset file (JSON/JSONL) into project folder `datasets/`, for example `datasets/mt_bench.jsonl`.
4. Build a CPU image:
```bash
docker build -t sp-samp .
```
5. Run tests (CPU):
```bash
docker run --rm sp-samp
```
6. Run a CPU benchmark (toy models):
```bash
docker run --rm sp-samp \
  python -m benchmarks.bench_speculative \
  --method both \
  --runs 1 \
  --max-samples 5 \
  --max-new-tokens 32 \
  --vocab-size 2048
```
7. Build a GPU image (CUDA example):
```bash
docker build -f Dockerfile.gpu \
  --build-arg BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  --build-arg TORCH_VERSION=2.9.1 \
  -t sp-samp-gpu .
```
8. Run a GPU benchmark (HF model, results saved to JSONL):
```bash
docker run --rm --gpus all -v "$(pwd)/datasets:/data" sp-samp-gpu \
  python -m benchmarks.bench_speculative \
  --dataset /data/mt_bench.jsonl \
  --hf-model RedHatAI/gpt-oss-20b \
  --device cuda \
  --use-chat-template \
  --max-samples 50 \
  --max-new-tokens 128 \
  --k 4 \
  --runs 5 \
  --out /data/results.jsonl
```
9. Run all methods in one launch (baseline + speculative + autojudge + topk + specexec):
```bash
docker run --rm --gpus all -v "$(pwd)/datasets:/data" sp-samp-gpu \
  python -m benchmarks.bench_speculative \
  --dataset /data/mt_bench.jsonl \
  --hf-model Qwen/Qwen2.5-3B-Instruct \
  --hf-draft-model Qwen/Qwen2.5-0.5B-Instruct \
  --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
  --draft-tokenizer Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --use-chat-template \
  --method all \
  --k 4 \
  --runs 5 \
  --out /data/results_all.jsonl
```
10. Run SpecExec only (branch execution parameters included):
```bash
docker run --rm --gpus all -v "$(pwd)/datasets:/data" sp-samp-gpu \
  python -m sp_samp.cli specexec \
  --config-dir configs \
  --experiment qwen25_3b_target_qwen25_0p5b_specexec_k4 \
  --dataset /data/mt_bench.jsonl \
  --parallel-branches 8 \
  --branch-prune-threshold 0.0 \
  --out /data/results_specexec.jsonl
```
11. Run AutoJudge only with checkpoint reuse (paper-aligned):
```bash
docker run --rm --gpus all -v "$(pwd)/datasets:/data" sp-samp-gpu \
  python -m benchmarks.bench_speculative \
  --dataset /data/gsm8k_train.jsonl \
  --hf-model Qwen/Qwen2.5-3B-Instruct \
  --hf-draft-model Qwen/Qwen2.5-0.5B-Instruct \
  --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
  --draft-tokenizer Qwen/Qwen2.5-0.5B-Instruct \
  --device cuda \
  --use-chat-template \
  --method autojudge \
  --autojudge-task gsm8k \
  --autojudge-train-dataset /data/gsm8k_train.jsonl \
  --autojudge-train-samples 4000 \
  --autojudge-recall-target 0.9 \
  --autojudge-train-split 0.9 \
  --autojudge-checkpoint /data/autojudge_llama3.pt \
  --out /data/results_autojudge.jsonl
```

**Make Targets**
Defaults for benchmark paths:
- `DATASET=datasets/mt_bench.jsonl`
- `OUT=datasets/results.jsonl`
1. Show all commands:
```bash
make help
```
2. Install/upgrade dependencies in safe mode:
```bash
make setup
```
3. Install/upgrade including GPU Python extras:
```bash
make setup-gpu
```
4. Syntax check:
```bash
make check
```
5. Validate benchmark JSONL schema:
```bash
make validate-results RESULTS=datasets/results.jsonl
```
6. List presets:
```bash
make list-presets
```
7. Validate config logic:
```bash
make validate-configs
```
8. Quick toy benchmark (no HF models):
```bash
make bench-toy OUT=/tmp/bench_toy.jsonl
```
9. Quick HF smoke run (needs `torch` + `transformers`, downloads tiny model):
```bash
make smoke-hf OUT=/tmp/smoke_hf.jsonl
```
GPU host variant:
```bash
make smoke-hf-gpu OUT=/tmp/smoke_hf_gpu.jsonl
python scripts/validate_results_jsonl.py --path /tmp/smoke_hf_gpu.jsonl --strict
```
Expected result for the GPU variant: one run summary in console and 2 JSONL records (`run` + `summary`) in output file.
10. Run experiment on MT-Bench:
```bash
make bench DATASET=datasets/mt_bench.jsonl OUT=datasets/results.jsonl
```
11. Run AutoJudge preset:
```bash
make autojudge DATASET=datasets/mt_bench.jsonl OUT=datasets/results_autojudge.jsonl
```
12. Run SpecExec preset:
```bash
make specexec DATASET=datasets/mt_bench.jsonl OUT=datasets/results_specexec.jsonl
```
13. Build and run GPU Docker flow:
```bash
make docker-build-gpu
make docker-bench DATASET=datasets/mt_bench.jsonl OUT=datasets/results.jsonl
make docker-specexec DATASET=datasets/mt_bench.jsonl OUT=datasets/results_specexec.jsonl
```
For one-off sudo usage without docker-group setup:
```bash
make docker-gpu-check DOCKER_CMD="sudo docker"
make docker-build-gpu-safe DOCKER_CMD="sudo docker"
```
If your Docker host hits BuildKit snapshot/export errors, use:
```bash
make docker-build-gpu-safe
```
or clean builder cache:
```bash
make docker-prune-builder
```
Check GPU passthrough before long runs:
```bash
make docker-gpu-check
make docker-gpu-check-image
```
`docker-gpu-check` first tries `nvidia-smi` in a clean CUDA container. If NVML fails there, it falls back to a `torch.cuda` check in your built image.
If fallback says image missing, build it first with `make docker-build-gpu-safe`.
14. Enforce headless GPU mode for long runs:
```bash
make bench DATASET=datasets/mt_bench.jsonl OUT=datasets/results.jsonl HEADLESS=1
```
15. Short load run in Docker (gpt_oss_20b_4bit preset):
```bash
sudo docker run --rm --gpus all -v "$(pwd)/datasets:/data" sp-samp-gpu \
  python -m sp_samp.cli bench \
  --config-dir configs \
  --experiment gptoss20b_target_gptoss20b_draft_k4 \
  --dataset /data/mt_bench.jsonl \
  --runs 1 \
  --max-samples 20 \
  --max-new-tokens 128 \
  --out /data/results_gptoss20b_load.jsonl
```
If `gpt-oss-20b` fails to load because of local bitsandbytes/triton runtime mismatch, fallback to:
```bash
sudo docker run --rm --gpus all -v "$(pwd)/datasets:/data" sp-samp-gpu \
  python -m sp_samp.cli bench \
  --config-dir configs \
  --experiment mistral_target_mistral_draft_k4 \
  --dataset /data/mt_bench.jsonl \
  --runs 1 \
  --max-samples 20 \
  --max-new-tokens 128 \
  --out /data/results_gptoss20b_load.jsonl
```
Validate the output:
```bash
python scripts/validate_results_jsonl.py --path datasets/results_gptoss20b_load.jsonl --strict
```
16. Paper-style GSM8K AutoJudge evaluation (Qwen2.5 0.5B -> 3B, medium sweep):
```bash
mkdir -p datasets
curl -fsSL https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl -o datasets/gsm8k_train.jsonl
curl -fsSL https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl -o datasets/gsm8k_test.jsonl
make paper-eval
```
Raw JSONL stays in `datasets/`; report artifacts are written to `reports/` as:
- `reports/autojudge_qwen25_paper_<date>.md`
- `reports/autojudge_qwen25_paper_<date>.csv`
- `reports/autojudge_qwen25_paper_<date>.json`

The benchmark supports dedicated paper-eval switches:
- `--eval-task mtbench|gsm8k`
- `--gsm8k-eval-mode zero_shot_cot|plain`
- `--topk-rank <int|all>`
- `--topk-grid <csv>`
- GSM8K output fields in JSONL: `gsm8k_exact_match`, `gsm8k_correct`, `gsm8k_total`

**Troubleshooting (RTX 50xx / Blackwell)**
- Symptom: `sm_120 is not compatible with the current PyTorch installation` or `no kernel image is available for execution on the device`.
- Cause: torch build without native Blackwell kernels.
- Fix: rebuild with cu128+ torch:
```bash
make docker-build-gpu-safe \
  CUDA_BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04 \
  TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  TORCH_VERSION=2.9.1
```

**Presets**
- Models: `configs/models.json`
- Methods: `configs/methods.json`
- Experiments (target/draft pairings): `configs/experiments.json`
- Method templates (AutoJudge/Top-K/SpecExec): `configs/method_templates.json`

**CLI Runner**
1. List presets:
```bash
python -m sp_samp.cli list-presets --config-dir configs
```
2. Direct method selection:
```bash
python -m benchmarks.bench_speculative \
  --method specexec \
  --dataset datasets/mt_bench.jsonl \
  --hf-model meta-llama/Meta-Llama-3-8B-Instruct \
  --hf-draft-model meta-llama/Meta-Llama-3-8B-Instruct \
  --parallel-branches 8 \
  --branch-prune-threshold 0.0
```
3. Run benchmark using presets:
```bash
python -m sp_samp.cli bench \
  --config-dir configs \
  --model-preset gpt_oss_20b_4bit \
  --method-preset speculative_k4 \
  --dataset datasets/mt_bench.jsonl \
  --out datasets/results.jsonl
```
4. Run benchmark using an experiment preset:
```bash
python -m sp_samp.cli bench \
  --config-dir configs \
  --experiment qwen25_3b_target_qwen25_0p5b_all_methods \
  --dataset datasets/mt_bench.jsonl \
  --out datasets/results.jsonl
```
5. Run paper-method bundle only (baseline + speculative + autojudge + topk):
```bash
python -m sp_samp.cli bench \
  --config-dir configs \
  --experiment qwen25_3b_target_qwen25_0p5b_all_paper \
  --dataset datasets/gsm8k_test.jsonl \
  --eval-task gsm8k \
  --gsm8k-eval-mode zero_shot_cot \
  --out datasets/results_all_paper.jsonl
```
6. Run AutoJudge shortcut command:
```bash
python -m sp_samp.cli autojudge \
  --config-dir configs \
  --experiment qwen25_3b_target_qwen25_0p5b_autojudge_k4 \
  --dataset datasets/mt_bench.jsonl \
  --out datasets/results_autojudge.jsonl
```
7. Run SpecExec shortcut command:
```bash
python -m sp_samp.cli specexec \
  --config-dir configs \
  --experiment qwen25_3b_target_qwen25_0p5b_specexec_k4 \
  --dataset datasets/mt_bench.jsonl \
  --out datasets/results_specexec.jsonl
```
7. Require headless GPU mode (fail-fast when display is active):
```bash
python -m sp_samp.cli bench \
  --config-dir configs \
  --experiment qwen25_3b_target_qwen25_0p5b_all_methods \
  --dataset datasets/mt_bench.jsonl \
  --require-headless \
  --out datasets/results.jsonl
```

**Metrics Output**
The benchmark writes JSONL records with per-run metrics and a summary record per method. Fields include:
- `status` (`ok`/`error`/`skipped`)
- `resume_key` (used to skip completed runs on re-launch)
- `tokens_per_sec`
- `acceptance_rate`
- `avg_tokens_per_step`
- `proposed`, `accepted`, `rejections`
- `judge_accept_rate` (AutoJudge only)
- `target_fallback_rate` (AutoJudge only)
- `autojudge_train_samples`, `autojudge_val_auc`, `autojudge_val_recall` (AutoJudge only)
- `autojudge_threshold_calibrated`, `autojudge_threshold_used` (AutoJudge only)
- legacy aliases: `autojudge_threshold_selected` (= calibrated), `autojudge_threshold` (= used)
- `topk_accept_rate`, `topk_rank_effective`, `topk_mismatches`, `topk_accepted_mismatches` (Top-K only)
- `branch_prune_rate` (SpecExec only)
- `effective_parallelism` (SpecExec only)
- `target_calls_per_token` (AutoJudge, Top-K, and SpecExec)
- `draft_calls_per_token` (SpecExec only)
- `cache_hit_rate` (SpecExec only)
- `max_active_branches` (SpecExec only)
- `gsm8k_exact_match`, `gsm8k_correct`, `gsm8k_total` (when `--eval-task gsm8k`)
- `error_type`, `error_message`, `traceback` (for failed runs)
- System metadata: `git_sha`, `hostname`, `gpu_name`, `gpu_driver`, `cuda_runtime`, `torch_version`, `transformers_version`, `display_active`
- Validate output schema with:
```bash
python scripts/validate_results_jsonl.py --path datasets/results.jsonl --strict
```

**Project Layout**
- `sp_samp/`: core library, AutoJudge, SpecExec, HF adapter.
- `benchmarks/`: benchmark runner.
- `configs/`: preset configs.
- `tests/`: tests.
- `Dockerfile`, `Dockerfile.gpu`: containers.

**Notes**
- SpecExec is implemented as an exact (distribution-preserving) decoder with speculative cache prefill and configurable `parallel_branches` / `branch_prune_threshold`.
- HF SpecExec uses KV-cache reuse and depth-wise tree passes for faster cache construction.
- Draft and target models must share an identical tokenizer vocabulary mapping for speculative, AutoJudge, Top-K, and SpecExec correctness.
- AutoJudge training in this implementation is `gsm8k`-specific. If `--autojudge-checkpoint` is absent, provide a GSM8K-compatible train dataset (`question` + `answer`) via `--autojudge-train-dataset` or `--dataset`.
- AutoJudge now fails fast with an actionable message when the train dataset path is missing or not GSM8K-compatible (for example, MT-Bench JSONL is not valid for judge training).
- `scripts/install_dependencies.sh` is idempotent for apt/pip dependencies and never modifies NVIDIA driver packages.
- `scripts/install_dependencies.sh` prefers `python3.11` when available and warns when running with lower versions.
- Ubuntu 17 is EOL. Script blocks by default on EOL Ubuntu unless `--allow-eol-ubuntu` is set explicitly.
- Re-running a benchmark with the same `OUT` file automatically skips completed runs (resume mode).
- Failed runs are written to JSONL and do not stop the whole benchmark method loop.
- `make validate-configs` checks config references and tokenizer compatibility in `configs/*.json`.
- `make validate-results` enforces JSONL schema compatibility for downstream analysis.
- Current Make defaults are paper-aligned for compatible open models: `Qwen2.5-0.5B-Instruct` draft -> `Qwen2.5-3B-Instruct` target.
- `Qwen2.5-0.5B -> Qwen2.5-7B` is kept as legacy preset pair, but it fails strict speculative methods because model vocab sizes differ.

**Docker Troubleshooting**
- Error `parent snapshot ... does not exist` during `docker build` is typically a Docker BuildKit cache/snapshot issue on host.
- Recovery sequence:
```bash
make docker-prune-builder
make docker-build-gpu-safe
```
- `docker-build-gpu-safe` first tries BuildKit and then automatically retries with legacy builder (`DOCKER_BUILDKIT=0`) if BuildKit fails.
- Error `RuntimeError: No CUDA GPUs are available` inside container means GPU runtime passthrough is not working (host-side setup), not model code.
- Error `Could not find the bitsandbytes CUDA binary` or `No module named 'triton.ops'` during GPT-OSS load indicates bitsandbytes/triton mismatch in container runtime path.
- Quick diagnostics:
```bash
make docker-gpu-check
make docker-gpu-check-image
```
- If `docker-gpu-check` fails, (re)configure `nvidia-container-toolkit` and restart Docker daemon.
- For deterministic progress while investigating GPT-OSS runtime issues, run the short load fallback preset `mistral_target_mistral_draft_k4`.
