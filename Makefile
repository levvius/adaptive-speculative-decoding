VENV_PYTHON ?= .venv/bin/python
PYTHON ?= $(if $(wildcard $(VENV_PYTHON)),$(VENV_PYTHON),python3)
CONFIG_DIR ?= configs
DATASET ?= datasets/mt_bench.jsonl
OUT ?= datasets/results.jsonl
RESULTS ?= $(OUT)
METHOD ?= all
EXPERIMENT ?= qwen25_3b_target_qwen25_0p5b_all_methods
AUTOJUDGE_EXPERIMENT ?= qwen25_3b_target_qwen25_0p5b_autojudge_k4
SPECEXEC_EXPERIMENT ?= qwen25_3b_target_qwen25_0p5b_specexec_k4
TARGET_PRESET ?= qwen25_3b_instruct
DRAFT_PRESET ?= qwen25_0p5b_instruct_compat
SMOKE_HF_MODEL ?= sshleifer/tiny-gpt2
SMOKE_HF_TOKENIZER ?= sshleifer/tiny-gpt2
SMOKE_HF_DEVICE ?= cpu

IMAGE_CPU ?= sp-samp
IMAGE_GPU ?= sp-samp-gpu
DOCKER_GPU_ARGS ?= --gpus all
CUDA_BASE_IMAGE ?= nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04
TORCH_INDEX_URL ?= https://download.pytorch.org/whl/cu128
TORCH_VERSION ?= 2.9.1
DOCKER_CMD ?= docker
HEADLESS ?= 0
HEADLESS_ARG := $(if $(filter 1 true yes,$(HEADLESS)),--require-headless,)
PAPER_DATE ?= $(shell date +%F)
PAPER_RAW ?= datasets/results_autojudge_qwen25_paper_$(PAPER_DATE).jsonl
PAPER_REPORT_PREFIX ?= reports/autojudge_qwen25_paper_$(PAPER_DATE)
PAPER_MANIFEST ?= reports/autojudge_run_manifest_$(PAPER_DATE).json

LOCAL_EVAL_DATE ?= $(shell date +%F)
LOCAL_REPORT_PREFIX ?= reports/yandex_local_7b_1p5b_$(LOCAL_EVAL_DATE)
LOCAL_MANIFEST ?= reports/local_7b_1p5b_run_manifest_$(LOCAL_EVAL_DATE).json

DATA_DIR ?= $(abspath $(dir $(DATASET)))
DATASET_IN_CONTAINER ?= /data/$(notdir $(DATASET))
OUT_IN_CONTAINER ?= /data/$(notdir $(OUT))
ALLOW_EOL_UBUNTU ?= 0
ALLOW_EOL_ARG := $(if $(filter 1 true yes,$(ALLOW_EOL_UBUNTU)),--allow-eol-ubuntu,)

.PHONY: help setup setup-gpu check validate-configs validate-results list-presets test bench-toy smoke-hf smoke-hf-gpu bench bench-method autojudge specexec bench-all paper-eval local-eval \
		docker-build docker-build-gpu docker-build-gpu-safe docker-prune-builder docker-gpu-check docker-gpu-check-image docker-test docker-bench docker-autojudge docker-specexec docker-bench-all

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Install/upgrade required system + Python deps (safe: does not touch NVIDIA driver)
	bash scripts/install_dependencies.sh $(ALLOW_EOL_ARG)

setup-gpu: ## Install/upgrade required deps including GPU Python extras (bitsandbytes/accelerate)
	bash scripts/install_dependencies.sh --gpu $(ALLOW_EOL_ARG)

check: ## Run syntax checks (compileall)
	$(PYTHON) -m compileall sp_samp benchmarks tests
	$(PYTHON) scripts/validate_configs.py --config-dir $(CONFIG_DIR)

validate-configs: ## Validate model/method/experiment preset consistency
	$(PYTHON) scripts/validate_configs.py --config-dir $(CONFIG_DIR)

validate-results: ## Validate benchmark JSONL schema (default: RESULTS=$(OUT))
	@if [ ! -f "$(RESULTS)" ]; then echo "Results file not found: $(RESULTS). Default is RESULTS=datasets/results.jsonl."; exit 2; fi
	$(PYTHON) scripts/validate_results_jsonl.py --path $(RESULTS) --strict

list-presets: ## List models/methods/experiments presets
	$(PYTHON) -m sp_samp.cli list-presets --config-dir $(CONFIG_DIR)

test: ## Run pytest locally
	@$(PYTHON) -c "import pytest" >/dev/null 2>&1 || { \
		echo "pytest is not installed for $(PYTHON). Run 'make setup' first."; \
		exit 2; \
	}
	$(PYTHON) -m pytest -q

bench-toy: ## Run toy benchmark without HF models
	$(PYTHON) -m benchmarks.bench_speculative \
		--method both \
		--runs 1 \
		--max-samples 5 \
		--max-new-tokens 32 \
		--vocab-size 2048 \
		--out $(OUT)

smoke-hf: ## Quick HF smoke run (requires torch+transformers and internet)
	$(PYTHON) -m benchmarks.bench_speculative \
		--method all \
		--hf-model $(SMOKE_HF_MODEL) \
		--hf-draft-model $(SMOKE_HF_MODEL) \
		--tokenizer $(SMOKE_HF_TOKENIZER) \
		--draft-tokenizer $(SMOKE_HF_TOKENIZER) \
		--device $(SMOKE_HF_DEVICE) \
		--runs 1 \
		--max-samples 2 \
		--max-new-tokens 8 \
		--k 2 \
		--out $(OUT)

smoke-hf-gpu: ## Quick HF smoke run on GPU (tiny model)
	$(MAKE) smoke-hf SMOKE_HF_DEVICE=cuda OUT=$(OUT)

bench: ## Run benchmark using EXPERIMENT preset (requires DATASET)
	@if [ ! -f "$(DATASET)" ]; then echo "Dataset not found: $(DATASET). Expected default datasets/mt_bench.jsonl or override DATASET=/absolute/path/to/mt_bench.jsonl"; exit 2; fi
	$(PYTHON) -m sp_samp.cli bench \
		--config-dir $(CONFIG_DIR) \
		--experiment $(EXPERIMENT) \
		--dataset $(DATASET) \
		$(HEADLESS_ARG) \
		--out $(OUT)

bench-method: ## Run benchmark with explicit METHOD using model presets (requires DATASET)
	@if [ ! -f "$(DATASET)" ]; then echo "Dataset not found: $(DATASET). Expected default datasets/mt_bench.jsonl or override DATASET=/absolute/path/to/mt_bench.jsonl"; exit 2; fi
	$(PYTHON) -m sp_samp.cli bench \
		--config-dir $(CONFIG_DIR) \
		--model-preset $(TARGET_PRESET) \
		--draft-preset $(DRAFT_PRESET) \
		--method $(METHOD) \
		--dataset $(DATASET) \
		$(HEADLESS_ARG) \
		--out $(OUT)

autojudge: ## Run AutoJudge preset (requires DATASET)
	@if [ ! -f "$(DATASET)" ]; then echo "Dataset not found: $(DATASET). Expected default datasets/mt_bench.jsonl or override DATASET=/absolute/path/to/mt_bench.jsonl"; exit 2; fi
	$(PYTHON) -m sp_samp.cli autojudge \
		--config-dir $(CONFIG_DIR) \
		--experiment $(AUTOJUDGE_EXPERIMENT) \
		--dataset $(DATASET) \
		$(HEADLESS_ARG) \
		--out $(OUT)

specexec: ## Run SpecExec preset (requires DATASET)
	@if [ ! -f "$(DATASET)" ]; then echo "Dataset not found: $(DATASET). Expected default datasets/mt_bench.jsonl or override DATASET=/absolute/path/to/mt_bench.jsonl"; exit 2; fi
	$(PYTHON) -m sp_samp.cli specexec \
		--config-dir $(CONFIG_DIR) \
		--experiment $(SPECEXEC_EXPERIMENT) \
		--dataset $(DATASET) \
		$(HEADLESS_ARG) \
		--out $(OUT)

bench-all: ## Run baseline+speculative+autojudge+topk+specexec in one call (requires DATASET)
	@if [ ! -f "$(DATASET)" ]; then echo "Dataset not found: $(DATASET). Expected default datasets/mt_bench.jsonl or override DATASET=/absolute/path/to/mt_bench.jsonl"; exit 2; fi
	$(PYTHON) -m sp_samp.cli bench \
		--config-dir $(CONFIG_DIR) \
		--experiment $(EXPERIMENT) \
		--method all \
		--dataset $(DATASET) \
		$(HEADLESS_ARG) \
		--out $(OUT)

paper-eval: ## Run paper-style GSM8K sweep (Qwen2.5 0.5B -> 3B) and build reports
	PYTHON_BIN="$(PYTHON)" OUT_RAW="$(PAPER_RAW)" REPORT_PREFIX="$(PAPER_REPORT_PREFIX)" MANIFEST_PATH="$(PAPER_MANIFEST)" scripts/run_autojudge_paper_eval.sh

local-eval: ## Run local Qwen2.5 7B/1.5B eval (GSM8K + LiveCodeBench) and build Yandex-style reports
	PYTHON_BIN="$(PYTHON)" REPORT_PREFIX="$(LOCAL_REPORT_PREFIX)" MANIFEST_PATH="$(LOCAL_MANIFEST)" scripts/run_local_7b_1p5b_eval.sh

docker-build: ## Build CPU Docker image
	$(DOCKER_CMD) build -t $(IMAGE_CPU) .

docker-build-gpu: ## Build GPU Docker image (default: CUDA 12.8 + torch cu128)
	$(DOCKER_CMD) build -f Dockerfile.gpu \
		--build-arg BASE_IMAGE=$(CUDA_BASE_IMAGE) \
		--build-arg TORCH_INDEX_URL=$(TORCH_INDEX_URL) \
		--build-arg TORCH_VERSION=$(TORCH_VERSION) \
		-t $(IMAGE_GPU) .

docker-build-gpu-safe: ## Build GPU image with fallback to legacy builder if BuildKit snapshot export fails
	$(DOCKER_CMD) build -f Dockerfile.gpu \
		--build-arg BASE_IMAGE=$(CUDA_BASE_IMAGE) \
		--build-arg TORCH_INDEX_URL=$(TORCH_INDEX_URL) \
		--build-arg TORCH_VERSION=$(TORCH_VERSION) \
		-t $(IMAGE_GPU) . || \
	(echo "[WARN] BuildKit build failed; retrying with legacy builder (DOCKER_BUILDKIT=0)."; \
	DOCKER_BUILDKIT=0 $(DOCKER_CMD) build -f Dockerfile.gpu \
		--build-arg BASE_IMAGE=$(CUDA_BASE_IMAGE) \
		--build-arg TORCH_INDEX_URL=$(TORCH_INDEX_URL) \
		--build-arg TORCH_VERSION=$(TORCH_VERSION) \
		-t $(IMAGE_GPU) .)

docker-prune-builder: ## Cleanup Docker builder cache (use when snapshot/export errors occur)
	$(DOCKER_CMD) builder prune -af

docker-gpu-check: ## Verify GPU passthrough in a clean NVIDIA CUDA container
	@set -e; \
	if $(DOCKER_CMD) run --rm --gpus all $(CUDA_BASE_IMAGE) nvidia-smi; then \
		echo "[OK] nvidia-smi check passed in $(CUDA_BASE_IMAGE)."; \
	else \
		echo "[WARN] nvidia-smi check failed (NVML path). Falling back to torch CUDA check in $(IMAGE_GPU)."; \
		if ! $(DOCKER_CMD) image inspect $(IMAGE_GPU) >/dev/null 2>&1; then \
			echo "[ERROR] Fallback image '$(IMAGE_GPU)' not found. Build it first: make docker-build-gpu-safe"; \
			exit 2; \
		fi; \
		$(DOCKER_CMD) run --rm --gpus all --entrypoint python $(IMAGE_GPU) -c "import sys, torch; ok=torch.cuda.is_available() and torch.cuda.device_count()>0; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'count', torch.cuda.device_count()); sys.exit(0 if ok else 1)"; \
	fi

docker-gpu-check-image: ## Verify torch CUDA visibility inside the built GPU image
	$(DOCKER_CMD) run --rm --gpus all --entrypoint python $(IMAGE_GPU) -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), 'count', torch.cuda.device_count())"

docker-test: ## Run tests in CPU Docker image
	$(DOCKER_CMD) run --rm $(IMAGE_CPU)

docker-bench: ## Run benchmark in GPU Docker using EXPERIMENT preset (requires DATASET)
	@if [ ! -f "$(DATASET)" ]; then echo "Dataset not found: $(DATASET). Expected default datasets/mt_bench.jsonl or override DATASET=/absolute/path/to/mt_bench.jsonl"; exit 2; fi
	$(DOCKER_CMD) run --rm $(DOCKER_GPU_ARGS) -v $(DATA_DIR):/data $(IMAGE_GPU) \
		python -m sp_samp.cli bench \
		--config-dir $(CONFIG_DIR) \
		--experiment $(EXPERIMENT) \
		--dataset $(DATASET_IN_CONTAINER) \
		$(HEADLESS_ARG) \
		--out $(OUT_IN_CONTAINER)

docker-autojudge: ## Run AutoJudge in GPU Docker (requires DATASET)
	@if [ ! -f "$(DATASET)" ]; then echo "Dataset not found: $(DATASET). Expected default datasets/mt_bench.jsonl or override DATASET=/absolute/path/to/mt_bench.jsonl"; exit 2; fi
	$(DOCKER_CMD) run --rm $(DOCKER_GPU_ARGS) -v $(DATA_DIR):/data $(IMAGE_GPU) \
		python -m sp_samp.cli autojudge \
		--config-dir $(CONFIG_DIR) \
		--experiment $(AUTOJUDGE_EXPERIMENT) \
		--dataset $(DATASET_IN_CONTAINER) \
		$(HEADLESS_ARG) \
		--out $(OUT_IN_CONTAINER)

docker-specexec: ## Run SpecExec in GPU Docker (requires DATASET)
	@if [ ! -f "$(DATASET)" ]; then echo "Dataset not found: $(DATASET). Expected default datasets/mt_bench.jsonl or override DATASET=/absolute/path/to/mt_bench.jsonl"; exit 2; fi
	$(DOCKER_CMD) run --rm $(DOCKER_GPU_ARGS) -v $(DATA_DIR):/data $(IMAGE_GPU) \
		python -m sp_samp.cli specexec \
		--config-dir $(CONFIG_DIR) \
		--experiment $(SPECEXEC_EXPERIMENT) \
		--dataset $(DATASET_IN_CONTAINER) \
		$(HEADLESS_ARG) \
		--out $(OUT_IN_CONTAINER)

docker-bench-all: ## Run all methods (baseline+speculative+autojudge+topk+specexec) in GPU Docker (requires DATASET)
	@if [ ! -f "$(DATASET)" ]; then echo "Dataset not found: $(DATASET). Expected default datasets/mt_bench.jsonl or override DATASET=/absolute/path/to/mt_bench.jsonl"; exit 2; fi
	$(DOCKER_CMD) run --rm $(DOCKER_GPU_ARGS) -v $(DATA_DIR):/data $(IMAGE_GPU) \
		python -m sp_samp.cli bench \
		--config-dir $(CONFIG_DIR) \
		--experiment $(EXPERIMENT) \
		--method all \
		--dataset $(DATASET_IN_CONTAINER) \
		$(HEADLESS_ARG) \
		--out $(OUT_IN_CONTAINER)
