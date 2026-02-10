PYTHON ?= python
CONFIG_DIR ?= configs
DATASET ?=
OUT ?= benchmarks/results.jsonl
METHOD ?= all
EXPERIMENT ?= llama3_all_methods
AUTOJUDGE_EXPERIMENT ?= llama3_target_llama3_autojudge_k4
SPECEXEC_EXPERIMENT ?= llama3_target_llama3_specexec_k4
TARGET_PRESET ?= llama3_8b_instruct
DRAFT_PRESET ?= llama3_8b_instruct
SMOKE_HF_MODEL ?= sshleifer/tiny-gpt2
SMOKE_HF_TOKENIZER ?= sshleifer/tiny-gpt2

IMAGE_CPU ?= sp-samp
IMAGE_GPU ?= sp-samp-gpu
DOCKER_GPU_ARGS ?= --gpus all
HEADLESS ?= 0
HEADLESS_ARG := $(if $(filter 1 true yes,$(HEADLESS)),--require-headless,)

DATA_DIR ?= $(if $(DATASET),$(shell dirname "$(DATASET)"),/tmp)
DATASET_IN_CONTAINER ?= $(if $(DATASET),/data/$(notdir $(DATASET)),)
OUT_IN_CONTAINER ?= /data/$(notdir $(OUT))

.PHONY: help check validate-configs list-presets test bench-toy smoke-hf bench bench-method autojudge specexec bench-all \
	docker-build docker-build-gpu docker-test docker-bench docker-autojudge docker-specexec docker-bench-all

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

check: ## Run syntax checks (compileall)
	$(PYTHON) -m compileall sp_samp benchmarks tests
	$(PYTHON) scripts/validate_configs.py --config-dir $(CONFIG_DIR)

validate-configs: ## Validate model/method/experiment preset consistency
	$(PYTHON) scripts/validate_configs.py --config-dir $(CONFIG_DIR)

list-presets: ## List models/methods/experiments presets
	$(PYTHON) -m sp_samp.cli list-presets --config-dir $(CONFIG_DIR)

test: ## Run pytest locally
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
		--device cpu \
		--runs 1 \
		--max-samples 2 \
		--max-new-tokens 8 \
		--k 2 \
		--out $(OUT)

bench: ## Run benchmark using EXPERIMENT preset (requires DATASET)
	@if [ -z "$(DATASET)" ]; then echo "Set DATASET=/path/to/mt_bench.jsonl"; exit 2; fi
	$(PYTHON) -m sp_samp.cli bench \
		--config-dir $(CONFIG_DIR) \
		--experiment $(EXPERIMENT) \
		--dataset $(DATASET) \
		$(HEADLESS_ARG) \
		--out $(OUT)

bench-method: ## Run benchmark with explicit METHOD using model presets (requires DATASET)
	@if [ -z "$(DATASET)" ]; then echo "Set DATASET=/path/to/mt_bench.jsonl"; exit 2; fi
	$(PYTHON) -m sp_samp.cli bench \
		--config-dir $(CONFIG_DIR) \
		--model-preset $(TARGET_PRESET) \
		--draft-preset $(DRAFT_PRESET) \
		--method $(METHOD) \
		--dataset $(DATASET) \
		$(HEADLESS_ARG) \
		--out $(OUT)

autojudge: ## Run AutoJudge preset (requires DATASET)
	@if [ -z "$(DATASET)" ]; then echo "Set DATASET=/path/to/mt_bench.jsonl"; exit 2; fi
	$(PYTHON) -m sp_samp.cli autojudge \
		--config-dir $(CONFIG_DIR) \
		--experiment $(AUTOJUDGE_EXPERIMENT) \
		--dataset $(DATASET) \
		$(HEADLESS_ARG) \
		--out $(OUT)

specexec: ## Run SpecExec preset (requires DATASET)
	@if [ -z "$(DATASET)" ]; then echo "Set DATASET=/path/to/mt_bench.jsonl"; exit 2; fi
	$(PYTHON) -m sp_samp.cli specexec \
		--config-dir $(CONFIG_DIR) \
		--experiment $(SPECEXEC_EXPERIMENT) \
		--dataset $(DATASET) \
		$(HEADLESS_ARG) \
		--out $(OUT)

bench-all: ## Run baseline+speculative+autojudge+specexec in one call (requires DATASET)
	@if [ -z "$(DATASET)" ]; then echo "Set DATASET=/path/to/mt_bench.jsonl"; exit 2; fi
	$(PYTHON) -m sp_samp.cli bench \
		--config-dir $(CONFIG_DIR) \
		--experiment $(EXPERIMENT) \
		--method all \
		--dataset $(DATASET) \
		$(HEADLESS_ARG) \
		--out $(OUT)

docker-build: ## Build CPU Docker image
	docker build -t $(IMAGE_CPU) .

docker-build-gpu: ## Build GPU Docker image (CUDA 12.4 example)
	docker build -f Dockerfile.gpu \
		--build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 \
		--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
		--build-arg TORCH_VERSION=2.3.1 \
		-t $(IMAGE_GPU) .

docker-test: ## Run tests in CPU Docker image
	docker run --rm $(IMAGE_CPU)

docker-bench: ## Run benchmark in GPU Docker using EXPERIMENT preset (requires DATASET)
	@if [ -z "$(DATASET)" ]; then echo "Set DATASET=/path/to/mt_bench.jsonl"; exit 2; fi
	docker run --rm $(DOCKER_GPU_ARGS) -v $(DATA_DIR):/data $(IMAGE_GPU) \
		python -m sp_samp.cli bench \
		--config-dir $(CONFIG_DIR) \
		--experiment $(EXPERIMENT) \
		--dataset $(DATASET_IN_CONTAINER) \
		$(HEADLESS_ARG) \
		--out $(OUT_IN_CONTAINER)

docker-autojudge: ## Run AutoJudge in GPU Docker (requires DATASET)
	@if [ -z "$(DATASET)" ]; then echo "Set DATASET=/path/to/mt_bench.jsonl"; exit 2; fi
	docker run --rm $(DOCKER_GPU_ARGS) -v $(DATA_DIR):/data $(IMAGE_GPU) \
		python -m sp_samp.cli autojudge \
		--config-dir $(CONFIG_DIR) \
		--experiment $(AUTOJUDGE_EXPERIMENT) \
		--dataset $(DATASET_IN_CONTAINER) \
		$(HEADLESS_ARG) \
		--out $(OUT_IN_CONTAINER)

docker-specexec: ## Run SpecExec in GPU Docker (requires DATASET)
	@if [ -z "$(DATASET)" ]; then echo "Set DATASET=/path/to/mt_bench.jsonl"; exit 2; fi
	docker run --rm $(DOCKER_GPU_ARGS) -v $(DATA_DIR):/data $(IMAGE_GPU) \
		python -m sp_samp.cli specexec \
		--config-dir $(CONFIG_DIR) \
		--experiment $(SPECEXEC_EXPERIMENT) \
		--dataset $(DATASET_IN_CONTAINER) \
		$(HEADLESS_ARG) \
		--out $(OUT_IN_CONTAINER)

docker-bench-all: ## Run all methods (baseline+speculative+autojudge+specexec) in GPU Docker (requires DATASET)
	@if [ -z "$(DATASET)" ]; then echo "Set DATASET=/path/to/mt_bench.jsonl"; exit 2; fi
	docker run --rm $(DOCKER_GPU_ARGS) -v $(DATA_DIR):/data $(IMAGE_GPU) \
		python -m sp_samp.cli bench \
		--config-dir $(CONFIG_DIR) \
		--experiment $(EXPERIMENT) \
		--method all \
		--dataset $(DATASET_IN_CONTAINER) \
		$(HEADLESS_ARG) \
		--out $(OUT_IN_CONTAINER)
