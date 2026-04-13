# MinerU Docker - Build, push, and run (all in this project)
# Fork: https://github.com/AITDEV00/MinerU | Upstream: https://github.com/opendatalab/MinerU

NAME := adeo-mineru-api
DATE := $(shell date +%d-%m-%Y)
# Split-VLM image tag (default: same as DATE); override e.g. make push-split-vlm-only SPLIT_TAG=01-04-2026
SPLIT_TAG ?= $(DATE)
REGISTRY := zdc-ai-harbor.ecouncil.ae
REPO := $(REGISTRY)/aiteam/$(NAME)
RUNTIME := $(shell command -v podman 2>/dev/null || echo docker)
# podman needs --tls-verify=false for self-signed Harbor; docker ignores
PUSH_OPTS := $(if $(findstring podman,$(RUNTIME)),--tls-verify=false,)
COMPOSE_SPLIT := -f docker-compose.api-split-vlm.yml
COMPOSE_SPLIT_ENV := MINERU_SPLIT_TAG=$(SPLIT_TAG)

.PHONY: help build push login build-split-vlm push-split-vlm push-split-vlm-only run-split-vlm run-split-vlm-build run-split-vlm-down

help:
	@echo "MinerU Docker (AITDEV00 fork)"
	@echo ""
	@echo "  make build              Build API image (CPU, split VLM client)"
	@echo "  make login              Login to Harbor"
	@echo "  make push               Push to Harbor ($(REPO))"
	@echo ""
	@echo "Split VLM (parse_method=vlm, layout=MinerU2.5 + extraction=Qwen3-VL):"
	@echo "  make build-split-vlm    Build split VLM API image"
	@echo "  make push-split-vlm     Build + push split VLM image to Harbor ($(REPO))"
	@echo "  make push-split-vlm-only  Push existing image (tag SPLIT_TAG, default $(DATE); no rebuild)"
	@echo "  make run-split-vlm      Start API on port 8085"
	@echo "  make run-split-vlm-build  Build + start"
	@echo "  make run-split-vlm-down   Stop API"
	@echo ""

build:
	@echo "Building $(NAME):$(DATE)..."
	$(RUNTIME) build -f Dockerfile.api -t $(NAME):$(DATE) -t $(NAME):latest ..

login:
	$(RUNTIME) login --tls-verify=false $(REGISTRY)

push: build
	@echo "Pushing to $(REPO)..."
	$(RUNTIME) tag $(NAME):$(DATE) $(REPO):$(DATE)
	$(RUNTIME) tag $(NAME):$(DATE) $(REPO):latest
	$(RUNTIME) push $(REPO):$(DATE)
	$(RUNTIME) push $(REPO):latest
	@echo "Pushed: $(REPO):$(DATE) $(REPO):latest"

build-split-vlm:
	@echo "Building $(NAME):$(SPLIT_TAG) (+ :latest)..."
	$(RUNTIME) build -f Dockerfile.api-split-vlm -t $(NAME):$(SPLIT_TAG) -t $(NAME):latest ..

push-split-vlm: build-split-vlm
	@echo "Pushing $(NAME):$(SPLIT_TAG) to $(REPO)..."
	$(RUNTIME) tag $(NAME):$(SPLIT_TAG) $(REPO):$(SPLIT_TAG)
	$(RUNTIME) tag $(NAME):$(SPLIT_TAG) $(REPO):latest
	$(RUNTIME) push $(PUSH_OPTS) $(REPO):$(SPLIT_TAG)
	$(RUNTIME) push $(PUSH_OPTS) $(REPO):latest
	@echo "Pushed: $(REPO):$(SPLIT_TAG) $(REPO):latest"

# Use after a successful build when push failed (e.g. forgot Harbor login). Does not run build-split-vlm.
push-split-vlm-only:
	@echo "Pushing existing $(NAME):$(SPLIT_TAG) to $(REPO) (no rebuild)..."
	$(RUNTIME) tag $(NAME):$(SPLIT_TAG) $(REPO):$(SPLIT_TAG)
	$(RUNTIME) tag $(NAME):$(SPLIT_TAG) $(REPO):latest
	$(RUNTIME) push $(PUSH_OPTS) $(REPO):$(SPLIT_TAG)
	$(RUNTIME) push $(PUSH_OPTS) $(REPO):latest
	@echo "Pushed: $(REPO):$(SPLIT_TAG) $(REPO):latest"

run-split-vlm:
	$(COMPOSE_SPLIT_ENV) $(RUNTIME) compose $(COMPOSE_SPLIT) up -d
	@echo "API: http://localhost:8085/docs"

run-split-vlm-build:
	$(COMPOSE_SPLIT_ENV) $(RUNTIME) compose $(COMPOSE_SPLIT) up -d --build
	@echo "API: http://localhost:8085/docs"

run-split-vlm-down:
	$(RUNTIME) compose $(COMPOSE_SPLIT) down
