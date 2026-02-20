# MinerU Docker - Build, push, and run (all in this project)
# Fork: https://github.com/AITDEV00/MinerU | Upstream: https://github.com/opendatalab/MinerU

NAME := adeo-mineru-api
DATE := $(shell date +%d-%m-%Y)
REGISTRY := zdc-ai-harbor.ecouncil.ae
REPO := $(REGISTRY)/aiteam/$(NAME)
RUNTIME := $(shell command -v podman 2>/dev/null || echo docker)
# podman needs --tls-verify=false for self-signed Harbor; docker ignores
PUSH_OPTS := $(if $(findstring podman,$(RUNTIME)),--tls-verify=false,)
COMPOSE_SPLIT := -f docker-compose.api-split-vlm.yml

.PHONY: help build push login build-split-vlm push-split-vlm run-split-vlm run-split-vlm-build run-split-vlm-down

help:
	@echo "MinerU Docker (AITDEV00 fork)"
	@echo ""
	@echo "  make build              Build API image (CPU, split VLM client)"
	@echo "  make login              Login to Harbor"
	@echo "  make push               Push to Harbor ($(REPO))"
	@echo ""
	@echo "Split VLM (parse_method=vlm, layout=MinerU2.5 + extraction=Qwen3-VL):"
	@echo "  make build-split-vlm    Build split VLM API image"
	@echo "  make push-split-vlm     Push split VLM image to Harbor ($(REPO))"
	@echo "  make run-split-vlm      Start API on port 8085"
	@echo "  make run-split-vlm-build  Build + start"
	@echo "  make run-split-vlm-down   Stop API"
	@echo ""

build:
	@echo "Building $(NAME):$(DATE)..."
	$(RUNTIME) build -f Dockerfile.api -t $(NAME):$(DATE) -t $(NAME):latest ..

login:
	$(RUNTIME) login $(REGISTRY)

push: build
	@echo "Pushing to $(REPO)..."
	$(RUNTIME) tag $(NAME):$(DATE) $(REPO):$(DATE)
	$(RUNTIME) tag $(NAME):$(DATE) $(REPO):latest
	$(RUNTIME) push $(REPO):$(DATE)
	$(RUNTIME) push $(REPO):latest
	@echo "Pushed: $(REPO):$(DATE) $(REPO):latest"

build-split-vlm:
	@echo "Building $(NAME):split-vlm..."
	$(RUNTIME) build -f Dockerfile.api-split-vlm -t $(NAME):split-vlm ..

push-split-vlm: build-split-vlm
	@echo "Pushing $(NAME):split-vlm to $(REPO)..."
	$(RUNTIME) tag $(NAME):split-vlm $(REPO):split-vlm
	$(RUNTIME) tag $(NAME):split-vlm $(REPO):latest
	$(RUNTIME) push $(PUSH_OPTS) $(REPO):split-vlm
	$(RUNTIME) push $(PUSH_OPTS) $(REPO):latest
	@echo "Pushed: $(REPO):split-vlm $(REPO):latest"

run-split-vlm:
	$(RUNTIME) compose $(COMPOSE_SPLIT) up -d
	@echo "API: http://localhost:8085/docs"

run-split-vlm-build:
	$(RUNTIME) compose $(COMPOSE_SPLIT) up -d --build
	@echo "API: http://localhost:8085/docs"

run-split-vlm-down:
	$(RUNTIME) compose $(COMPOSE_SPLIT) down
