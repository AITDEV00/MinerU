# Default image name if not provided
NAME := adeo-icarus-mineru
USERNAME := jya0

# Get today's date in dd-mm-yyyy format
DATE := $(shell date +%d-%m-%Y)

# Docker image tag with date
IMAGE_TAG := $(NAME):$(DATE)

.PHONY: all docker_build

all: docker_build

docker_build:
	@echo "Building docker image: $(IMAGE_TAG)"
	@docker build ./global -t $(IMAGE_TAG)

docker_upload:
	@echo "Uploading docker image: $(IMAGE_TAG) to Docker Hub"
	@docker tag $(IMAGE_TAG) $(USERNAME)/$(IMAGE_TAG)
	@docker push $(USERNAME)/$(IMAGE_TAG)


# docker tag ray-jina-colbert-v2:rc-09102025 jya0/ray-jina-colbert-v2:rc-09102025
# docker save -o /home/jyao/Documents/ray-jina-colbert-v2:rc-09102025.tar ray-jina-colbert-v2:rc-09102025ray-jina-colbert-v2:rc-09102025