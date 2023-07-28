NAME=ml-kubeflow
TAG_PREFIX=neural-tpps
VERSION=$(shell git rev-parse HEAD)
REPO=babylon.jfrog.io/classic-dev-docker-virtual/babylonhealth
TAG=$(TAG_PREFIX)-$(VERSION)
IMAGE=$(REPO)/$(NAME):$(TAG)
IMAGE_BASE=$(REPO)/$(NAME):$(TAG_PREFIX)-base

.PHONY: build
build:
	docker build --file Dockerfile -t $(IMAGE) .

.PHONY: build-base
build-base:
	docker build --file Dockerfile-base -t $(IMAGE_BASE) .

.PHONY: publish
publish: check_clean build push update_deploy

.PHONY: check_clean
check_clean:
	git diff-index --quiet HEAD || (\
	echo "Working directory is not clean! Commit before publishing"; \
	exit 1;)

.PHONY: tag
tag:
	git tag $(TAG)
	git push --tags
	docker push $(IMAGE)

.PHONY: push
push:
	git push --tags
	docker push $(IMAGE)

.PHONY: push-base
push-base:
	git push --tags
	docker push $(IMAGE_BASE)

.PHONY: update_deploy
update_deploy:
	python deploy/update_deploy.py --image $(IMAGE)
