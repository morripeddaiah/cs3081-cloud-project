.PHONY: help up down train test lint eval clean

SHELL := /bin/bash
CLUSTER_NAME := cloud-project
K3D_CONFIG := cluster/k3d.yaml
SUBSET ?= 1500

help:
	@echo "Cloud Computing Mini Project - Command Parser"
	@echo "Available commands:"
	@echo "  make up               Create k3d cluster and deploy all services (<= 3 commands)"
	@echo "  make down             Delete cluster and teardown local resources"
	@echo "  make train            Fine-tune student model on full dataset (1500 samples)"
	@echo "  make train SUBSET=50  Fine-tune student model on subset (50 samples)"
	@echo "  make test             Run test suite"
	@echo "  make lint             Run lint checks"
	@echo "  make eval             Evaluate student model on validation dataset"

up:
	# 1. Create cluster from configuration
	k3d cluster create --config $(K3D_CONFIG) || true
	# 2. Apply Kubernetes namespaces and infrastructure (SeaweedFS + secrets)
	kubectl apply -f k8s/namespaces.yaml && kubectl apply -f k8s/s3-secret.yaml && kubectl apply -f k8s/seaweedfs.yaml
	# 3. Apply application deployment, service, ingress, and HPA
	kubectl apply -f k8s/deployment.yaml && kubectl apply -f k8s/service.yaml && kubectl apply -f k8s/ingress.yaml && kubectl apply -f k8s/hpa.yaml

down:
	# Delete k3d cluster and all resources
	k3d cluster delete $(CLUSTER_NAME) || true

train:
	python training/train.py --config training/config.yaml --subset $(SUBSET) --out-dir models/v1

eval:
	python training/evaluate.py --mode student --model-path models/v1 --gold data/validation.jsonl --out-report results/student_v1_validation.json --min-validity 95

test:
	python -m unittest discover -s tests -p "test_*.py" -v

lint:
	python -m py_compile worker/*.py service/*.py training/*.py tests/*.py
	@echo "All python files compile cleanly."

clean:
	rm -rf shards/*.jsonl datasets/v1/*.jsonl models/v1/* results/predictions.jsonl
