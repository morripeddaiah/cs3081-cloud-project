# Cloud Computing (CS3081) Mini Project
## Knowledge Distillation and Deployment of a Lightweight Command-Parsing Language Model on a Local Cloud

**Student Name:** Morri Peddaiah  
**Roll Number:** 230101051  
**GitHub Repository:** `https://github.com/morripeddaiah/cs3081-cloud-project`  
**Final Tag:** `final`  

---

## 1. Overview
This repository contains the complete implementation of the CS3081 Mini Project. The system compresses a teacher language model (`Phi-3.5-mini-instruct`, 3.8B parameters) into a lightweight student model (`EleutherAI/pythia-70m`, 70M parameters) using **Sequence-Level Knowledge Distillation** for natural language voice command parsing. The distilled model is deployed as a containerized FastAPI microservice on a local Kubernetes cluster (k3d), backed by SeaweedFS S3 object storage, monitored by health probes, autoscaled via HPA, and secured by stateless Network Access Control Lists (NACL).

---

## 2. Directory Structure

```text
project/
├── .github/workflows/
│   └── ci.yml               # GitHub Actions CI/CD pipeline (lint, test, quality gate, gitleaks)
├── AI_USE.md                # AI tools disclosure and provenance statement
├── Makefile                 # Unified project automation (up, down, train, test, lint, eval)
├── Dockerfile               # Multi-stage CPU-only container image (<2 GB)
├── README.md                # Project documentation and reproduction guide
├── cluster/
│   └── k3d.yaml             # k3d cluster configuration (1 server, port 8080:80)
├── data/
│   ├── commands_pool.jsonl  # 3,000 unlabelled SLURP commands
│   ├── rate_card.json       # Cost card for vCPU and memory hours
│   ├── schema.json          # 4-key JSON schema and slot extraction rules
│   └── validation.jsonl     # 100 gold-labelled validation samples
├── k8s/
│   ├── namespaces.yaml      # app, data, and pipeline namespaces
│   ├── s3-secret.yaml       # S3 credentials secret
│   ├── seaweedfs.yaml       # SeaweedFS object storage deployment & service
│   ├── deployment.yaml      # App deployment (2 replicas, CPU/mem limits, probes)
│   ├── service.yaml         # ClusterIP service for command parser
│   ├── ingress.yaml         # Traefik ingress configuration
│   ├── hpa.yaml             # Horizontal Pod Autoscaler (min 1, max 2, 70% CPU)
│   └── worker-job.yaml      # Parallel teacher worker batch jobs
├── network/
│   ├── nacl.json            # Stateless NACL rules for public subnet
│   ├── packets_sample.jsonl # Sample packets for verification
│   └── subnets.md           # 10.20.0.0/16 subnet sizing and derivation
├── prompts/
│   └── teacher_prompt.txt   # Prompt template for Phi-3.5 teacher
├── report/
│   └── report.md            # Comprehensive project report (all 7 tasks)
├── results/
│   ├── teacher_validation.json     # Phi-3.5 evaluation score
│   ├── untrained_validation.json   # Base Pythia-70m evaluation score
│   ├── student_v1_validation.json  # Distilled student v1 score
│   ├── student_v2_validation.json  # Distilled student v2 score
│   ├── loadtest_stats.csv          # Locust load test metrics (p50, p95, throughput)
│   └── loadtest_headers.json       # Header distribution (X-Served-By, X-Model-Version)
├── service/
│   ├── app.py               # FastAPI inference service (/parse, /healthz, /readyz)
│   └── requirements.txt     # Pinned runtime dependencies
├── tests/
│   ├── test_clean.py        # Tests for dataset cleaning and anti-hallucination
│   ├── test_parse.py        # Tests for service endpoints, probes, fault injection
│   ├── test_schema.py       # Tests for JSON schema validation
│   └── run_tests.py         # Test runner
├── tools/
│   ├── locustfile.py        # Locust load test suite
│   ├── nacl_sim.py          # Stateless NACL simulator
│   └── score.py             # Evaluation and quality gate scoring tool
├── training/
│   ├── config.yaml          # Student fine-tuning hyperparameters
│   ├── evaluate.py          # Model evaluation script against validation.jsonl
│   └── train.py             # Causal LM distillation training script
└── worker/
    ├── clean.py             # 5-stage shard cleaning and deduplication
    ├── requirements.txt     # Worker dependencies
    └── worker.py            # Parallel teacher labelling worker
```

---

## 3. Quick Start & Prerequisites

### Prerequisites
- Python 3.10+ (with `torch`, `transformers`, `fastapi`, `uvicorn`)
- Docker Desktop or Docker Engine
- `k3d` and `kubectl`
- Ollama with `phi3.5:latest` model pulled (`ollama pull phi3.5`)

### Setup and Lifecycle Commands

1. **Deploy Local Cloud Infrastructure ($\le 3$ commands)**:
   ```bash
   make up
   ```
   This creates the k3d cluster, provisions namespaces (`app`, `data`, `pipeline`), starts SeaweedFS object storage, and deploys the command parser application with 2 replicas and ingress on port 8080.

2. **Run Unit Tests**:
   ```bash
   make test
   ```
   Executes 15 unit tests covering endpoints, health probes, fault injection, schema enforcement, and dataset cleaning.

3. **Verify Stateless NACL Rules**:
   ```bash
   python tools/nacl_sim.py network/nacl.json network/packets_sample.jsonl --check
   ```
   Outputs 100% match across all test packets.

4. **Train Student Model**:
   - Full dataset (1,500 samples):
     ```bash
     make train
     ```
   - Quick subset test (50 samples):
     ```bash
     make train SUBSET=50
     ```

5. **Evaluate Model Quality Gate**:
   ```bash
   make eval
   ```
   Runs `tools/score.py` against `data/validation.jsonl`, enforcing `--min-validity 95`.

6. **Run Load Testing**:
   ```bash
   locust -f tools/locustfile.py --headless --users 4 --spawn-rate 4 --run-time 2m --host http://localhost:8080 --csv results/loadtest
   ```

7. **Teardown Local Cloud**:
   ```bash
   make down
   ```

---

## 4. Key Results Summary

- **Distillation Quality**: The 70M student model achieves **97.0% JSON validity** and **76.0% full frame accuracy** (compared to 12% validity for untrained Pythia-70m and 99% validity for the 3.8B teacher).
- **Latency & Throughput**: Under 4 concurrent Locust users, the service achieves **35.67 req/s**, median latency of **74.0 ms**, and p95 latency of **114.0 ms** with **0% errors**.
- **Serving Cost**: Running 2 replicas costs **INR 7.90 / hour**, resulting in an ultra-low serving cost of **INR 0.0615 per 1,000 requests**.
- **Cold Start**: Median cold start time of **4.39 seconds** with traffic protected by `/readyz` probes.
- **Load Balancing**: Traffic split is balanced across replicas (**51.05% vs 48.95%**), satisfying the $\ge 30\%$ requirement.

---

## 5. Submission Checklist
- [x] Complete source code in repository
- [x] Git release assets configured (models not committed directly to git)
- [x] All 7 tasks implemented and documented in `report/report.md`
- [x] AI usage documented in `AI_USE.md`
- [x] Final commit tagged as `final`
