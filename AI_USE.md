# AI Use Disclosure & Provenance Statement

**Course:** Cloud Computing (CS3081) — Mini Project  
**Student Name:** Morri Peddaiah  
**Student Roll Number:** 230101051  

---

## 1. Overview of AI Assistance
In accordance with course guidelines permitting AI assistance with critical awareness and full disclosure, this document outlines where and how AI tools were employed in the design, development, and validation of this project.

## 2. Tools Used
- **Antigravity Coding Assistant (Gemini 3.8 Flash / Claude 3.5 Sonnet)**: Used for scaffolding repository structure, initial boilerplate code, unit test generation, and architectural cross-referencing.
- **Phi-3.5-mini-instruct (via Ollama)**: Employed as the teacher model for sequence-level knowledge distillation on the SLURP command dataset.
- **EleutherAI/pythia-70m**: Pretrained causal language model used as the lightweight student model.

## 3. Specific Areas of AI Generation and Human Review

| Component | AI Role | Human Validation & Modifications |
|-----------|---------|----------------------------------|
| **Teacher Prompting** (`prompts/teacher_prompt.txt`) | Generated prompt template incorporating slot extraction rules. | Tested and verified against sample SLURP inputs to ensure strict 4-key JSON and adherence to zero-temperature deterministic outputs. |
| **Data Cleaning** (`worker/clean.py`) | Generated JSON validator and hallucination detection logic. | Verified 5-point data sanitization criteria (exact keys, allowed intents, substring match for non-hallucination, deduplication). |
| **Model Distillation & Training** (`training/train.py`) | Provided standard PyTorch/HuggingFace CausalLM training loop. | Tuned sequence length, learning rate (5e-4), AdamW scheduler, and CPU-only training constraints. |
| **Inference Service** (`service/app.py`) | Generated FastAPI endpoints and probe definitions. | Enforced strict HTTP 500 error propagation on schema failure, fault injection handling (`FAULT_INJECT=1`), and response headers (`X-Served-By`, `X-Model-Version`). |
| **Kubernetes Manifests** (`k8s/*.yaml`) | Provided standard resource blocks for Deployment, Service, HPA, Ingress. | Verified resource requests (500m / 512Mi) and limits (1000m / 1Gi), health probe timings, rolling update constraints (`maxSurge: 1, maxUnavailable: 0`), and Traefik ingress rules. |
| **Subnet Allocation & NACL** (`network/*`) | Suggested CIDR partitioning for 10.20.0.0/16. | Manually derived host bit requirements for 1000, 250, 100, and 20 hosts; validated 100% of packets against `tools/nacl_sim.py`. |
| **CI/CD & Testing** (`.github/workflows/ci.yml`, `tests/*`) | Generated test cases and GitHub Actions workflow. | Verified linting, unit test execution, gitleaks secret scanning, and quality gate score enforcement (`--min-validity 95`). |

## 4. Academic Integrity & Understanding
All design decisions, distillation methodologies (sequence-level vs. logit-based matching), Kubernetes scheduling mechanics, and networking policies have been thoroughly verified and are fully understood by the student.

