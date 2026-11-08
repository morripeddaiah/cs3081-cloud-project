# Handwritten Reflection Notes & Learning Synthesis (Full 3-Page Script)

**Student Name:** Morri Peddaiah  
**Roll Number:** 230101051  
**Course:** CS3081 Cloud Computing  
**GitHub Repository:** https://github.com/morripeddaiah/cs3081-cloud-project  

---

## PAGE 1: Header, Learnings & Knowledge Distillation

### 1. Executive Summary & Core Objectives
In this mini project, I designed, implemented, and deployed an end-to-end cloud-native machine learning system for a command-parsing language model. The core challenge addressed was bridging the gap between resource-heavy deep learning workloads and low-end, constrained local cloud infrastructure. Sequence-level knowledge distillation (KD) was applied to compress a 3.8-billion-parameter teacher model (`Phi-3.5-mini-instruct`) into an ultra-compact 70-million-parameter student model (`EleutherAI/pythia-70m`) deployed on a local k3d Kubernetes cluster.

### 2. Theoretical Learnings & Distillation in Practice
Prior to this project, I conceptualized knowledge distillation purely as a mathematical optimization problem where a student model matches the softened softmax logit distributions of a teacher via Kullback-Leibler (KL) divergence. However, practical engineering constraints quickly revealed that logit-based distillation was infeasible for two fundamental reasons:
1. **Vocabulary Incompatibility:** The teacher (`phi3.5`) utilizes a 32,064-token tokenizer, whereas the student (`pythia-70m`) uses a distinct 50,304-token GPT-NeoX vocabulary. Because output probability dimensions do not correspond to the same lexical tokens, direct token-level logit matching is mathematically undefined without an expensive cross-projection layer.
2. **Compute Decoupling:** Running both a 3.8B teacher and a 70M student simultaneously on consumer CPU hardware for forward and backward passes would exceed laptop memory bandwidth and cause severe throttling.

Sequence-level knowledge distillation (Kim & Rush, 2016) provided the solution: by running the teacher offline with deterministic parameters (temperature $0.0$, seed $42$) to generate greedy target strings, distillation was transformed into supervised sequence-to-sequence training. The student model learned the structured output format without requiring joint execution.

---

## PAGE 2: Architectural Decoupling, Networking & Practical Challenges

### 3. Storage Decoupling & Container Optimization
A frequent anti-pattern in containerized machine learning is embedding model checkpoints directly into Docker images. Doing so would produce multi-gigabyte container images, resulting in slow scheduling, high bandwidth consumption, and container engine timeouts. 

To solve this, I decoupled the storage layer:
- Deployed SeaweedFS in the `data` namespace as an S3-compatible object store.
- Built a multi-stage Dockerfile based on `python:3.11-slim` using CPU-only PyTorch wheels, keeping image size below 1.2 GB.
- Configured dynamic runtime weight fetching: when a pod initializes, it downloads model weights directly from `s3://models/v1/` using credentials injected securely via Kubernetes Secrets (`s3-credentials`).

### 4. Challenges Encountered & Concrete Solutions
1. **CPU Backpropagation Bottlenecks:** Fine-tuning `pythia-70m` on CPU initially took ~35 seconds per batch step at sequence length 128. Profiling the SLURP dataset revealed that the 95th percentile command length was only 44 tokens. By reducing `max_seq_len` to 64 and tuning gradient accumulation, throughput tripled while remaining within memory limits.
2. **Teacher Hallucination & Data Sanitization:** The teacher occasionally outputted markdown fences or hallucinated slot entities. I developed a 5-stage cleaning script (`worker/clean.py`) enforcing valid JSON syntax, exact 4 keys (`intent`, `date`, `time`, `place`), verified the 8 allowed intents, filtered hallucinated slots via substring matching against user input, and deduplicated identical commands by retaining the lowest ID. This yielded 2,937 pristine training records with SHA-256 verification.
3. **Stateless Access Control Lists (NACL):** Implementing the public subnet NACL in `network/nacl.json` highlighted the nature of stateless networking: return traffic is not tracked. Rule 100 was set to strictly `DENY 198.51.100.0/24` on all ports, preceding web allow rules (ports 80/443). Furthermore, explicit outbound allow rules for ephemeral client ports (1024–65535) and port 8000 to the app subnet were established, passing 100% of simulator verification packets.

---

## PAGE 3: Empirical Findings, Cloud Economics & Proposed Improvements

### 5. Empirical Observations & Performance Under Load
Load testing with Locust (4 concurrent users for 2 minutes against the ingress endpoint) demonstrated high reliability:
- **Throughput & Latency:** 4,280 requests completed with 0 errors (0.0% failure rate). Median latency (p50) was **74 ms**, and 95th-percentile (p95) was **114 ms** at **35.67 req/s**.
- **Traffic Balancing:** The Traefik load balancer distributed traffic evenly across the two pods (51.05% vs 48.95%).
- **Cold Start & Fault Recovery:** Deleting all pods demonstrated dynamic weight retrieval from SeaweedFS with a median cold start to Ready state of **7.02 seconds**. Rolling updates (`maxSurge: 1, maxUnavailable: 0`) and pod kill tests caused <0.65% dropped requests, recovering the 2-replica state in 7.15 seconds.
- **Economic Feasibility:** Based on `data/rate_card.json`, cost per 1,000 requests was calculated at **INR 0.0615**. Under our 99% SLO, the error budget was 42 requests, of which 0 was consumed.

### 6. Limitations & Proposed Improvements
1. **Single-Node Infrastructure:** The k3d cluster operates on a single local laptop host. In production, deploying across multi-zone managed Kubernetes (EKS/GKE) with pod anti-affinity would prevent single-host node failures.
2. **Dynamic Logit Distillation with Shared Tokenizer:** If utilizing a teacher and student with unified vocabulary (e.g., Llama-3 teacher to Llama-100M student), token-level cross-entropy combined with KL-divergence logit loss could capture dark knowledge and subtle token distribution probabilities.
3. **Quantization & Edge Compilation:** Exporting Pythia-70M to ONNX Runtime or 4-bit INT quantization (GGUF) would reduce model RAM footprint from 340 MB to under 100 MB and double CPU inference speed for edge microcontrollers.
