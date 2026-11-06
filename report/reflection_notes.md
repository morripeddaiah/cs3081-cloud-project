# Handwritten Reflection Notes & Learning Synthesis
**Student Name:** Morri Peddaiah  
**Roll Number:** 230101051  
**Course:** CS3081 Cloud Computing  

---

### Reflection 1: Knowledge Distillation in Practice
When embarking on this project, I initially perceived knowledge distillation as a mathematical matching problem where student logits are forced to match teacher softmax distributions. However, practical constraints quickly set in: running a 3.8-billion parameter teacher and student simultaneously on CPU hardware was infeasible. Sequence-level knowledge distillation proved to be far more elegant for real-world engineering. By having the teacher generate ground-truth command outputs offline at temperature 0, the distillation process decoupled the compute requirements of the teacher from the student.

The trade-off was striking: although the student model is only 70M parameters (a 98% reduction in size), it maintained over 97% JSON validity and 76% full frame accuracy. The small model essentially learned the syntax and semantic slot templates without needing massive reasoning capabilities.

### Reflection 2: Storage Decoupling & Stateless Ingress
In conventional application development, one often falls into the trap of packaging all assets—including model weights—into the Docker image. Doing so would have resulted in bloated 3+ GB images, slowing down cold starts and Kubernetes scheduling. Decoupling the model checkpoint to SeaweedFS (S3-compatible object storage) was a crucial architectural decision. The inference container image remained lean (~1.15 GB), and rolling updates could update models by simply pointing to `s3://models/v2/` via environment variables.

Furthermore, implementing stateless NACL rules underscored the fundamental difference between stateful firewalls and stateless access control lists: every packet, including the ephemeral return traffic to clients, must have an explicit allow rule.

### Reflection 3: Cloud Economics & Resource Boundaries
Deploying microservices without resource requests and limits in Kubernetes is dangerous. Configuring `500m/512Mi` requests and `1000m/1Gi` limits ensured predictability and enabled HPA to scale accurately at 70% CPU threshold. Calculating cloud costs using the rate card showed that our 2-replica deployment cost merely INR 0.0615 per 1,000 requests. This concrete economic calculation revealed how cloud infrastructure choices directly govern financial feasibility.

*Note: For the physical submission, please attach the scanned handwritten copy of these reflection notes alongside this transcript.*
