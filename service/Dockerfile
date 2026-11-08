# Stage 1: Build & install dependencies
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install CPU-only PyTorch and python packages
COPY service/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Stage 2: Final lightweight runtime container (< 2GB, no model weights baked in)
FROM python:3.11-slim AS runner

# Create non-root user and group
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Copy python libraries from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy service application code
COPY service/app.py /app/app.py

# Set proper ownership and permissions
RUN chown -R appuser:appgroup /app /tmp
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_URI="models/v1" \
    MODEL_VERSION="v1" \
    S3_ENDPOINT="http://seaweedfs.data.svc.cluster.local:8333" \
    MAX_NEW_TOKENS=64 \
    FAULT_INJECT=0

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

