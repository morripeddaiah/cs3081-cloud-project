#!/usr/bin/env python3
"""Inference Service for Lightweight Command-Parsing Language Model.

FastAPI application providing:
  - POST /parse: runs command through Pythia-70M student model, returns 4-key JSON.
  - GET  /healthz: liveness probe (HTTP 200).
  - GET  /readyz: readiness probe (HTTP 200 when model loaded, 503 otherwise).
"""

import json
import os
import socket
import sys
from contextlib import asynccontextmanager
from typing import Optional

try:
    import boto3
    from botocore.client import Config
except ImportError:
    boto3 = None
    Config = None

from fastapi import FastAPI, HTTPException, Request, Response, status
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration from Environment Variables
MODEL_URI = os.environ.get("MODEL_URI", "models/v1")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://seaweedfs.data.svc.cluster.local:8333")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "seaweedfsadmin")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "seaweedfsadmin")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
FAULT_INJECT = os.environ.get("FAULT_INJECT", "0")

# Host / Pod identification for X-Served-By header
SERVED_BY = os.environ.get("HOSTNAME", socket.gethostname())

REQUIRED_KEYS = {"intent", "date", "time", "place"}
ALLOWED_INTENTS = {
    "alarm_set",
    "alarm_query",
    "calendar_set",
    "calendar_query",
    "calendar_remove",
    "weather_query",
    "datetime_query",
    "recommendation_events",
}

# Global runtime state
state = {
    "model": None,
    "tokenizer": None,
    "is_ready": False,
    "device": torch.device("cpu"),
}


def download_model_from_s3(s3_uri: str, target_dir: str):
    """Download model artifacts from SeaweedFS / S3 URI (s3://bucket/prefix)."""
    print(f"Downloading model artifacts from {s3_uri} to {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = os.path.relpath(key, prefix) if prefix else key
            dest_file = os.path.join(target_dir, rel_path)
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            print(f"Downloading s3://{bucket}/{key} -> {dest_file}")
            s3.download_file(bucket, key, dest_file)


def load_model():
    """Load model and tokenizer into memory."""
    local_path = MODEL_URI
    if MODEL_URI.startswith("s3://"):
        cache_dir = os.path.join("/tmp", "cached_model", MODEL_VERSION)
        download_model_from_s3(MODEL_URI, cache_dir)
        local_path = cache_dir

    print(f"Loading model and tokenizer from {local_path} on CPU...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(local_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(local_path)
        model.to(state["device"])
        model.eval()

        state["model"] = model
        state["tokenizer"] = tokenizer
        state["is_ready"] = True
        print(f"Model {MODEL_VERSION} successfully loaded and ready.")
    except Exception as e:
        print(f"[ERROR] Failed to load model from {local_path}: {e}")
        state["is_ready"] = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    load_model()
    yield
    # Teardown logic if needed
    state["is_ready"] = False


app = FastAPI(
    title="Command Parsing Inference Service",
    description="CS3081 Knowledge Distillation Command Parser Service",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


class ParseRequest(BaseModel):
    text: str = Field(..., description="The user command to parse", example="set an alarm for seven am")


class ParseResponse(BaseModel):
    intent: str
    date: Optional[str] = None
    time: Optional[str] = None
    place: Optional[str] = None


def validate_schema(data: dict):
    """Ensure data is a dictionary with exactly the 4 required keys and valid types."""
    if not isinstance(data, dict):
        raise ValueError("Output is not a valid JSON object")
    if set(data.keys()) != REQUIRED_KEYS:
        raise ValueError(f"Keys mismatch: expected {REQUIRED_KEYS}, got {set(data.keys())}")
    if data["intent"] not in ALLOWED_INTENTS:
        raise ValueError(f"Invalid intent: {data['intent']}")
    for k in ("date", "time", "place"):
        if data[k] is not None and not isinstance(data[k], str):
            raise ValueError(f"Slot {k} must be string or null")


@app.get("/healthz", status_code=status.HTTP_200_OK)
def healthz():
    """Liveness probe: verifies service process is running."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    """Readiness probe: returns 200 when model is loaded, 503 while loading."""
    if not state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet loaded into memory",
        )
    return {"status": "ready", "model_version": MODEL_VERSION}


@app.post("/parse", response_model=ParseResponse)
def parse_command(req: ParseRequest, response: Response):
    """Parse a natural language command into structured JSON slots and intent."""
    response.headers["X-Model-Version"] = MODEL_VERSION
    response.headers["X-Served-By"] = SERVED_BY

    # Fault injection mode for testing error handling
    if os.environ.get("FAULT_INJECT", "0") == "1" or FAULT_INJECT == "1":
        print("[FAULT_INJECT] Forcing invalid output: not json")
        return Response(
            content=json.dumps({"error": "invalid_model_output"}),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
            headers={
                "X-Model-Version": MODEL_VERSION,
                "X-Served-By": SERVED_BY,
            },
        )

    if not state["is_ready"]:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is not ready (model still loading)",
        )

    model = state["model"]
    tokenizer = state["tokenizer"]

    prompt = f"Command: {req.text}\nJSON: "
    inputs = tokenizer(prompt, return_tensors="pt").to(state["device"])

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=1.0,
        )

    gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    # Extract JSON line
    first_line = gen_text.split("\n")[0].strip()

    try:
        parsed = json.loads(first_line)
        validate_schema(parsed)
    except Exception as err:
        # Schema failure must return HTTP 500 with exact error body
        print(f"[ERROR] Schema validation error on '{first_line}': {err}")
        return Response(
            content=json.dumps({"error": "invalid_model_output"}),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            media_type="application/json",
            headers={
                "X-Model-Version": MODEL_VERSION,
                "X-Served-By": SERVED_BY,
            },
        )

    return parsed


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
