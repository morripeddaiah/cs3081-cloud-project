#!/usr/bin/env python3
"""Worker for parallel teacher labelling using Ollama (phi3.5:latest).

Reads a subset of commands according to SHARD_INDEX and SHARD_COUNT,
queries the teacher model via Ollama REST API with temperature=0 and seed=42,
and uploads the labelled shard to SeaweedFS (S3-compatible object storage).
"""

import json
import os
import sys
import time
import requests
import boto3
from botocore.client import Config

# Configuration from Environment Variables with sensible defaults
SHARD_INDEX = int(os.environ.get("SHARD_INDEX", "0"))
SHARD_COUNT = int(os.environ.get("SHARD_COUNT", "2"))
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "phi3.5")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT", "http://seaweedfs.data.svc.cluster.local:8333")
S3_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY_ID", "seaweedfsadmin")
S3_SECRET_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "seaweedfsadmin")
S3_BUCKET = os.environ.get("S3_BUCKET", "datasets")
INPUT_FILE = os.environ.get("INPUT_FILE", "data/commands_pool.jsonl")
PROMPT_TEMPLATE_PATH = os.environ.get("PROMPT_TEMPLATE", "prompts/teacher_prompt.txt")
LOCAL_OUTPUT_DIR = os.environ.get("LOCAL_OUTPUT_DIR", "shards")


def get_s3_client():
    """Create and return an S3 client configured for SeaweedFS."""
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=S3_ACCESS_KEY,
            aws_secret_access_key=S3_SECRET_KEY,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )
        return s3
    except Exception as e:
        print(f"[WARN] Failed to initialize S3 client: {e}")
        return None


def load_prompt_template(path):
    """Load teacher prompt template from file."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        # Fallback inline template if file not found
        return (
            "You are a command parser. Return a JSON object with keys intent, date, time, place.\n"
            "Command: {command}\nOutput:"
        )


def query_teacher(prompt, command_text):
    """Query Phi-3.5 via Ollama API with deterministic parameters (temp=0, seed=42)."""
    full_prompt = prompt.replace("{command}", command_text)
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": TEACHER_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "seed": 42,
        },
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("response", "").strip()
            else:
                print(f"[WARN] Ollama returned status {resp.status_code}: {resp.text}")
        except Exception as err:
            print(f"[WARN] Request failed (attempt {attempt+1}/{max_retries}): {err}")
            time.sleep(2)
            
    return ""


def main():
    print(f"Starting Worker: SHARD_INDEX={SHARD_INDEX}, SHARD_COUNT={SHARD_COUNT}")
    print(f"Ollama Host: {OLLAMA_HOST}, Model: {TEACHER_MODEL}")

    prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)
    
    # Read all commands from input file
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file {INPUT_FILE} not found!")
        sys.exit(1)
        
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        all_commands = [json.loads(line) for line in f if line.strip()]

    total_cmds = len(all_commands)
    # Partition commands deterministically
    shard_commands = [
        cmd for idx, cmd in enumerate(all_commands)
        if idx % SHARD_COUNT == SHARD_INDEX
    ]
    print(f"Processing {len(shard_commands)} of {total_cmds} total commands.")

    os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
    local_shard_file = os.path.join(LOCAL_OUTPUT_DIR, f"shard_{SHARD_INDEX}.jsonl")
    s3_key = f"shards/shard_{SHARD_INDEX}.jsonl"

    labelled_count = 0
    with open(local_shard_file, "w", encoding="utf-8") as out_f:
        for i, cmd in enumerate(shard_commands):
            cmd_id = cmd.get("id")
            text = cmd.get("text", "")
            
            raw_reply = query_teacher(prompt_template, text)
            
            record = {
                "id": cmd_id,
                "text": text,
                "teacher_output": raw_reply,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            labelled_count += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(shard_commands):
                print(f"Progress: {i+1}/{len(shard_commands)} commands labelled.")

    print(f"Shard {SHARD_INDEX} complete. Written {labelled_count} records to {local_shard_file}.")

    # Upload shard to SeaweedFS / S3
    s3 = get_s3_client()
    if s3:
        try:
            print(f"Uploading {local_shard_file} to s3://{S3_BUCKET}/{s3_key}...")
            # Ensure bucket exists
            try:
                s3.head_bucket(Bucket=S3_BUCKET)
            except Exception:
                try:
                    s3.create_bucket(Bucket=S3_BUCKET)
                except Exception:
                    pass
            s3.upload_file(local_shard_file, S3_BUCKET, s3_key)
            print("Upload to S3 successful.")
        except Exception as e:
            print(f"[WARN] Could not upload to S3: {e}. Local file preserved.")


if __name__ == "__main__":
    main()

