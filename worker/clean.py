#!/usr/bin/env python3
"""Dataset cleaning script for teacher-generated shards.

Applies strict data cleaning criteria:
1. Valid JSON: output parses as a valid JSON object.
2. Exact 4 keys: contains exactly 'intent', 'date', 'time', 'place'.
3. Valid Intent: intent is one of the 8 allowed intents from schema.json.
4. No Hallucinated Slots: any non-null slot string must actually appear in the command text (case-insensitive).
5. Deduplication: removes duplicate commands based on command text.

Outputs cleaned data to datasets/v1/train.jsonl and calculates SHA-256 checksum.
"""

import argparse
import glob
import hashlib
import json
import os
import sys

try:
    import boto3
    from botocore.client import Config
except ImportError:
    boto3 = None
    Config = None

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

REQUIRED_KEYS = {"intent", "date", "time", "place"}


def parse_teacher_reply(raw_text):
    """Attempt to extract and parse JSON object from teacher output."""
    if not raw_text:
        return None
    raw_text = str(raw_text).strip()
    # Strip markdown formatting if present
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    raw_text = raw_text.strip()
    
    try:
        obj = json.loads(raw_text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
        
    # Attempt to extract first substring enclosed in curly braces
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            candidate = raw_text[start : end + 1]
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
            
    return None


def validate_record(cmd_text, parsed_obj):
    """Validate parsed object against cleaning criteria.
    Returns (is_valid, reason).
    """
    if parsed_obj is None:
        return False, "invalid_json"

    # Check exact keys
    if set(parsed_obj.keys()) != REQUIRED_KEYS:
        return False, "wrong_keys"

    # Check intent
    intent = parsed_obj.get("intent")
    if intent not in ALLOWED_INTENTS:
        return False, "invalid_intent"

    # Check slot types and hallucination
    cmd_lower = cmd_text.lower()
    for slot_name in ("date", "time", "place"):
        val = parsed_obj.get(slot_name)
        if val is not None:
            if not isinstance(val, str):
                return False, f"slot_{slot_name}_not_string_or_null"
            # Slot value must be a substring of the command text (no hallucination)
            val_clean = val.strip().lower()
            if val_clean and val_clean not in cmd_lower:
                return False, f"hallucinated_slot_{slot_name}"

    return True, "ok"


def compute_sha256(filepath):
    """Calculate SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Clean teacher-generated shards.")
    parser.add_argument("--shards-dir", default="shards", help="Directory containing shard JSONL files")
    parser.add_argument("--out-dir", default="datasets/v1", help="Output directory")
    parser.add_argument("--s3-sync", action="store_true", help="Sync results to S3/SeaweedFS")
    parser.add_argument("--s3-endpoint", default=os.environ.get("S3_ENDPOINT", "http://seaweedfs.data.svc.cluster.local:8333"))
    parser.add_argument("--s3-bucket", default="datasets")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, "train.jsonl")
    checksum_file = os.path.join(args.out_dir, "train.sha256")

    shard_files = sorted(glob.glob(os.path.join(args.shards-dir if hasattr(args, "shards-dir") else args.shards_dir, "*.jsonl")))
    
    # If no shard files found in specified dir, check root shards/
    if not shard_files:
        shard_files = sorted(glob.glob("shards/*.jsonl"))

    print(f"Found {len(shard_files)} shard files: {shard_files}")

    stats = {
        "total_read": 0,
        "invalid_json": 0,
        "wrong_keys": 0,
        "invalid_intent": 0,
        "hallucinated_slot": 0,
        "slot_type_error": 0,
        "duplicates": 0,
        "valid_kept": 0,
    }

    seen_texts = set()
    cleaned_records = []

    for s_path in shard_files:
        print(f"Reading {s_path}...")
        with open(s_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stats["total_read"] += 1
                try:
                    record = json.loads(line)
                except Exception:
                    stats["invalid_json"] += 1
                    continue

                cmd_text = record.get("text", "")
                raw_teacher = record.get("teacher_output", "")
                parsed = parse_teacher_reply(raw_teacher)

                is_valid, reason = validate_record(cmd_text, parsed)
                if not is_valid:
                    if "hallucinated" in reason:
                        stats["hallucinated_slot"] += 1
                    elif "not_string" in reason:
                        stats["slot_type_error"] += 1
                    elif reason in stats:
                        stats[reason] += 1
                    else:
                        stats["invalid_json"] += 1
                    continue

                # Deduplication check
                norm_text = " ".join(cmd_text.strip().lower().split())
                if norm_text in seen_texts:
                    stats["duplicates"] += 1
                    continue

                seen_texts.add(norm_text)
                
                # Standard clean format for student training
                clean_entry = {
                    "id": record.get("id"),
                    "text": cmd_text,
                    "target": parsed,
                }
                cleaned_records.append(clean_entry)
                stats["valid_kept"] += 1

    # Write output file
    with open(out_file, "w", encoding="utf-8") as f_out:
        for item in cleaned_records:
            f_out.write(json.dumps(item) + "\n")

    # Compute and save SHA-256
    sha256_hash = compute_sha256(out_file)
    with open(checksum_file, "w", encoding="utf-8") as f_sha:
        f_sha.write(f"{sha256_hash}  train.jsonl\n")

    print("\n" + "=" * 45)
    print("DATASET CLEANING REPORT")
    print("=" * 45)
    for k, v in stats.items():
        print(f"  {k:<22}: {v}")
    print(f"  SHA-256 Checksum     : {sha256_hash}")
    print(f"  Output File          : {out_file}")
    print("=" * 45)

    if args.s3_sync:
        try:
            s3 = boto3.client(
                "s3",
                endpoint_url=args.s3_endpoint,
                aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "seaweedfsadmin"),
                aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "seaweedfsadmin"),
                config=Config(signature_version="s3v4"),
                region_name="us-east-1",
            )
            print(f"Uploading cleaned dataset to s3://{args.s3_bucket}/v1/train.jsonl...")
            s3.upload_file(out_file, args.s3_bucket, "v1/train.jsonl")
            s3.upload_file(checksum_file, args.s3_bucket, "v1/train.sha256")
            print("S3 upload complete.")
        except Exception as e:
            print(f"[WARN] Failed to sync to S3: {e}")


if __name__ == "__main__":
    main()
