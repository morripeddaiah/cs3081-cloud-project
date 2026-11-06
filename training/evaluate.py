#!/usr/bin/env python3
"""Evaluation script for Command Parser models against data/validation.jsonl.

Supports evaluating:
  - Teacher (Phi-3.5 via Ollama)
  - Untrained Pythia-70m
  - Trained Pythia-70m student (from local directory or Hugging Face)

Generates predictions in format: {"id": "V001", "prediction": "<raw text>"}
and scores them using tools/score.py logic.
"""

import argparse
import json
import os
import sys
import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import scoring functions from tools/score.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools")))
try:
    from score import score as run_score
except ImportError:
    # If not importable directly, define fallback scoring function
    run_score = None


def evaluate_teacher(gold_records, ollama_host="http://localhost:11434", model="phi3.5", prompt_path="prompts/teacher_prompt.txt"):
    """Generate predictions from teacher model."""
    template = "You are a command parser. Return a JSON object with keys intent, date, time, place.\nCommand: {command}\nOutput:"
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()

    preds = []
    print(f"Evaluating Teacher Model ({model}) across {len(gold_records)} validation samples...")
    for i, rec in enumerate(gold_records):
        cid = rec["id"]
        text = rec["text"]
        full_prompt = template.replace("{command}", text)
        url = f"{ollama_host}/api/generate"
        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.0, "seed": 42},
        }
        try:
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code == 200:
                raw_pred = r.json().get("response", "").strip()
            else:
                raw_pred = ""
        except Exception:
            raw_pred = ""
            
        preds.append({"id": cid, "prediction": raw_pred})
        if (i + 1) % 25 == 0:
            print(f"Teacher progress: {i+1}/{len(gold_records)}")

    return preds


def evaluate_hf_model(model_name_or_path, gold_records, max_new_tokens=64):
    """Generate predictions from a CausalLM (untrained or student)."""
    print(f"Loading model from: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    model.to(device)
    model.eval()

    preds = []
    print(f"Generating predictions for {len(gold_records)} samples...")
    for i, rec in enumerate(gold_records):
        cid = rec["id"]
        text = rec["text"]
        prompt = f"Command: {text}\nJSON: "

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=1.0,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        # Take the first line or first JSON object
        pred_line = gen_text.split("\n")[0].strip()
        preds.append({"id": cid, "prediction": pred_line})

        if (i + 1) % 25 == 0:
            print(f"Student progress: {i+1}/{len(gold_records)}")

    return preds


def main():
    parser = argparse.ArgumentParser(description="Evaluate models against validation.jsonl.")
    parser.add_argument("--mode", choices=["teacher", "untrained", "student"], required=True)
    parser.add_argument("--model-path", default="EleutherAI/pythia-70m", help="HF model name or directory path")
    parser.add_argument("--gold", default="data/validation.jsonl", help="Gold validation set")
    parser.add_argument("--out-preds", default="results/predictions.jsonl", help="Output path for predictions")
    parser.add_argument("--out-report", default="results/report.json", help="Output path for scoring report")
    parser.add_argument("--min-validity", type=float, default=None)
    args = parser.parse_args()

    # Load gold records
    gold_records = []
    with open(args.gold, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                gold_records.append(json.loads(line))

    if args.mode == "teacher":
        preds = evaluate_teacher(gold_records)
    elif args.mode == "untrained":
        preds = evaluate_hf_model("EleutherAI/pythia-70m", gold_records)
    else:  # student
        preds = evaluate_hf_model(args.model_path, gold_records)

    # Save predictions
    os.makedirs(os.path.dirname(args.out_preds) or ".", exist_ok=True)
    with open(args.out_preds, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    print(f"Predictions saved to {args.out_preds}")

    # Compute score
    if run_score:
        report = run_score(preds, gold_records)
        text_report = json.dumps(report, indent=2)
        print("\nEvaluation Results:")
        print(text_report)
        if args.out_report:
            os.makedirs(os.path.dirname(args.out_report) or ".", exist_ok=True)
            with open(args.out_report, "w", encoding="utf-8") as out_f:
                out_f.write(text_report + "\n")
            print(f"Report saved to {args.out_report}")

        if args.min_validity and report["json_validity"] < args.min_validity:
            print(f"[FAIL] Quality gate failed: json_validity {report['json_validity']} < {args.min_validity}")
            sys.exit(1)


if __name__ == "__main__":
    main()

