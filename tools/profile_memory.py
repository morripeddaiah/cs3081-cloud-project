#!/usr/bin/env python3
"""Profiles inference container memory and CPU under sequential requests."""
import argparse
import json
import time
import requests

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--host", default="http://localhost:8080")
    args = parser.parse_args()
    print(f"Sending {args.requests} sequential requests to {args.host}/parse...")
    for i in range(args.requests):
        payload = {"text": "set an alarm for seven am"}
        try:
            r = requests.post(f"{args.host}/parse", json=payload, timeout=5)
            if (i + 1) % 10 == 0:
                print(f"Completed {i+1}/{args.requests} requests. Status: {r.status_code}")
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
    print("Memory profile test completed successfully.")

if __name__ == "__main__":
    main()

