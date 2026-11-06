#!/usr/bin/env python3
"""Standard library test runner for unit tests without requiring pytest."""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from tests.test_schema import (
    test_valid_schema_object,
    test_missing_required_key,
    test_extra_key_forbidden,
    test_invalid_intent,
    test_invalid_slot_type,
)
from tests.test_clean import (
    test_parse_clean_json,
    test_parse_markdown_wrapped_json,
    test_parse_invalid_text,
    test_validate_record_valid,
    test_validate_record_hallucinated_slot,
    test_validate_record_wrong_keys,
)

test_cases = [
    test_valid_schema_object,
    test_missing_required_key,
    test_extra_key_forbidden,
    test_invalid_intent,
    test_invalid_slot_type,
    test_parse_clean_json,
    test_parse_markdown_wrapped_json,
    test_parse_invalid_text,
    test_validate_record_valid,
    test_validate_record_hallucinated_slot,
    test_validate_record_wrong_keys,
]

failed = 0
passed = 0

print("Running Unit Tests:")
print("-" * 50)
for t in test_cases:
    try:
        t()
        print(f" [PASS] {t.__name__}")
        passed += 1
    except Exception as e:
        print(f" [FAIL] {t.__name__}: {e}")
        failed += 1

print("-" * 50)
print(f"Summary: {passed} passed, {failed} failed.")
if failed > 0:
    sys.exit(1)
sys.exit(0)
