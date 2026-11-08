"""Unit tests for FastAPI command parsing service."""

import os
import unittest
from fastapi.testclient import TestClient

# Set environment before importing app
os.environ["MODEL_VERSION"] = "v1"
os.environ["FAULT_INJECT"] = "0"

from service.app import app, state


class DummyModel:
    pass


class DummyTokenizer:
    eos_token_id = 0


class TestInferenceService(unittest.TestCase):
    def setUp(self):
        state["is_ready"] = True
        state["model"] = DummyModel()
        state["tokenizer"] = DummyTokenizer()
        self.client = TestClient(app)

    def tearDown(self):
        state["is_ready"] = False
        os.environ["FAULT_INJECT"] = "0"

    def test_healthz_endpoint(self):
        """Test liveness probe returns HTTP 200 and ok status."""
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_readyz_endpoint_ready(self):
        """Test readiness probe returns HTTP 200 when model is loaded."""
        response = self.client.get("/readyz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ready")
        self.assertEqual(response.json()["model_version"], "v1")

    def test_readyz_endpoint_not_ready(self):
        """Test readiness probe returns HTTP 503 when model is not ready."""
        state["is_ready"] = False
        response = self.client.get("/readyz")
        self.assertEqual(response.status_code, 503)

    def test_fault_injection(self):
        """Test that FAULT_INJECT=1 causes endpoint to return HTTP 500 with exact error body."""
        os.environ["FAULT_INJECT"] = "1"
        response = self.client.post("/parse", json={"text": "what time is it"})
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"error": "invalid_model_output"})


if __name__ == "__main__":
    unittest.main()
