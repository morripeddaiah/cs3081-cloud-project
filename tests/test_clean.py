"""Unit tests for dataset cleaning logic in worker/clean.py."""

import unittest
from worker.clean import parse_teacher_reply, validate_record


class TestDatasetCleaning(unittest.TestCase):
    def test_parse_clean_json(self):
        raw = '{"intent": "alarm_set", "date": null, "time": "6 am", "place": null}'
        parsed = parse_teacher_reply(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["intent"], "alarm_set")
        self.assertEqual(parsed["time"], "6 am")

    def test_parse_markdown_wrapped_json(self):
        raw = '```json\n{"intent": "weather_query", "date": null, "time": null, "place": "tokyo"}\n```'
        parsed = parse_teacher_reply(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["place"], "tokyo")

    def test_parse_invalid_text(self):
        raw = "I am an AI assistant and I think the user wants to set an alarm."
        parsed = parse_teacher_reply(raw)
        self.assertIsNone(parsed)

    def test_validate_record_valid(self):
        cmd = "set my alarm for six am"
        obj = {"intent": "alarm_set", "date": None, "time": "six am", "place": None}
        valid, reason = validate_record(cmd, obj)
        self.assertTrue(valid)
        self.assertEqual(reason, "ok")

    def test_validate_record_hallucinated_slot(self):
        cmd = "what is the weather"
        obj = {"intent": "weather_query", "date": None, "time": None, "place": "paris"}
        valid, reason = validate_record(cmd, obj)
        self.assertFalse(valid)
        self.assertIn("hallucinated", reason)

    def test_validate_record_wrong_keys(self):
        cmd = "what is the weather in london"
        obj = {"intent": "weather_query", "location": "london"}
        valid, reason = validate_record(cmd, obj)
        self.assertFalse(valid)
        self.assertEqual(reason, "wrong_keys")


if __name__ == "__main__":
    unittest.main()
