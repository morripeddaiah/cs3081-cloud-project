"""Unit tests for JSON schema validation and slot constraints."""

import unittest
from service.app import validate_schema


class TestSchemaValidation(unittest.TestCase):
    def test_valid_schema_object(self):
        """Valid 4-key dictionary passes validation without error."""
        valid_data = {
            "intent": "weather_query",
            "date": None,
            "time": None,
            "place": "london",
        }
        # Should not raise exception
        validate_schema(valid_data)

    def test_missing_required_key(self):
        """Dictionary missing a key raises ValueError."""
        data = {
            "intent": "weather_query",
            "date": None,
            "time": None,
            # missing place
        }
        with self.assertRaises(ValueError):
            validate_schema(data)

    def test_extra_key_forbidden(self):
        """Dictionary with unexpected extra key raises ValueError."""
        data = {
            "intent": "alarm_set",
            "date": "tomorrow",
            "time": "7 am",
            "place": None,
            "confidence": 0.99,
        }
        with self.assertRaises(ValueError):
            validate_schema(data)

    def test_invalid_intent(self):
        """Disallowed intent string raises ValueError."""
        data = {
            "intent": "play_music",  # Not in allowed 8 intents
            "date": None,
            "time": None,
            "place": None,
        }
        with self.assertRaises(ValueError):
            validate_schema(data)

    def test_invalid_slot_type(self):
        """Slot value that is not string or null raises ValueError."""
        data = {
            "intent": "alarm_set",
            "date": None,
            "time": 700,  # Integer instead of string or null
            "place": None,
        }
        with self.assertRaises(ValueError):
            validate_schema(data)


if __name__ == "__main__":
    unittest.main()
