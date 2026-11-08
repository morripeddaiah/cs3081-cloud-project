"""Complete test suite unifying test_parse, test_schema, and test_clean."""
import unittest
from tests.test_parse import TestInferenceService
from tests.test_schema import TestSchemaValidation
from tests.test_clean import TestDatasetCleaning

if __name__ == "__main__":
    unittest.main()

