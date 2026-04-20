import unittest

import numpy as np

from pipeline.batch_metadata import normalize_batch_metadata_values


class BatchMetadataTests(unittest.TestCase):
    def test_normalize_batch_metadata_values_wraps_numpy_scalar_string(self):
        normalized = normalize_batch_metadata_values(np.str_("patient-1"))
        self.assertEqual(normalized.tolist(), ["patient-1"])

    def test_normalize_batch_metadata_values_wraps_numpy_scalar_integer(self):
        normalized = normalize_batch_metadata_values(np.int64(7))
        self.assertEqual(normalized.tolist(), [7])

    def test_normalize_batch_metadata_values_preserves_vector_input(self):
        normalized = normalize_batch_metadata_values(np.asarray(["a", "b"]))
        self.assertEqual(normalized.tolist(), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
