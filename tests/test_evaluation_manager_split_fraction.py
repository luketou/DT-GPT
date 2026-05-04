import os
import unittest
from unittest.mock import patch

from pipeline.EvaluationManager import apply_patient_split_fraction


class PatientSplitFractionTests(unittest.TestCase):
    def test_unset_fraction_keeps_all_patientids(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(apply_patient_split_fraction([1, 2, 3, 4], "TRAIN"), [1, 2, 3, 4])

    def test_half_fraction_keeps_first_half_of_each_split(self):
        with patch.dict(os.environ, {"DTGPT_PATIENT_SPLIT_FRACTION": "0.5"}, clear=True):
            self.assertEqual(apply_patient_split_fraction([1, 2, 3, 4, 5], "TRAIN"), [1, 2])
            self.assertEqual(apply_patient_split_fraction([10, 11, 12, 13], "VALIDATION"), [10, 11])

    def test_full_fraction_keeps_all_patientids(self):
        with patch.dict(os.environ, {"DTGPT_PATIENT_SPLIT_FRACTION": "1"}, clear=True):
            self.assertEqual(apply_patient_split_fraction([1, 2, 3, 4], "TEST"), [1, 2, 3, 4])

    def test_fraction_must_be_above_zero_and_at_most_one(self):
        for value in ["0", "-0.1", "1.1", "not-a-number"]:
            with self.subTest(value=value):
                with patch.dict(os.environ, {"DTGPT_PATIENT_SPLIT_FRACTION": value}, clear=True):
                    with self.assertRaises(ValueError):
                        apply_patient_split_fraction([1, 2, 3, 4], "TRAIN")


if __name__ == "__main__":
    unittest.main()
