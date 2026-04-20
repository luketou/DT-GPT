import json
import unittest
import warnings
from unittest.mock import patch

import pandas as pd

with patch("transformers.AutoTokenizer.from_pretrained", return_value=object()):
    from pipeline.data_generators.DataFrameConvertTDBDMIMIC import (
        DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC,
    )


class MimicPredictionParsingTests(unittest.TestCase):
    def setUp(self):
        self.column_name_mapping = pd.DataFrame(
            {
                "descriptive_column_name": [
                    "Heart Rate",
                    "Respiratory Rate",
                ],
                "original_column_names": [
                    "lab_hr",
                    "lab_rr",
                ],
            }
        )
        self.all_prediction_days = pd.to_datetime(
            [
                "2026-04-20 00:00:00",
                "2026-04-20 01:00:00",
                "2026-04-20 02:00:00",
            ]
        )
        self.all_column_names = [
            "lab_hr",
            "lab_rr",
            "date",
            "patientid",
            "patient_sample_index",
        ]

    def test_mismatched_column_lengths_are_normalized_to_prediction_horizon(self):
        prediction = json.dumps(
            {
                "Heart Rate": ["80", "82"],
                "Respiratory Rate": ["18", "19", "20", "21"],
            }
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = (
                DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC.convert_from_strings_to_df(
                    self.column_name_mapping,
                    prediction,
                    self.all_prediction_days,
                    patientid=55,
                    patient_sample_index="split_0",
                    all_column_names=self.all_column_names,
                    all_unk_columns=[],
                )
            )

        self.assertEqual(len(result), 3)
        self.assertEqual(result.loc[0, "lab_hr"], "80")
        self.assertEqual(result.loc[1, "lab_hr"], "82")
        self.assertTrue(pd.isna(result.loc[2, "lab_hr"]))
        self.assertEqual(result["lab_rr"].tolist(), ["18", "19", "20"])
        self.assertFalse(
            any(isinstance(w.message, FutureWarning) for w in caught),
            msg=[str(w.message) for w in caught],
        )

    def test_sparse_prediction_hours_expand_to_the_full_output_horizon(self):
        prediction = json.dumps(
            {
                "Heart Rate": ["80", "84"],
                "Respiratory Rate": ["18", "19", "20"],
            }
        )
        prediction_days_column_wise = {
            "Variables to predict for respective hours": {
                "Heart Rate": [24, 72],
                "Respiratory Rate": [24, 48, 72],
            }
        }

        result = (
            DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC.convert_from_strings_to_df(
                self.column_name_mapping,
                prediction,
                self.all_prediction_days,
                patientid=55,
                patient_sample_index="split_0",
                all_column_names=self.all_column_names,
                all_unk_columns=[],
                prediction_days_column_wise=prediction_days_column_wise,
            )
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(result.loc[0, "lab_hr"], "80")
        self.assertTrue(pd.isna(result.loc[1, "lab_hr"]))
        self.assertEqual(result.loc[2, "lab_hr"], "84")

    def test_missing_prediction_column_becomes_all_nan_instead_of_crashing(self):
        prediction = json.dumps(
            {
                "Respiratory Rate": ["18", "19", "20"],
            }
        )

        result = (
            DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC.convert_from_strings_to_df(
                self.column_name_mapping,
                prediction,
                self.all_prediction_days,
                patientid=55,
                patient_sample_index="split_0",
                all_column_names=self.all_column_names,
                all_unk_columns=[],
            )
        )

        self.assertEqual(len(result), 3)
        self.assertTrue(pd.isna(result.loc[:, "lab_hr"]).all())
        self.assertEqual(result["lab_rr"].tolist(), ["18", "19", "20"])


if __name__ == "__main__":
    unittest.main()
