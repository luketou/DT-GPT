import json
import unittest

import pandas as pd

from pipeline.data_generators.DataFrameConvertTDBDMIMIC import (
    DTGPTDataFrameConverterTemplateTextBasicDescriptionMIMIC,
)


class ConvertFromStringsToDfTests(unittest.TestCase):
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
                "2026-04-19 00:00:00",
                "2026-04-19 01:00:00",
                "2026-04-19 02:00:00",
            ]
        )
        self.all_column_names = [
            "lab_hr",
            "lab_rr",
            "date",
            "patientid",
            "patient_sample_index",
        ]

    def test_short_column_is_padded_to_prediction_horizon(self):
        prediction = json.dumps(
            {
                "Heart Rate": ["80", "82"],
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
        self.assertEqual(result.loc[0, "lab_hr"], "80")
        self.assertEqual(result.loc[1, "lab_hr"], "82")
        self.assertTrue(pd.isna(result.loc[2, "lab_hr"]))

    def test_long_column_is_truncated_to_prediction_horizon(self):
        prediction = json.dumps(
            {
                "Heart Rate": ["80", "82", "84", "86"],
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
        self.assertEqual(result["lab_hr"].tolist(), ["80", "82", "84"])

    def test_sparse_position_mapping_keeps_declared_hours(self):
        prediction = json.dumps(
            {
                "Heart Rate": ["80", "84"],
                "Respiratory Rate": ["18", "19", "20"],
            }
        )
        prediction_days_column_wise = {
            "Variables to predict for respective hours": {
                "Heart Rate": [1, 3],
                "Respiratory Rate": [1, 2, 3],
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


if __name__ == "__main__":
    unittest.main()
