import unittest

import pandas as pd

from pipeline.Experiment import align_prediction_to_target_shape


class PredictionAlignmentTests(unittest.TestCase):
    def test_missing_prediction_cells_do_not_copy_target_values(self):
        target_df = pd.DataFrame(
            {
                "patientid": ["p1", "p1", "p1"],
                "patient_sample_index": ["s0", "s0", "s0"],
                "date": [1, 2, 3],
                "220210": [10.0, 20.0, 30.0],
                "220277": [90.0, 91.0, 92.0],
            }
        )
        predicted_df = pd.DataFrame({"220210": [11.0]})

        aligned = align_prediction_to_target_shape(predicted_df, target_df, ["220210", "220277"])

        self.assertEqual(aligned["patientid"].tolist(), ["p1", "p1", "p1"])
        self.assertEqual(aligned["date"].tolist(), [1, 2, 3])
        self.assertEqual(aligned["220210"].tolist()[0], 11.0)
        self.assertTrue(pd.isna(aligned.loc[1, "220210"]))
        self.assertTrue(pd.isna(aligned.loc[2, "220210"]))
        self.assertTrue(aligned["220277"].isna().all())


if __name__ == "__main__":
    unittest.main()
