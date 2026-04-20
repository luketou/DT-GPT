import unittest
import warnings

import numpy as np
import pandas as pd

from pipeline.prediction_aggregation import (
    aggregate_prediction_cube,
    build_prediction_cube,
)


class PredictionAggregationTests(unittest.TestCase):
    def test_build_prediction_cube_coerces_pandas_na_to_nan(self):
        prediction_frames = [
            pd.DataFrame(
                {
                    "lab_hr": [80, pd.NA],
                    "lab_rr": ["18", "19"],
                }
            ),
            pd.DataFrame(
                {
                    "lab_hr": ["82", "84"],
                    "lab_rr": [pd.NA, "21"],
                }
            ),
        ]

        cube = build_prediction_cube(prediction_frames, ["lab_hr", "lab_rr"])

        self.assertEqual(cube.dtype, np.float32)
        self.assertEqual(cube.shape, (2, 2, 2))
        self.assertEqual(cube[0, 0, 0], 80.0)
        self.assertTrue(np.isnan(cube[1, 0, 0]))
        self.assertTrue(np.isnan(cube[0, 1, 1]))
        self.assertEqual(cube[1, 1, 1], 21.0)

    def test_mean_aggregation_preserves_nan_without_empty_slice_warning(self):
        cube = np.array(
            [
                [[np.nan, np.nan]],
                [[1.0, np.nan]],
            ],
            dtype=np.float32,
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            aggregated = aggregate_prediction_cube(cube, "mean")

        self.assertTrue(np.isnan(aggregated[0, 0]))
        self.assertEqual(aggregated[1, 0], 1.0)
        self.assertFalse(
            any("Mean of empty slice" in str(w.message) for w in caught),
            msg=[str(w.message) for w in caught],
        )

    def test_percentile_strategy_keeps_existing_behavior_for_non_nan_values(self):
        cube = np.array(
            [
                [[2.0, 4.0, 6.0]],
            ],
            dtype=np.float32,
        )

        aggregated = aggregate_prediction_cube(cube, "50th percentile")
        self.assertEqual(aggregated[0, 0], 4.0)


if __name__ == "__main__":
    unittest.main()
