import unittest

import numpy as np
import pandas as pd

from pipeline.MetricManager import MetricManager


class MetricManagerCompatibilityTests(unittest.TestCase):
    def test_rmse_and_nrmse_do_not_require_squared_keyword_support(self):
        manager = MetricManager.__new__(MetricManager)
        targets = pd.Series([1.0, 2.0, 4.0])
        predictions = pd.Series([1.0, 4.0, 4.0])

        self.assertAlmostEqual(manager.rmse(targets, predictions), np.sqrt(4.0 / 3.0))
        self.assertAlmostEqual(
            manager.nrmse(targets, predictions),
            np.sqrt(4.0 / 3.0) / np.std(targets),
        )


if __name__ == "__main__":
    unittest.main()
