import unittest

import pandas as pd

from pipeline.Splitters import After24HSplitter


class _EvalManagerStub:
    def __init__(self):
        self._current_master_constants_table = pd.DataFrame(
            [{"patientid": "p1", "constant": "value"}]
        )

    def get_column_usage(self):
        return (
            ["present_input", "present_target", "missing_input"],
            ["missing_future_known"],
            ["present_target"],
        )


class After24HSplitterMissingColumnsTests(unittest.TestCase):
    def test_missing_declared_event_columns_are_treated_as_empty_columns(self):
        events = pd.DataFrame(
            [
                {
                    "date": "2024-01-01 01:00:00",
                    "patientid": "p1",
                    "present_input": 1.0,
                    "present_target": 2.0,
                },
                {
                    "date": "2024-01-02 01:00:00",
                    "patientid": "p1",
                    "present_input": 3.0,
                    "present_target": 4.0,
                },
            ]
        )

        split_events, meta_data = After24HSplitter().setup_split_indices(
            [events],
            _EvalManagerStub(),
        )

        self.assertEqual(len(split_events), 1)
        self.assertEqual(len(meta_data), 1)
        _, true_events_input, true_future_events_input, target_dataframe = split_events[0]
        self.assertIn("missing_input", true_events_input.columns)
        self.assertTrue(true_events_input["missing_input"].isna().all())
        self.assertIn("missing_future_known", true_future_events_input.columns)
        self.assertTrue(true_future_events_input["missing_future_known"].isna().all())
        self.assertEqual(target_dataframe["present_target"].tolist(), [4.0])


if __name__ == "__main__":
    unittest.main()
