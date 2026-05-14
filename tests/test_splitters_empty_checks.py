import unittest

import pandas as pd

from pipeline.Splitters import After24HSplitter


class _EvalManagerStub:
    def __init__(self, patientids):
        self._current_master_constants_table = pd.DataFrame(
            [{"patientid": patientid} for patientid in patientids]
        )

    def get_column_usage(self):
        return (
            ["target_lab"],
            [],
            ["target_lab"],
        )


class After24HSplitterEmptyCheckTests(unittest.TestCase):
    def test_zero_and_nan_targets_are_empty_when_zeros_count_as_nans(self):
        events = pd.DataFrame(
            [
                {
                    "date": "2024-01-01 01:00:00",
                    "patientid": "p1",
                    "target_lab": 1.0,
                },
                {
                    "date": "2024-01-02 01:00:00",
                    "patientid": "p1",
                    "target_lab": 0.0,
                },
                {
                    "date": "2024-01-02 02:00:00",
                    "patientid": "p1",
                    "target_lab": None,
                },
            ]
        )

        split_events, meta_data = After24HSplitter().setup_split_indices(
            [events],
            _EvalManagerStub(["p1"]),
        )

        self.assertEqual(split_events, [])
        self.assertEqual(meta_data, [])

    def test_skipped_patients_are_logged_as_summary(self):
        skipped_events = pd.DataFrame(
            [
                {
                    "date": "2024-01-01 01:00:00",
                    "patientid": "p1",
                    "target_lab": 1.0,
                },
                {
                    "date": "2024-01-02 01:00:00",
                    "patientid": "p1",
                    "target_lab": None,
                },
            ]
        )
        kept_events = pd.DataFrame(
            [
                {
                    "date": "2024-01-01 01:00:00",
                    "patientid": "p2",
                    "target_lab": 1.0,
                },
                {
                    "date": "2024-01-02 01:00:00",
                    "patientid": "p2",
                    "target_lab": 2.0,
                },
            ]
        )

        with self.assertLogs(level="INFO") as log_context:
            split_events, meta_data = After24HSplitter().setup_split_indices(
                [skipped_events, kept_events],
                _EvalManagerStub(["p1", "p2"]),
            )

        self.assertEqual(len(split_events), 1)
        self.assertEqual(len(meta_data), 1)
        self.assertTrue(
            any("24H Splitter skipped 1 / 2 patients" in message for message in log_context.output)
        )
        self.assertFalse(
            any("Skipping patient due to empty input or output" in message for message in log_context.output)
        )


if __name__ == "__main__":
    unittest.main()
