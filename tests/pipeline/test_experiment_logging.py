import logging
import tempfile
import unittest

from pipeline.Experiment import Experiment


class ExperimentLoggingTests(unittest.TestCase):
    def test_setup_logging_defaults_to_info_and_quiets_noisy_libraries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Experiment(
                "logging_test",
                experiment_folder_root=f"{tmpdir}/",
                timestamp_to_use="2026_04_19___00_00_00_000000",
            )

        self.assertEqual(logging.getLogger().level, logging.INFO)
        self.assertEqual(logging.getLogger("asyncio").level, logging.WARNING)
        self.assertEqual(logging.getLogger("filelock").level, logging.WARNING)
        self.assertEqual(logging.getLogger("urllib3").level, logging.WARNING)
        self.assertEqual(logging.getLogger("git").level, logging.WARNING)
        self.assertEqual(logging.getLogger("wandb").level, logging.WARNING)


if __name__ == "__main__":
    unittest.main()
