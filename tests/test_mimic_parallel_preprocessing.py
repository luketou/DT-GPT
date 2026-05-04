import importlib.util
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = (
    REPO_ROOT
    / "1_experiments"
    / "2024_02_08_mimic_iv"
    / "1_data"
    / "1_preprocessing"
)
BUILD_JOB_SCRIPT = REPO_ROOT / "job" / "submit_mimic_build_final_data.sh"


def load_script_module(filename, module_name):
    sys.modules.setdefault(
        "wandb",
        types.SimpleNamespace(init=lambda *args, **kwargs: None),
    )
    path = PREPROCESSING_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MimicParallelPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stats_module = load_script_module(
            "2024_02_01_overall_stats_generation.py",
            "mimic_stats_generation",
        )
        cls.filter_module = load_script_module(
            "2024_03_13_filter_columns.py",
            "mimic_filter_columns",
        )

    def test_resolve_worker_count_uses_slurm_cpu_count(self):
        env = {"SLURM_CPUS_PER_TASK": "8"}
        self.assertEqual(self.stats_module.resolve_worker_count(env=env), 8)
        self.assertEqual(self.filter_module.resolve_worker_count(env=env), 8)

    def test_collect_and_merge_stay_stats_preserves_existing_counts(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            stay_dir = raw_dir / "stay_a"
            stay_dir.mkdir()
            (stay_dir / "dynamic.csv").write_text(
                "ignored metadata\n220635,123\n0,5\n2,0\n"
            )
            (stay_dir / "static.csv").write_text("ignored metadata\nA01\n1\n")

            stay_stats = self.stats_module.collect_stay_stats(("stay_a", str(raw_dir)))
            helper_items = pd.DataFrame(
                {
                    "itemid": ["220635", "123"],
                    "label": ["Magnesium", "Other"],
                    "linksto": ["chartevents", "inputevents"],
                    "category": ["Labs", "Inputs"],
                }
            )
            diagnoses = pd.DataFrame(
                {"icd_code": ["A01"], "long_title": ["Typhoid fever"]}
            )

            merged = self.stats_module.merge_stats(
                [stay_stats],
                helper_items,
                diagnoses,
            )

        self.assertEqual(merged["220635"]["non_na"], 2)
        self.assertEqual(merged["220635"]["non_zero"], 1)
        self.assertEqual(merged["220635"]["non_na_or_zero"], 1)
        self.assertEqual(merged["220635"]["label"], "Magnesium")
        self.assertEqual(merged["A01"]["label"], "Typhoid fever")
        self.assertEqual(merged["A01"]["linksto"], "COND")

    def test_process_stay_for_final_data_returns_filtered_frames(self):
        with TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir)
            stay_dir = raw_dir / "stay_a"
            stay_dir.mkdir()
            (stay_dir / "dynamic.csv").write_text(
                "ignored metadata\n220635,220210,drop_me\n0,10,99\n4,0,99\n"
            )
            (stay_dir / "demo.csv").write_text("age,gender\n70,F\n")
            (stay_dir / "static.csv").write_text("ignored metadata\nA01,B02\n1,0\n")

            stay_name, event_df, constant_df = (
                self.filter_module.process_stay_for_final_data(
                    (
                        "stay_a",
                        str(raw_dir),
                        ["220635", "220210"],
                        ["A01"],
                        ["220635", "A01"],
                    )
                )
            )

        self.assertEqual(stay_name, "stay_a")
        self.assertEqual(list(event_df.columns), ["220635", "220210"])
        self.assertTrue(np.isnan(event_df.loc[0, "220635"]))
        self.assertEqual(event_df.loc[1, "220635"], 4)
        self.assertEqual(list(constant_df.columns), ["age", "gender", "A01"])
        self.assertEqual(constant_df.loc[0, "A01"], "diagnosed")

    def test_build_final_data_job_targets_cpu_parallel_workers(self):
        script = BUILD_JOB_SCRIPT.read_text()

        self.assertIn("#SBATCH --partition=cpu-2g", script)
        self.assertIn("#SBATCH --cpus-per-task=32", script)
        self.assertNotIn("--gres=gpu", script)
        self.assertIn(
            'export DTGPT_MIMIC_NUM_WORKERS="${DTGPT_MIMIC_NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-32}}"',
            script,
        )
        self.assertIn('export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"', script)


if __name__ == "__main__":
    unittest.main()
