from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_SCRIPT = REPO_ROOT / "job" / "submit_mimic_dora.sh"


class TestMimicDoraJobConfig(unittest.TestCase):

    def test_targets_two_l40s_gpus_by_default(self):
        script = JOB_SCRIPT.read_text()

        self.assertIn("#SBATCH --partition=l40s", script)
        self.assertIn("#SBATCH --account=l40s", script)
        self.assertIn("#SBATCH --gres=gpu:2", script)
        self.assertIn('NPROC_PER_NODE="${DTGPT_NPROC_PER_NODE:-2}"', script)

    def test_enables_dora_unsloth_memory_savers_by_default(self):
        script = JOB_SCRIPT.read_text()

        self.assertIn('USE_DORA="${DTGPT_USE_DORA:-1}"', script)
        self.assertIn('USE_UNSLOTH="${DTGPT_USE_UNSLOTH:-1}"', script)
        self.assertIn('GRADIENT_CHECKPOINTING="${DTGPT_GRADIENT_CHECKPOINTING:-1}"', script)

    def test_uses_half_of_each_patient_split_by_default(self):
        script = JOB_SCRIPT.read_text()

        self.assertIn(
            'export DTGPT_PATIENT_SPLIT_FRACTION="${DTGPT_PATIENT_SPLIT_FRACTION:-0.5}"',
            script,
        )
        self.assertIn('echo "Patient split fraction: ${DTGPT_PATIENT_SPLIT_FRACTION}"', script)
        self.assertIn('RUN_SPLIT_SMOKE_CHECK="${DTGPT_RUN_SPLIT_SMOKE_CHECK:-1}"', script)
        self.assertIn("Running MIMIC split smoke check", script)
        self.assertIn("EvaluationManager('2024_03_15_mimic_iv', load_statistics_file=False)", script)


if __name__ == "__main__":
    unittest.main()
