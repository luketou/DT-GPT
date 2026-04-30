from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_SCRIPT = REPO_ROOT / "job" / "submit_mimic_dora_sweep_v100.sh"


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


if __name__ == "__main__":
    unittest.main()
