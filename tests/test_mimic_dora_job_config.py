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

    def test_resume_l40s_wrapper_uses_larger_eval_batch_default(self):
        script = (REPO_ROOT / "job" / "submit_mimic_dora_resume1395_to4185.sh").read_text()

        self.assertIn('export DTGPT_VALIDATION_BATCH_SIZE="${DTGPT_VALIDATION_BATCH_SIZE:-8}"', script)
        self.assertIn("#SBATCH --gres=gpu:1", script)
        self.assertNotIn("#SBATCH --exclusive", script)

    def test_mimic_training_uses_validation_batch_for_eval_batch_size(self):
        script = (
            REPO_ROOT
            / "1_experiments"
            / "2024_02_08_mimic_iv"
            / "4_dt_gpt_instruction"
            / "2024_04_11_biomistral_td_bd_summarized_row"
            / "dt_gpt_fft_2024_04_11_biomistral_td_bd_sr.py"
        ).read_text()

        self.assertIn("per_device_eval_batch_size=BATCH_SIZE_VALIDATION", script)
        self.assertNotIn("per_device_eval_batch_size=BATCH_SIZE_TRAINING", script)

    def test_l40s_sharded_eval_wrapper_uses_eager_eval_body_and_1024_tokens(self):
        wrapper = (
            REPO_ROOT / "job" / "submit_mimic_dora_checkpoint700_eval_l40s_array.sh"
        ).read_text()
        body = (REPO_ROOT / "job" / "submit_mimic_dora_checkpoint700_eval_v100.sh").read_text()

        self.assertIn("#SBATCH --partition=l40s", wrapper)
        self.assertIn("#SBATCH --account=l40s", wrapper)
        self.assertIn("#SBATCH --array=0-7%1", wrapper)
        self.assertIn(
            'export DTGPT_EVAL_NUM_SHARDS="${DTGPT_EVAL_NUM_SHARDS:-8}"',
            wrapper,
        )
        self.assertIn(
            'export DTGPT_EVAL_SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${DTGPT_EVAL_SHARD_INDEX:-0}}"',
            wrapper,
        )
        self.assertIn("bash job/submit_mimic_dora_checkpoint700_eval_v100.sh", wrapper)
        self.assertIn(
            'export DTGPT_ATTN_IMPLEMENTATION="${DTGPT_ATTN_IMPLEMENTATION:-eager}"',
            body,
        )
        self.assertIn(
            '--max-new-tokens-to-generate "${DTGPT_MAX_NEW_TOKENS:-1024}"',
            body,
        )

    def test_v100_sharded_eval_wrapper_uses_16_shards_with_two_concurrent_tasks(self):
        wrapper = (
            REPO_ROOT / "job" / "submit_mimic_dora_checkpoint700_eval_v100_array.sh"
        ).read_text()

        self.assertIn("#SBATCH --partition=v100-32g", wrapper)
        self.assertIn("#SBATCH --account=v100-32g", wrapper)
        self.assertIn("#SBATCH --gres=gpu:1", wrapper)
        self.assertIn("#SBATCH --array=0-15%2", wrapper)
        self.assertIn(
            'export DTGPT_EVAL_NUM_SHARDS="${DTGPT_EVAL_NUM_SHARDS:-16}"',
            wrapper,
        )
        self.assertIn(
            'export DTGPT_EVAL_SHARD_INDEX="${SLURM_ARRAY_TASK_ID:-${DTGPT_EVAL_SHARD_INDEX:-0}}"',
            wrapper,
        )
        self.assertIn(
            'export DTGPT_MAX_NEW_TOKENS="${DTGPT_MAX_NEW_TOKENS:-1024}"',
            wrapper,
        )
        self.assertIn("bash job/submit_mimic_dora_checkpoint700_eval_v100.sh", wrapper)

    def test_hf_generation_uses_only_max_new_tokens_length_cap(self):
        experiment_source = (REPO_ROOT / "pipeline" / "Experiment.py").read_text()

        self.assertIn("max_new_tokens=max_new_tokens", experiment_source)
        self.assertNotIn(
            "max_new_tokens=max_new_tokens, max_length=max_output_length + 92",
            experiment_source,
        )


if __name__ == "__main__":
    unittest.main()
