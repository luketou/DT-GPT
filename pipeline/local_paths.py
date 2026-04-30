import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MIMIC_EXPERIMENT_DIR = REPO_ROOT / "1_experiments" / "2024_02_08_mimic_iv"
MIMIC_DATA_DIR = Path(
    os.getenv("DTGPT_MIMIC_DATA_ROOT", str(MIMIC_EXPERIMENT_DIR / "1_data"))
).expanduser().resolve()
MIMIC_PREPROCESSING_DIR = MIMIC_DATA_DIR / "1_preprocessing"
MIMIC_POSTPROCESS_DIR = MIMIC_DATA_DIR / "2_data_setup"
MIMIC_FINAL_DATA_DIR = MIMIC_DATA_DIR / "0_final_data"
MIMIC_FINAL_EVENTS_DIR = MIMIC_FINAL_DATA_DIR / "events"
MIMIC_PATIENT_SUBSETS_DIR = MIMIC_FINAL_DATA_DIR / "patient_subsets"
MIMIC_DEMO_DATA_DIR = REPO_ROOT / "mimic-iv-clinical-database-demo-2.2"
MIMIC_RAW_EVENTS_DIR = MIMIC_PREPROCESSING_DIR / "1_raw_events" / "csv"
MIMIC_RAW_HELPER_DIR = MIMIC_PREPROCESSING_DIR / "0_raw_helper_files"
MIMIC_RAW_STATS_PATH = MIMIC_PREPROCESSING_DIR / "2024_02_01_raw_data_stats.json"
DEFAULT_REPO_RESULTS_DIR = REPO_ROOT / "3_results" / "raw_experiments" / "DT-GPT"


def ensure_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def as_posix_str(path):
    return str(Path(path).expanduser().resolve())


def repo_root():
    return REPO_ROOT


def get_mimic_preprocessing_dir():
    return MIMIC_PREPROCESSING_DIR


def get_mimic_postprocess_dir():
    return MIMIC_POSTPROCESS_DIR


def get_mimic_final_data_dir():
    return MIMIC_FINAL_DATA_DIR


def get_mimic_final_events_dir():
    return MIMIC_FINAL_EVENTS_DIR


def get_mimic_patient_subsets_dir():
    return MIMIC_PATIENT_SUBSETS_DIR


def get_mimic_dataset_statistics_path():
    return MIMIC_FINAL_DATA_DIR / "dataset_statistics.json"


def get_mimic_column_mapping_json_path():
    return MIMIC_FINAL_DATA_DIR / "column_mapping.json"


def get_mimic_column_descriptive_mapping_path():
    return MIMIC_FINAL_DATA_DIR / "column_descriptive_name_mapping.csv"


def get_mimic_constants_path():
    return MIMIC_FINAL_DATA_DIR / "constants.csv"


def get_mimic_raw_events_dir():
    return Path(
        os.getenv("DTGPT_MIMIC_RAW_EVENTS_DIR", str(MIMIC_RAW_EVENTS_DIR))
    ).expanduser().resolve()


def get_mimic_raw_stats_path():
    return Path(
        os.getenv("DTGPT_MIMIC_RAW_STATS_PATH", str(MIMIC_RAW_STATS_PATH))
    ).expanduser().resolve()


def get_mimic_demo_data_dir():
    return Path(
        os.getenv("DTGPT_MIMIC_DEMO_DATA_DIR", str(MIMIC_DEMO_DATA_DIR))
    ).expanduser().resolve()


def get_mimic_helper_items_path():
    explicit = MIMIC_RAW_HELPER_DIR / "d_items.csv"
    if explicit.exists():
        return explicit
    demo_items = get_mimic_demo_data_dir() / "icu" / "d_items.csv.gz"
    return demo_items


def get_mimic_helper_diagnosis_path():
    explicit = MIMIC_RAW_HELPER_DIR / "d_icd_diagnoses.csv"
    if explicit.exists():
        return explicit
    demo_diagnosis = get_mimic_demo_data_dir() / "hosp" / "d_icd_diagnoses.csv.gz"
    return demo_diagnosis


def get_mimic_external_pipeline_root():
    env_value = os.getenv("DTGPT_MIMIC_PIPELINE_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return None


def get_default_experiment_output_root():
    env_value = os.getenv("DTGPT_EXPERIMENT_ROOT")
    if env_value:
        return as_posix_str(ensure_directory(Path(env_value).expanduser()))
    return as_posix_str(ensure_directory(DEFAULT_REPO_RESULTS_DIR)) + "/"


def resolve_biomistral_model_path(env=None, local_candidate_exists=None):
    env = os.environ if env is None else env

    env_override = env.get("DTGPT_BIOMISTRAL_MODEL_PATH")
    if env_override:
        return env_override

    local_candidate = Path.home() / "llm_model" / "BioMistral-7B-DARE"
    if local_candidate_exists is None:
        local_candidate_exists = local_candidate.exists()

    if local_candidate_exists:
        return str(local_candidate)

    return "BioMistral/BioMistral-7B-DARE"


def get_biomistral_model_path():
    return resolve_biomistral_model_path()


def resolve_tokenizer_model_path(env=None, local_candidate_exists=None):
    env = os.environ if env is None else env

    env_override = env.get("DTGPT_TOKENIZER_MODEL_PATH")
    if env_override:
        return env_override

    return resolve_biomistral_model_path(
        env=env,
        local_candidate_exists=local_candidate_exists,
    )


def get_tokenizer_model_path():
    return resolve_tokenizer_model_path()


def ensure_runtime_cache_env(env=None):
    env = os.environ if env is None else env

    cache_root = Path(env.get("DTGPT_RUNTIME_CACHE_ROOT", "/tmp/dtgpt_runtime_cache"))
    hf_home = ensure_directory(cache_root / "hf_home")
    triton_cache = ensure_directory(cache_root / "triton")
    mpl_config = ensure_directory(cache_root / "matplotlib")

    env.setdefault("HF_HOME", str(hf_home))
    env.pop("TRANSFORMERS_CACHE", None)
    env.setdefault("TRITON_CACHE_DIR", str(triton_cache))
    env.setdefault("MPLCONFIGDIR", str(mpl_config))

    return {
        "HF_HOME": env["HF_HOME"],
        "TRITON_CACHE_DIR": env["TRITON_CACHE_DIR"],
        "MPLCONFIGDIR": env["MPLCONFIGDIR"],
    }


def select_precision_config(cuda_available, capability_major, training=False):
    if not cuda_available:
        return {
            "torch_dtype_name": "float32",
            "bf16": False,
            "fp16": False,
            "attn_implementation": "eager",
        }

    if capability_major is not None and capability_major >= 8:
        return {
            "torch_dtype_name": "bfloat16",
            "bf16": True,
            "fp16": False,
            "attn_implementation": "flash_attention_2",
        }

    if training:
        return {
            "torch_dtype_name": "float32",
            "bf16": False,
            "fp16": True,
            "attn_implementation": "eager",
        }

    return {
        "torch_dtype_name": "float16",
        "bf16": False,
        "fp16": True,
        "attn_implementation": "eager",
    }


def get_precision_config(training=False):
    import torch

    if not torch.cuda.is_available():
        return select_precision_config(
            cuda_available=False,
            capability_major=None,
            training=training,
        )

    major, _minor = torch.cuda.get_device_capability()
    return select_precision_config(
        cuda_available=True,
        capability_major=major,
        training=training,
    )


def get_torch_dtype(dtype_name):
    import torch

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def is_unsloth_available():
    """Return *True* when the ``unsloth`` package is importable."""
    try:
        import unsloth  # noqa: F401
        return True
    except ImportError:
        return False


def get_model_load_kwargs(cache_dir, device_map="auto", training=False):
    precision = get_precision_config(training=training)
    if training and device_map == "auto":
        device_map = None

    kwargs = {
        "cache_dir": cache_dir,
        "torch_dtype": get_torch_dtype(precision["torch_dtype_name"]),
    }
    if device_map is not None:
        kwargs["device_map"] = device_map

    attn_implementation = precision["attn_implementation"]
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    return kwargs
