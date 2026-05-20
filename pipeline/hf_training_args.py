"""Compatibility helpers for Hugging Face training argument API changes."""

import inspect


def normalize_training_argument_kwargs(kwargs, training_arguments_cls):
    """Return kwargs accepted by the installed TrainingArguments class.

    Transformers 5 uses ``eval_strategy`` while older versions used
    ``evaluation_strategy``. Experiment scripts keep the older name for
    readability/history; this helper maps that alias at the boundary.
    """
    normalized_kwargs = dict(kwargs)
    parameters = inspect.signature(training_arguments_cls.__init__).parameters
    has_evaluation_strategy = "evaluation_strategy" in parameters
    has_eval_strategy = "eval_strategy" in parameters

    if "evaluation_strategy" in normalized_kwargs and "eval_strategy" in normalized_kwargs:
        raise ValueError("Both 'evaluation_strategy' and 'eval_strategy' were provided.")

    if "evaluation_strategy" in normalized_kwargs and not has_evaluation_strategy and has_eval_strategy:
        normalized_kwargs["eval_strategy"] = normalized_kwargs.pop("evaluation_strategy")
    elif "eval_strategy" in normalized_kwargs and not has_eval_strategy and has_evaluation_strategy:
        normalized_kwargs["evaluation_strategy"] = normalized_kwargs.pop("eval_strategy")

    return normalized_kwargs


def create_training_arguments(**kwargs):
    """Create ``transformers.TrainingArguments`` with keyword compatibility."""
    from transformers import TrainingArguments

    return TrainingArguments(**normalize_training_argument_kwargs(kwargs, TrainingArguments))
