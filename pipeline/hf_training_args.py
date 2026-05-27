"""Compatibility helpers for Hugging Face training argument API changes."""

import inspect


def normalize_training_argument_kwargs(kwargs, training_arguments_cls):
    """Return kwargs accepted by the installed TrainingArguments class.

    Transformers 5 uses ``eval_strategy`` while older versions used
    ``evaluation_strategy``. Experiment scripts keep the older name for
    readability/history; this helper maps that alias at the boundary.
    Transformers 5 also replaced ``group_by_length=True`` with
    ``train_sampling_strategy="group_by_length"``.
    """
    normalized_kwargs = dict(kwargs)
    parameters = inspect.signature(training_arguments_cls.__init__).parameters
    has_evaluation_strategy = "evaluation_strategy" in parameters
    has_eval_strategy = "eval_strategy" in parameters
    has_group_by_length = "group_by_length" in parameters
    has_train_sampling_strategy = "train_sampling_strategy" in parameters

    if "evaluation_strategy" in normalized_kwargs and "eval_strategy" in normalized_kwargs:
        raise ValueError("Both 'evaluation_strategy' and 'eval_strategy' were provided.")

    if "evaluation_strategy" in normalized_kwargs and not has_evaluation_strategy and has_eval_strategy:
        normalized_kwargs["eval_strategy"] = normalized_kwargs.pop("evaluation_strategy")
    elif "eval_strategy" in normalized_kwargs and not has_eval_strategy and has_evaluation_strategy:
        normalized_kwargs["evaluation_strategy"] = normalized_kwargs.pop("eval_strategy")

    if "group_by_length" in normalized_kwargs and "train_sampling_strategy" in normalized_kwargs:
        raise ValueError("Both 'group_by_length' and 'train_sampling_strategy' were provided.")

    if "group_by_length" in normalized_kwargs and not has_group_by_length and has_train_sampling_strategy:
        group_by_length = normalized_kwargs.pop("group_by_length")
        if group_by_length:
            normalized_kwargs["train_sampling_strategy"] = "group_by_length"
    elif (
        "train_sampling_strategy" in normalized_kwargs
        and not has_train_sampling_strategy
        and has_group_by_length
    ):
        train_sampling_strategy = normalized_kwargs.pop("train_sampling_strategy")
        if train_sampling_strategy == "group_by_length":
            normalized_kwargs["group_by_length"] = True
        elif train_sampling_strategy != "random":
            raise ValueError(
                "Installed TrainingArguments does not support "
                f"train_sampling_strategy={train_sampling_strategy!r}."
            )

    return normalized_kwargs


def create_training_arguments(**kwargs):
    """Create ``transformers.TrainingArguments`` with keyword compatibility."""
    from transformers import TrainingArguments

    return TrainingArguments(**normalize_training_argument_kwargs(kwargs, TrainingArguments))
