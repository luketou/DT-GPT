import torch


_WRAPPED_MODEL_ATTRIBUTE_NAMES = ("base_model", "model")
_PREFERRED_EMBEDDING_SUFFIXES = (
    "embed_tokens",
    "tok_embeddings",
    "word_embeddings",
    "wte",
    "embed_in",
)


def _iter_model_chain(model):
    current_model = model
    seen_ids = set()

    while current_model is not None and id(current_model) not in seen_ids:
        yield current_model
        seen_ids.add(id(current_model))

        next_model = None
        for attr_name in _WRAPPED_MODEL_ATTRIBUTE_NAMES:
            candidate = getattr(current_model, attr_name, None)
            if candidate is not None and candidate is not current_model:
                next_model = candidate
                break
        current_model = next_model


def _normalize_device(device_spec):
    if isinstance(device_spec, torch.device):
        return device_spec

    if isinstance(device_spec, int):
        return torch.device(f"cuda:{device_spec}")

    if isinstance(device_spec, str):
        if device_spec.isdigit():
            return torch.device(f"cuda:{device_spec}")
        if device_spec == "disk":
            return None
        return torch.device(device_spec)

    return None


def _get_hf_device_map(model):
    for candidate in _iter_model_chain(model):
        hf_device_map = getattr(candidate, "hf_device_map", None)
        if hf_device_map:
            return hf_device_map
    return None


def model_uses_hf_device_map(model):
    return _get_hf_device_map(model) is not None


def _select_device_from_hf_map(hf_device_map):
    preferred_devices = []
    fallback_devices = []

    for module_name, device_spec in hf_device_map.items():
        normalized_device = _normalize_device(device_spec)
        if normalized_device is None:
            continue

        fallback_devices.append(normalized_device)
        if str(module_name).endswith(_PREFERRED_EMBEDDING_SUFFIXES):
            preferred_devices.append(normalized_device)

    for device in preferred_devices + fallback_devices:
        if device.type != "cpu":
            return device

    if preferred_devices:
        return preferred_devices[0]
    if fallback_devices:
        return fallback_devices[0]
    return torch.device("cpu")


def get_generation_input_device(model):
    hf_device_map = _get_hf_device_map(model)
    if hf_device_map:
        return _select_device_from_hf_map(hf_device_map)

    for candidate in _iter_model_chain(model):
        candidate_device = _normalize_device(getattr(candidate, "device", None))
        if candidate_device is not None:
            return candidate_device

        parameters = getattr(candidate, "parameters", None)
        if parameters is None:
            continue

        try:
            first_parameter = next(parameters())
            return first_parameter.device
        except (StopIteration, TypeError):
            continue

    return torch.device("cpu")
