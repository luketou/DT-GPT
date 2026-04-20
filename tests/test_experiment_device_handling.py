import unittest

import torch

from pipeline.model_device import (
    get_generation_input_device,
    model_uses_hf_device_map,
)


class _LeafModel:
    def __init__(self, hf_device_map=None, device=torch.device("cpu")):
        self.hf_device_map = hf_device_map
        self.device = device


class _WrappedModel:
    def __init__(self, wrapped):
        self.base_model = wrapped


class ExperimentDeviceHandlingTests(unittest.TestCase):
    def test_model_uses_hf_device_map_for_wrapped_model(self):
        model = _WrappedModel(_LeafModel(hf_device_map={"model.embed_tokens": "cuda:1"}))
        self.assertTrue(model_uses_hf_device_map(model))

    def test_get_generation_input_device_prefers_embed_tokens_device(self):
        model = _LeafModel(
            hf_device_map={
                "lm_head": "cuda:0",
                "model.embed_tokens": "cuda:1",
            }
        )
        self.assertEqual(
            get_generation_input_device(model),
            torch.device("cuda:1"),
        )

    def test_get_generation_input_device_supports_integer_cuda_ids(self):
        model = _LeafModel(hf_device_map={"model.embed_tokens": 2})
        self.assertEqual(
            get_generation_input_device(model),
            torch.device("cuda:2"),
        )

    def test_get_generation_input_device_falls_back_to_model_device(self):
        model = _LeafModel(device=torch.device("cpu"))
        self.assertEqual(
            get_generation_input_device(model),
            torch.device("cpu"),
        )


if __name__ == "__main__":
    unittest.main()
