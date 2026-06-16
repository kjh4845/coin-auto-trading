from __future__ import annotations

import unittest

from ai_auto_trading.ai.local_server import _single_runtime_device


class _ModelWithMap:
    hf_device_map = {
        "embed_tokens": "mps",
        "layers.0": "mps",
        "layers.1": "cpu",
        "lm_head": "cpu",
    }


class _ModelWithCpuOnlyMap:
    hf_device_map = {
        "embed_tokens": "cpu",
        "lm_head": "disk",
    }


class _ModelWithDevice:
    device = "mps"


class LocalServerTest(unittest.TestCase):
    def test_single_runtime_device_prefers_accelerator_from_device_map(self) -> None:
        self.assertEqual(_single_runtime_device(_ModelWithMap()), "mps")

    def test_single_runtime_device_ignores_cpu_only_device_map(self) -> None:
        self.assertIsNone(_single_runtime_device(_ModelWithCpuOnlyMap()))

    def test_single_runtime_device_uses_model_device_without_map(self) -> None:
        self.assertEqual(_single_runtime_device(_ModelWithDevice()), "mps")


if __name__ == "__main__":
    unittest.main()
