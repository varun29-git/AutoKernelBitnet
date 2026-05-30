from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ScaffoldTests(unittest.TestCase):
    def test_kernel_catalog_matches_paper_ops(self) -> None:
        expected = {
            "matmul",
            "softmax",
            "layernorm",
            "rmsnorm",
            "flash_attention",
            "fused_mlp",
            "cross_entropy",
            "rotary_embedding",
            "reduce",
        }

        triton_kernels = {
            path.stem
            for path in (ROOT / "kernels").glob("*.py")
            if path.name != "__init__.py"
        }
        cuda_kernels = {
            path.stem
            for path in (ROOT / "kernels" / "cuda").glob("*.py")
            if path.name not in {"__init__.py", "_compile.py"}
        }

        self.assertEqual(expected, triton_kernels)
        self.assertEqual(expected, cuda_kernels)

    def test_kernel_classifier(self) -> None:
        profile = load_script("autokernel_profile_script", ROOT / "profile.py")

        self.assertEqual(profile.classify_kernel("cublasLtMatmulKernel"), "matmul")
        self.assertEqual(profile.classify_kernel("aten::_softmax"), "softmax")
        self.assertEqual(profile.classify_kernel("rms_norm_forward_kernel"), "rmsnorm")
        self.assertEqual(profile.classify_kernel("rope_apply_kernel"), "rotary_embedding")
        self.assertEqual(profile.classify_kernel("unrelated_kernel"), "other")

    def test_profile_loads_dataclass_model_file(self) -> None:
        profile = load_script("autokernel_profile_loader_script", ROOT / "profile.py")

        source = textwrap.dedent(
            """
            from __future__ import annotations

            from dataclasses import dataclass
            import torch.nn as nn


            @dataclass(frozen=True)
            class TinyConfig:
                width: int = 1


            class TinyModel(nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    self.config = TinyConfig()

                def forward(self, x):
                    return x
            """
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny_model.py"
            path.write_text(source, encoding="utf-8")
            model = profile._load_model_from_file(str(path), "TinyModel")

        self.assertEqual(type(model).__name__, "TinyModel")

    def test_amdahl_estimate(self) -> None:
        orchestrate = load_script("autokernel_orchestrate_script", ROOT / "orchestrate.py")

        kernels = [
            {"pct_total": 50.0, "speedup": 2.0},
            {"pct_total": 10.0, "speedup": None},
        ]
        self.assertAlmostEqual(orchestrate.estimate_aggregate_speedup(kernels), 1.0 / 0.75)

    def test_notebook_is_valid_json(self) -> None:
        notebook = ROOT / "notebooks" / "autokernel_h100.ipynb"
        data = json.loads(notebook.read_text(encoding="utf-8"))

        self.assertEqual(data["nbformat"], 4)
        self.assertGreaterEqual(len(data["cells"]), 6)

    def test_test2_bitnet_tiny_forward(self) -> None:
        model_mod = load_script("autokernel_test2_bitnet", ROOT / "models" / "test2_bitnet.py")

        config = model_mod.BitNetLlamaConfig(
            vocab_size=128,
            dim=64,
            n_layers=1,
            n_heads=4,
            n_kv_heads=2,
            ffn_dim=128,
            max_seq_len=32,
        )
        model = model_mod.BitNetLlama(config)
        input_ids = __import__("torch").randint(1, config.vocab_size, (1, 8))
        output = model(input_ids=input_ids, return_logits=False)

        self.assertIn("loss", output)
        self.assertIsNone(output["logits"])


if __name__ == "__main__":
    unittest.main()
