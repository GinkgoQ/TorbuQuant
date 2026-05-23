from __future__ import annotations

import pytest
import torch

from torbuquant.integration.vllm.calibration import (
    ActivationAccumulator,
    ModelShape,
    build_calibrated_metadata,
    derive_model_shape,
    discover_projection_modules,
    load_prompts,
    prompts_sha256,
    resolve_layer_indices,
    select_high_precision_indices,
    validate_calibration_model_choice,
)
from torbuquant.integration.vllm.metadata import CalibrationMetadata


class _Config:
    hidden_size = 32
    num_attention_heads = 4
    num_key_value_heads = 2
    num_hidden_layers = 3
    layer_types = ("full_attention", "sliding_attention", "full_attention")


class _TextConfig:
    text_config = _Config()


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.k_proj = torch.nn.Linear(8, 8)
        self.self_attn.v_proj = torch.nn.Linear(8, 8)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(), _Layer()])


def _calibration() -> CalibrationMetadata:
    return CalibrationMetadata(
        method="activation_energy_v1",
        objective="sum_squared_activation",
        num_prompts=2,
        max_seq_len=16,
        batch_size=1,
        num_observed_tokens=8,
        dtype="float32",
        device="cpu",
        prompts_sha256="abc",
    )


def test_prompt_loader_and_hash(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("\nalpha\n\nbeta\n", encoding="utf-8")

    prompts = load_prompts(prompt_file, limit=1)

    assert prompts == ["alpha"]
    assert prompts_sha256(["alpha", "beta"]) == prompts_sha256(["alpha", "beta"])


def test_model_shape_from_config_and_text_config():
    shape = derive_model_shape(_Config())
    nested = derive_model_shape(_TextConfig())

    assert shape == ModelShape(
        head_dim=8,
        num_kv_heads=2,
        num_hidden_layers=3,
        layer_types=("full_attention", "sliding_attention", "full_attention"),
        quantization_config=None,
    )
    assert nested.head_dim == 8


def test_layer_resolver_keeps_full_attention_layers():
    assert resolve_layer_indices(3, ("full_attention", "sliding_attention", "full_attention")) == [0, 2]


def test_projection_discovery_finds_k_and_v_modules():
    modules = discover_projection_modules(_Model(), [0, 1])

    assert sorted(modules) == [(0, "key"), (0, "value"), (1, "key"), (1, "value")]


def test_calibration_model_guard_rejects_quantized_target():
    with pytest.raises(ValueError, match="non-quantized"):
        validate_calibration_model_choice(
            target_model="m",
            calibration_model="m",
            quantization_config={"quant_method": "gptq"},
        )


def test_high_precision_selection_is_sorted_and_deterministic():
    scores = torch.tensor([[1.0, 3.0, 3.0, 2.0], [4.0, 1.0, 2.0, 3.0]])

    selected = select_high_precision_indices(scores, 2)

    assert selected == ((1, 2), (0, 3))


def test_activation_accumulator_uses_attention_mask():
    projection = torch.nn.Linear(8, 8, bias=False)
    with torch.no_grad():
        projection.weight.copy_(torch.eye(8))
    accumulator = ActivationAccumulator(num_kv_heads=2, head_dim=4)
    accumulator.set_attention_mask(torch.tensor([[1, 0, 1]], dtype=torch.long))
    handle = projection.register_forward_hook(accumulator.hook(0, "key"))
    x = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    projection(x)
    handle.remove()

    score = accumulator.channel_scores[(0, "key")]
    expected = x[:, [0, 2]].reshape(-1, 2, 4).square().sum(dim=0)
    torch.testing.assert_close(score, expected)


def test_metadata_builder_from_scores_uses_recipe_group_count():
    scores = {
        (0, "key"): torch.arange(256, dtype=torch.float32).reshape(2, 128),
        (0, "value"): torch.arange(256, dtype=torch.float32).reshape(2, 128),
    }

    metadata = build_calibrated_metadata(
        recipe="turboquant25",
        head_dim=128,
        model_name="qwen-test",
        num_hidden_layers=1,
        layer_types=None,
        layer_pattern="model.layers.{i}.self_attn",
        num_kv_heads=2,
        calibration_scores=scores,
        calibration=_calibration(),
    )

    layer = metadata.get_layer("model.layers.0.self_attn")
    assert len(layer.key.high_precision_indices) == 2
    assert len(layer.key.high_precision_indices[0]) == 32
