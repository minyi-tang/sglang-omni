# SPDX-License-Identifier: Apache-2.0
"""bsgemm-moe: block-FP8 experts only, and only the checkpoint's exact format."""

from __future__ import annotations

import pytest
import torch
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

from sglang_omni.models.fishaudio_drama3.quantization import ExpertFp8Config
from tests.unit_test.fishaudio_drama3.mtp_fixtures import (
    FP8_BLOCK_AMAX_FLOOR,
    FP8_MAX,
    block_expanded,
    dequantize_block_fp8,
    quantize_block_fp8,
)

CHECKPOINT_QUANTIZATION = {
    "quant_method": "bsgemm-moe",
    "weight_block_size": [128, 128],
    "activation_scheme": "dynamic",
}


class ExpertLayer(FusedMoE):
    """FusedMoE for isinstance dispatch; its own __init__ needs an SGLang runtime."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)


class ProjectionLayer(LinearBase):
    """LinearBase for isinstance dispatch, without building weights."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)


def test_only_experts_get_block_fp8() -> None:
    config = ExpertFp8Config.from_config(CHECKPOINT_QUANTIZATION)

    expert_method = config.get_quant_method(ExpertLayer(), "model.layers.0.mlp.experts")
    projection_method = config.get_quant_method(
        ProjectionLayer(), "model.layers.0.mlp.shared_expert.gate_up_proj"
    )
    other_method = config.get_quant_method(torch.nn.Module(), "model.embed_tokens")

    assert isinstance(expert_method, Fp8MoEMethod)
    assert expert_method.block_quant is True
    assert expert_method.weight_block_size == [128, 128]
    assert expert_method.quant_config.activation_scheme == "dynamic"
    assert expert_method.quant_config.is_checkpoint_fp8_serialized is True
    assert isinstance(projection_method, UnquantizedLinearMethod)
    assert other_method is None
    assert config.get_name() == "bsgemm-moe"
    assert config.get_supported_act_dtypes() == [torch.bfloat16, torch.half]
    assert ExpertFp8Config.get_min_capability() == 80
    assert ExpertFp8Config.get_config_filenames() == []
    assert config.get_scaled_act_names() == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("quant_method", "fp8"),
        ("activation_scheme", "static"),
        ("weight_block_size", None),
        ("weight_block_size", [0, 128]),
        ("weight_block_size", [-1, 128]),
        ("weight_block_size", [64, 128]),
        ("weight_block_size", [128, 128, 1]),
    ],
)
def test_other_formats_are_rejected(key: str, value: object) -> None:
    quantization = dict(CHECKPOINT_QUANTIZATION)
    quantization[key] = value

    with pytest.raises(ValueError, match=key):
        ExpertFp8Config.from_config(quantization)


def test_missing_block_size_is_rejected() -> None:
    quantization = dict(CHECKPOINT_QUANTIZATION)
    quantization.pop("weight_block_size")

    with pytest.raises(ValueError, match="weight_block_size"):
        ExpertFp8Config.from_config(quantization)


def test_block_quantizer_keeps_each_block_scale_in_place() -> None:
    """The format the engine test writes: one multiplier per 128x128 block."""
    # These block amaxes give exact multipliers 2^-10, 2^-4 and 2^-7; one block is zero.
    block_amaxes = torch.tensor([[448 * 2.0**-10, 448 * 2.0**-4], [448 * 2.0**-7, 0.0]])
    unit_weight = (
        torch.rand(256, 256, generator=torch.Generator().manual_seed(0)) * 2 - 1
    )
    unit_weight[0::128, 0::128] = 1.0
    weight = (unit_weight * block_expanded(block_amaxes)).to(torch.bfloat16)
    floor = torch.tensor(FP8_BLOCK_AMAX_FLOOR).to(torch.bfloat16).float()

    values, scale_inv = quantize_block_fp8(weight)

    assert values.dtype == torch.float8_e4m3fn
    assert scale_inv.dtype == torch.float32
    assert torch.equal(
        scale_inv,
        torch.tensor([[2.0**-10, 2.0**-4], [2.0**-7, float(1.0 / (FP8_MAX / floor))]]),
    )
    assert values[0::128, 0::128].float().tolist() == [
        [FP8_MAX, FP8_MAX],
        [FP8_MAX, 0.0],
    ]
    assert not values[128:, 128:].float().any()
    # e4m3 keeps 3 mantissa bits; below its normal range the spacing is 2^-9.
    error_bound = 2.0**-4 * weight.float().abs() + 2.0**-10 * block_expanded(scale_inv)
    dequantized = dequantize_block_fp8(values, scale_inv)
    assert bool(((dequantized - weight.float()).abs() <= error_bound).all())
    with pytest.raises(ValueError, match="divisible by 128"):
        quantize_block_fp8(torch.zeros(100, 128))
