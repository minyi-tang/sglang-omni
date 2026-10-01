# SPDX-License-Identifier: Apache-2.0
"""HF config for S3-drama checkpoints (model_type fish_omni).

A fish_omni_text text_config is validated and converted to Qwen3_5MoeTextConfig;
the converted form is what to_dict writes and what a saved config reloads from.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from sglang.srt.configs.qwen3_5 import Qwen3_5MoeTextConfig
from transformers import PretrainedConfig

FISH_SCHEMA_VERSION = 1
SPEECH_TEMPLATE_VERSION = "reference-bracket-v1"
FISH_TEXT_MODEL_TYPE = "fish_omni_text"
# note (SunskyXH): fish_omni_text keys Qwen3_5MoeTextConfig has no field for;
# their values are checked against what the backbone hard-codes, then dropped.
FISH_ONLY_TEXT_KEYS = (
    "model_type",
    "layer_types",
    "mlp_layer_types",
    "rotary_interleaved",
    "use_qk_norm",
    "layernorm_zero_centered_gamma",
    "shared_expert_gate",
)
# Values the backbone hard-codes, with the fish_omni_text default when absent.
FISH_TEXT_INVARIANTS = (
    ("rotary_interleaved", False, False),
    ("use_qk_norm", True, True),
    ("attn_output_gate", True, True),
    ("layernorm_zero_centered_gamma", False, True),
)
POSITIVE_TEXT_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "vocab_size",
    "max_position_embeddings",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "full_attention_interval",
)


def checked_int(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; got {value!r}")
    else:
        return value


def checked_number(value: object, name: str, upper_bound: float) -> float:
    """A finite number in (0, upper_bound]."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number; got {value!r}")
    elif not (math.isfinite(value) and 0.0 < value <= upper_bound):
        raise ValueError(
            f"{name} must be finite and in (0, {upper_bound}]; got {value}"
        )
    else:
        return float(value)


def checked_rope_parameters(rope_parameters: object) -> dict[str, float]:
    if not isinstance(rope_parameters, Mapping):
        raise ValueError(
            f"text_config.rope_parameters must be a dict; got {rope_parameters!r}"
        )
    elif rope_parameters.get("rope_type", rope_parameters.get("type", "default")) != (
        "default"
    ):
        raise NotImplementedError(
            f"rope_type of {dict(rope_parameters)!r} is not supported; a scaled "
            f"rope must not load as the default rope"
        )
    elif "mrope_section" in rope_parameters:
        raise NotImplementedError("mrope_section is not supported on this backbone")
    else:
        return {
            "rope_theta": checked_number(
                rope_parameters.get("rope_theta"),
                "rope_parameters.rope_theta",
                math.inf,
            ),
            "partial_rotary_factor": checked_number(
                rope_parameters.get("partial_rotary_factor", 1.0),
                "rope_parameters.partial_rotary_factor",
                1.0,
            ),
        }


def fish_text_kwargs(text_config: Mapping[str, object]) -> dict[str, object]:
    """Validate a fish_omni_text dict and return Qwen3_5MoeTextConfig kwargs for it."""
    num_hidden_layers = checked_int(
        text_config.get("num_hidden_layers"), "text_config.num_hidden_layers", 1
    )
    layer_types = text_config.get("layer_types")
    if (
        not isinstance(layer_types, list)
        or len(layer_types) != num_hidden_layers
        or not set(layer_types) <= {"linear_attention", "full_attention"}
        or "full_attention" not in layer_types
    ):
        raise ValueError(
            f"text_config.layer_types must list linear_attention / full_attention "
            f"for each of the {num_hidden_layers} layers, with at least one "
            f"full_attention; got {layer_types!r}"
        )
    else:
        pass
    full_attention_interval = layer_types.index("full_attention") + 1
    if any(
        (layer_type == "full_attention")
        != ((layer_index + 1) % full_attention_interval == 0)
        for layer_index, layer_type in enumerate(layer_types)
    ):
        raise NotImplementedError(
            "text_config.layer_types is not periodic; Qwen3_5MoeTextConfig only "
            "expresses every full_attention_interval-th layer as full attention"
        )
    elif text_config.get("mlp_layer_types") != ["moe"] * num_hidden_layers:
        raise NotImplementedError(
            "every text_config.mlp_layer_types entry must be moe for the "
            "Qwen3.5 MoE backbone"
        )
    else:
        pass
    for key, fish_default, required in FISH_TEXT_INVARIANTS:
        if text_config.get(key, fish_default) is not required:
            raise NotImplementedError(
                f"text_config.{key} must be {required} (the backbone hard-codes "
                f"it); got {text_config.get(key, fish_default)!r}"
            )
        else:
            pass
    if text_config.get("shared_expert_gate", True) is not True and text_config.get(
        "shared_expert_intermediate_size"
    ):
        raise NotImplementedError(
            "text_config.shared_expert_gate=False is not supported: the backbone "
            "always gates its shared expert"
        )
    else:
        pass
    kwargs = {
        key: value
        for key, value in text_config.items()
        if key not in FISH_ONLY_TEXT_KEYS
    }
    kwargs["full_attention_interval"] = full_attention_interval
    return kwargs


def checked_text_config(text_kwargs: dict[str, object]) -> Qwen3_5MoeTextConfig:
    """Build the backbone config; its RoPE fields must all repeat rope_parameters."""
    rope_parameters = text_kwargs.get("rope_parameters")
    rope = checked_rope_parameters(rope_parameters)
    for key, value in (
        ("rope_theta", rope["rope_theta"]),
        ("partial_rotary_factor", rope["partial_rotary_factor"]),
        ("rope_scaling", rope_parameters),
    ):
        if text_kwargs.get(key) in (None, value):
            text_kwargs[key] = value
        else:
            raise ValueError(
                f"text_config.{key}={text_kwargs[key]!r} disagrees with "
                f"rope_parameters={rope_parameters!r}"
            )
    if text_kwargs.get("shared_expert_intermediate_size") is None:
        text_kwargs["shared_expert_intermediate_size"] = 0
    else:
        pass
    text_config = Qwen3_5MoeTextConfig(**text_kwargs)
    for name in POSITIVE_TEXT_FIELDS:
        checked_int(getattr(text_config, name), f"text_config.{name}", 1)
    checked_int(
        text_config.shared_expert_intermediate_size,
        "text_config.shared_expert_intermediate_size",
        0,
    )
    checked_number(text_config.rms_norm_eps, "text_config.rms_norm_eps", math.inf)
    rotary_dim = int(text_config.head_dim * text_config.partial_rotary_factor)
    if text_config.num_attention_heads % text_config.num_key_value_heads != 0:
        raise ValueError(
            f"text_config.num_attention_heads={text_config.num_attention_heads} is "
            f"not a multiple of num_key_value_heads={text_config.num_key_value_heads}"
        )
    elif text_config.linear_num_value_heads % text_config.linear_num_key_heads != 0:
        raise ValueError(
            f"text_config.linear_num_value_heads={text_config.linear_num_value_heads} "
            f"is not a multiple of linear_num_key_heads="
            f"{text_config.linear_num_key_heads}"
        )
    elif text_config.num_experts_per_tok > text_config.num_experts:
        raise ValueError(
            f"text_config.num_experts_per_tok={text_config.num_experts_per_tok} "
            f"exceeds num_experts={text_config.num_experts}"
        )
    elif text_config.full_attention_interval > text_config.num_hidden_layers:
        raise ValueError(
            f"text_config.full_attention_interval leaves no full attention layer "
            f"in {text_config.num_hidden_layers} layers"
        )
    elif text_config.attn_output_gate is not True:
        raise NotImplementedError("text_config.attn_output_gate must be True")
    elif rotary_dim == 0 or rotary_dim % 2 != 0:
        raise ValueError(
            f"text_config.head_dim * partial_rotary_factor must give a positive "
            f"even rotary width; got {rotary_dim}"
        )
    else:
        return text_config


class FishOmniAudioDecoderConfig(PretrainedConfig):
    """The codebook decoder tower config, under the checkpoint's own field names.

    rotary_interleaved and attention_qk_norm are stored but not read: the
    tower always uses interleaved rope and per-head q/k norm, whatever they say.
    """

    model_type = "fish_qwen3_audio_decoder"
    base_config_key = "audio_decoder_config"
    has_no_defaults_at_init = True

    def __init__(
        self,
        *,
        decoder_type: str | None = None,
        dim: int | None = None,
        n_head: int | None = None,
        n_local_heads: int | None = None,
        head_dim: int | None = None,
        n_layer: int | None = None,
        intermediate_size: int | None = None,
        norm_eps: float | None = None,
        num_codebooks: int | None = None,
        vocab_size: int | None = None,
        text_dim: int | None = None,
        max_seq_len: int | None = None,
        shared_enrichment_layers: int | None = None,
        rope_base: float | None = None,
        rotary_percent: float | None = None,
        rotary_interleaved: bool | None = None,
        attention_output_gate: bool | None = None,
        attention_qk_norm: bool | None = None,
        attention_qkv_bias: bool = False,
        attention_o_bias: bool = False,
        layernorm_zero_centered_gamma: bool = False,
        vq_feature_scope: str = "all_vq",
        vq_feature_placement: str | None = None,
        vq_backbone_codebooks: int = 0,
        mtp_spec: Mapping[str, object] | None = None,
        use_moe: bool = False,
        mtp_use_swa: bool = False,
        window_size: int | None = None,
        window_attn_skip_freq: int | None = None,
        window_rotary_base: float | None = None,
        dropout: float = 0.0,
        **kwargs: object,
    ) -> None:
        declared_model_type = kwargs.pop("model_type", self.model_type)
        if declared_model_type != self.model_type:
            raise ValueError(
                f"audio_decoder_config.model_type={declared_model_type!r}; expected "
                f"{self.model_type!r}"
            )
        else:
            pass
        self.dim = checked_int(dim, "audio_decoder_config.dim", 1)
        self.n_head = checked_int(n_head, "audio_decoder_config.n_head", 1)
        self.n_local_heads = checked_int(
            n_local_heads, "audio_decoder_config.n_local_heads", 1
        )
        self.head_dim = checked_int(head_dim, "audio_decoder_config.head_dim", 1)
        self.n_layer = checked_int(n_layer, "audio_decoder_config.n_layer", 1)
        self.intermediate_size = checked_int(
            intermediate_size, "audio_decoder_config.intermediate_size", 1
        )
        self.num_codebooks = checked_int(
            num_codebooks, "audio_decoder_config.num_codebooks", 2
        )
        self.vocab_size = checked_int(vocab_size, "audio_decoder_config.vocab_size", 1)
        self.text_dim = checked_int(text_dim, "audio_decoder_config.text_dim", 1)
        self.max_seq_len = checked_int(
            max_seq_len, "audio_decoder_config.max_seq_len", 1
        )
        self.shared_enrichment_layers = checked_int(
            shared_enrichment_layers, "audio_decoder_config.shared_enrichment_layers", 1
        )
        self.vq_backbone_codebooks = checked_int(
            vq_backbone_codebooks, "audio_decoder_config.vq_backbone_codebooks", 0
        )
        self.norm_eps = checked_number(
            norm_eps, "audio_decoder_config.norm_eps", math.inf
        )
        self.rope_base = checked_number(
            rope_base, "audio_decoder_config.rope_base", math.inf
        )
        self.rotary_percent = checked_number(
            rotary_percent, "audio_decoder_config.rotary_percent", 1.0
        )
        if mtp_spec is None:
            self.mtp_spec: dict[str, object] = {}
        elif isinstance(mtp_spec, Mapping):
            self.mtp_spec = dict(mtp_spec)
        else:
            raise ValueError(
                f"audio_decoder_config.mtp_spec must be a dict or null; got "
                f"{mtp_spec!r}"
            )
        if decoder_type != "codebook_mtp_shared":
            raise NotImplementedError(
                f"audio_decoder_config.decoder_type={decoder_type!r} is not supported"
            )
        elif vq_feature_scope != "all_vq":
            raise NotImplementedError(
                f"audio_decoder_config.vq_feature_scope={vq_feature_scope!r} is not "
                f"supported"
            )
        elif vq_feature_placement == "backbone_partial" and not (
            1 <= self.vq_backbone_codebooks <= self.num_codebooks - 1
        ):
            raise ValueError(
                f"backbone_partial needs 1 <= vq_backbone_codebooks <= "
                f"{self.num_codebooks - 1}; got {self.vq_backbone_codebooks}"
            )
        elif vq_feature_placement == "mtp_shared_pre" and self.vq_backbone_codebooks:
            raise ValueError("mtp_shared_pre needs vq_backbone_codebooks = 0")
        elif vq_feature_placement not in ("backbone_partial", "mtp_shared_pre"):
            raise NotImplementedError(
                f"audio_decoder_config.vq_feature_placement={vq_feature_placement!r} "
                f"is not supported"
            )
        elif self.n_layer != self.shared_enrichment_layers:
            raise ValueError(
                f"audio_decoder_config.n_layer={self.n_layer} differs from "
                f"shared_enrichment_layers={self.shared_enrichment_layers}"
            )
        elif self.max_seq_len != self.num_codebooks + 1:
            raise ValueError(
                f"audio_decoder_config.max_seq_len={self.max_seq_len} is not "
                f"num_codebooks + 1 = {self.num_codebooks + 1}"
            )
        elif self.dim != self.text_dim:
            raise NotImplementedError(
                f"audio_decoder_config.dim={self.dim} differs from text_dim="
                f"{self.text_dim}"
            )
        elif self.n_head % self.n_local_heads != 0:
            raise ValueError(
                f"audio_decoder_config.n_head={self.n_head} is not a multiple of "
                f"n_local_heads={self.n_local_heads}"
            )
        elif self.rotary_dim % 2 != 0 or self.rotary_dim == 0:
            raise ValueError(
                f"head_dim * rotary_percent must give a positive even rotary width; "
                f"got {self.rotary_dim}"
            )
        elif attention_output_gate is not True:
            raise NotImplementedError(
                "the decoder tower is implemented with the attention output gate only"
            )
        elif layernorm_zero_centered_gamma is not True:
            raise NotImplementedError(
                "the decoder tower is implemented with zero-centered norm gammas only"
            )
        elif self.mtp_spec.get("attention_variant", "full_attention") != (
            "full_attention"
        ):
            raise NotImplementedError("the decoder tower is full attention only")
        elif (
            use_moe
            or any(
                key in ("num_experts", "num_experts_per_tok") or key.startswith("moe_")
                for key in self.mtp_spec
            )
            or mtp_use_swa
            or attention_qkv_bias
            or attention_o_bias
            or window_size is not None
            or window_attn_skip_freq is not None
            or window_rotary_base is not None
            or dropout != 0.0
        ):
            raise NotImplementedError(
                "MoE, sliding windows, attention biases and dropout are not part of "
                "the decoder tower"
            )
        else:
            pass
        self.decoder_type = decoder_type
        self.rotary_interleaved = rotary_interleaved
        self.attention_output_gate = True
        self.attention_qk_norm = attention_qk_norm
        self.layernorm_zero_centered_gamma = True
        self.vq_feature_scope = vq_feature_scope
        self.vq_feature_placement = vq_feature_placement
        super().__init__(**kwargs)

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rotary_percent)


class FishOmniConfig(PretrainedConfig):
    """S3-drama checkpoint: a Qwen3.5 MoE backbone plus the codebook decoder tower."""

    model_type = "fish_omni"
    sub_configs = {
        "text_config": Qwen3_5MoeTextConfig,
        "audio_decoder_config": FishOmniAudioDecoderConfig,
    }
    has_no_defaults_at_init = True

    def __init__(
        self,
        text_config: Mapping[str, object] | None = None,
        audio_decoder_config: Mapping[str, object] | None = None,
        audio_encoder_config: Mapping[str, object] | None = None,
        semantic_start_token_id: int | None = None,
        semantic_end_token_id: int | None = None,
        speech_template_version: str | None = None,
        extra_samplable_token_ids: list[int] | None = None,
        fish_schema_version: int = FISH_SCHEMA_VERSION,
        **kwargs: object,
    ) -> None:
        if fish_schema_version != FISH_SCHEMA_VERSION:
            raise ValueError(
                f"fish_schema_version={fish_schema_version!r} is not supported "
                f"(expected {FISH_SCHEMA_VERSION})"
            )
        elif speech_template_version != SPEECH_TEMPLATE_VERSION:
            raise ValueError(
                f"speech_template_version={speech_template_version!r}; this model "
                f"only builds {SPEECH_TEMPLATE_VERSION} prompts"
            )
        elif audio_encoder_config is not None:
            raise NotImplementedError("audio_encoder_config is not supported")
        elif extra_samplable_token_ids:
            raise NotImplementedError("extra_samplable_token_ids is not supported")
        elif not isinstance(text_config, Mapping):
            raise ValueError(f"text_config must be a dict; got {text_config!r}")
        elif not isinstance(audio_decoder_config, Mapping):
            raise ValueError(
                f"audio_decoder_config must be a dict; got {audio_decoder_config!r}"
            )
        elif text_config.get("model_type") == FISH_TEXT_MODEL_TYPE:
            text_kwargs = fish_text_kwargs(text_config)
        elif text_config.get("model_type") == Qwen3_5MoeTextConfig.model_type:
            text_kwargs = {
                key: value for key, value in text_config.items() if key != "model_type"
            }
        else:
            raise ValueError(
                f"text_config.model_type={text_config.get('model_type')!r}; expected "
                f"{FISH_TEXT_MODEL_TYPE!r} or {Qwen3_5MoeTextConfig.model_type!r}"
            )
        self.text_config = checked_text_config(text_kwargs)
        self.audio_decoder_config = FishOmniAudioDecoderConfig(**audio_decoder_config)
        self.fish_schema_version = fish_schema_version
        self.speech_template_version = speech_template_version
        self.audio_encoder_config = None
        super().__init__(**kwargs)
        if not self.architectures:
            self.architectures = ["FishOmniModel"]
        else:
            pass

        text_vocab_size = self.text_config.vocab_size
        self.semantic_start_token_id = checked_int(
            semantic_start_token_id, "semantic_start_token_id", 0
        )
        self.semantic_end_token_id = checked_int(
            semantic_end_token_id, "semantic_end_token_id", self.semantic_start_token_id
        )
        eos_token_id = checked_int(self.eos_token_id, "eos_token_id", 0)
        if self.semantic_end_token_id >= text_vocab_size:
            raise ValueError(
                f"semantic_end_token_id={self.semantic_end_token_id} is outside the "
                f"text vocabulary of {text_vocab_size}"
            )
        elif (
            self.semantic_end_token_id - self.semantic_start_token_id + 1
            != self.audio_decoder_config.vocab_size
        ):
            raise ValueError(
                f"the semantic band [{self.semantic_start_token_id}, "
                f"{self.semantic_end_token_id}] does not match the codebook size "
                f"{self.audio_decoder_config.vocab_size}"
            )
        elif eos_token_id >= text_vocab_size:
            raise ValueError(
                f"eos_token_id={eos_token_id} is outside the text vocabulary of "
                f"{text_vocab_size}"
            )
        elif self.semantic_start_token_id <= eos_token_id <= self.semantic_end_token_id:
            raise ValueError(f"eos_token_id={eos_token_id} is inside the semantic band")
        elif self.audio_decoder_config.dim != self.text_config.hidden_size:
            raise ValueError(
                f"audio_decoder_config.dim={self.audio_decoder_config.dim} differs "
                f"from text hidden_size={self.text_config.hidden_size}"
            )
        else:
            pass
