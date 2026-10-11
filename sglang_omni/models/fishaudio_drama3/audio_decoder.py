# SPDX-License-Identifier: Apache-2.0
"""Drama 3 frame embeddings and the codebook tower that predicts codes
1..num_codebooks - 1."""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import GemmaRMSNorm
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from torch import nn

from sglang_omni.models.fishaudio_drama3.hf_config import (
    FishOmniAudioDecoderConfig,
    FishOmniConfig,
)
from sglang_omni.models.fishaudio_drama3.sampling import (
    sample_semantic_with_ras,
    sample_top_k_top_p,
)
from sglang_omni.models.fishaudio_drama3.sampling_kernels import (
    FIRST_CODEBOOK_HEAD_STREAM,
    CodebookSampler,
)
from sglang_omni.models.fishaudio_drama3.tower_kernels import (
    FusedTowerLayers,
    NormWeights,
    TowerLayerWeights,
    fused_tower_layers_unsupported_reason,
    tower_layer_shape,
)
from sglang_omni.platforms import current_platform
from sglang_omni.platforms.interface import JointRopeInplaceKernel


class DecoderAttention(Protocol):
    def __call__(
        self,
        layer_index: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Write this call's key/value rows to tower layer layer_index (0-based, shared
        layers first), then attend.

        query [rows, heads * head_dim] and key/value [rows, kv_heads * head_dim];
        query and key arrive normed and rotated. Return causal GQA attention (scale
        1 / sqrt(head_dim)) over each row's own request, [rows, heads * head_dim],
        before the output gate. The callee owns request/position metadata, which
        must match the tower's positions.
        """


@runtime_checkable
class PagedDecoderAttention(DecoderAttention, Protocol):
    def forward_mode(self) -> ForwardMode: ...

    def cache_slots(self) -> torch.Tensor: ...

    def key_buffer(self, layer_index: int) -> torch.Tensor: ...

    def value_buffer(self, layer_index: int) -> torch.Tensor: ...

    def attend(self, layer_index: int, query: torch.Tensor) -> torch.Tensor: ...


def norm_weights(norm: GemmaRMSNorm) -> NormWeights:
    return NormWeights(weight=norm.weight, eps=norm.variance_epsilon)


class InterleavedRotaryEmbedding(nn.Module):
    def __init__(self, config: FishOmniAudioDecoderConfig, max_position: int) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        inverse_frequencies = 1.0 / config.rope_base ** (
            torch.arange(0, config.rotary_dim, 2, dtype=torch.float32)
            / config.rotary_dim
        )
        angles = (
            torch.arange(max_position, dtype=torch.float32).unsqueeze(-1)
            * inverse_frequencies
        )
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((angles.cos(), angles.sin()), dim=-1),
            persistent=False,
        )
        self.kernel: JointRopeInplaceKernel | None = (
            current_platform.get_joint_rope_inplace_kernel()
        )

    def forward(
        self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row_count = positions.shape[0]
        rotary_query = query.view(row_count, -1, self.head_dim)[..., : self.rotary_dim]
        rotary_key = key.view(row_count, -1, self.head_dim)[..., : self.rotary_dim]
        if self.kernel is not None and query.is_cuda:
            self.kernel(
                rotary_query, rotary_key, self.cos_sin_cache, positions, is_neox=False
            )
        else:
            cosine, sine = self.cos_sin_cache[positions].unsqueeze(1).chunk(2, dim=-1)
            for rotary_states in (rotary_query, rotary_key):
                even = rotary_states[..., 0::2].float()
                odd = rotary_states[..., 1::2].float()
                rotary_states.copy_(
                    torch.stack(
                        (even * cosine - odd * sine, odd * cosine + even * sine), dim=-1
                    ).flatten(-2)
                )
        return query, key


class DecoderSelfAttention(nn.Module):
    def __init__(
        self,
        config: FishOmniAudioDecoderConfig,
        rotary_embedding: InterleavedRotaryEmbedding,
    ) -> None:
        super().__init__()
        self.head_count = config.n_head
        self.kv_head_count = config.n_local_heads
        self.head_dim = config.head_dim
        query_width = config.n_head * config.head_dim
        kv_width = config.n_local_heads * config.head_dim
        # Output rows: each head's [query, gate] pair, then key, then value.
        self.qkv_proj = nn.Linear(
            config.dim, 2 * query_width + 2 * kv_width, bias=False
        )
        self.o_proj = nn.Linear(query_width, config.dim, bias=False)
        self.q_norm = GemmaRMSNorm(config.head_dim, config.norm_eps)
        self.k_norm = GemmaRMSNorm(config.head_dim, config.norm_eps)
        self.rotary_embedding = rotary_embedding

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        attention: DecoderAttention,
        layer_index: int,
    ) -> torch.Tensor:
        row_count = hidden_states.shape[0]
        query_width = self.head_count * self.head_dim
        kv_width = self.kv_head_count * self.head_dim
        query_and_gate, key, value = self.qkv_proj(hidden_states).split(
            [2 * query_width, kv_width, kv_width], dim=-1
        )
        query, gate = query_and_gate.view(
            row_count, self.head_count, 2 * self.head_dim
        ).chunk(2, dim=-1)
        query, key = self.rotary_embedding(
            positions,
            self.q_norm(query.reshape(-1, self.head_dim)).view(row_count, query_width),
            self.k_norm(key.reshape(-1, self.head_dim)).view(row_count, kv_width),
        )
        attended = attention(layer_index, query, key, value)
        return self.o_proj(
            attended * torch.sigmoid(gate.reshape(row_count, query_width))
        )


class DecoderMlp(nn.Module):
    def __init__(self, dim: int, intermediate_size: int) -> None:
        super().__init__()
        # Output rows: gate, then up.
        self.gate_up_proj = nn.Linear(dim, 2 * intermediate_size, bias=False)
        self.silu_and_mul = SiluAndMul()
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.silu_and_mul(self.gate_up_proj(hidden_states)))


class DecoderLayer(nn.Module):
    """One tower layer: mix a code embedding into the hidden, then a pre-norm block."""

    def __init__(
        self,
        config: FishOmniAudioDecoderConfig,
        rotary_embedding: InterleavedRotaryEmbedding,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.enorm = GemmaRMSNorm(config.dim, config.norm_eps)
        self.hnorm = GemmaRMSNorm(config.dim, config.norm_eps)
        self.eh_proj = nn.Linear(2 * config.dim, config.dim, bias=False)
        self.self_attn = DecoderSelfAttention(config, rotary_embedding)
        self.mlp = DecoderMlp(config.dim, config.intermediate_size)
        self.input_layernorm = GemmaRMSNorm(config.dim, config.norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.dim, config.norm_eps)
        self.final_layernorm = GemmaRMSNorm(config.dim, config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        code_ids: torch.Tensor,
        label_is_semantic: torch.Tensor,
        shared_input: torch.Tensor | None,
        positions: torch.Tensor,
        attention: DecoderAttention,
        layer_index: int,
    ) -> torch.Tensor:
        # Clamping keeps placeholder codes of non-semantic labels inside the
        # table; the label mask then zeroes those rows.
        embeddings = torch.where(
            label_is_semantic.unsqueeze(-1),
            self.token_embedding(
                code_ids.clamp(0, self.token_embedding.num_embeddings - 1)
            ),
            0.0,
        )
        if shared_input is not None:
            embeddings = embeddings + shared_input.to(embeddings.dtype)
        else:
            pass
        mixed = self.eh_proj(
            torch.cat((self.enorm(embeddings), self.hnorm(hidden_states)), dim=-1)
        )
        mlp_input, residual = self.post_attention_layernorm(
            self.self_attn(
                self.input_layernorm(mixed), positions, attention, layer_index
            ),
            mixed,
        )
        normed, _ = self.final_layernorm(self.mlp(mlp_input), residual)
        return normed

    def tower_layer_weights(self) -> TowerLayerWeights:
        attention = self.self_attn
        return TowerLayerWeights(
            token_embedding=self.token_embedding.weight,
            enorm=norm_weights(self.enorm),
            hnorm=norm_weights(self.hnorm),
            eh_proj=self.eh_proj,
            input_norm=norm_weights(self.input_layernorm),
            qkv_proj=attention.qkv_proj,
            q_norm=norm_weights(attention.q_norm),
            k_norm=norm_weights(attention.k_norm),
            o_proj=attention.o_proj,
            post_norm=norm_weights(self.post_attention_layernorm),
            gate_up_proj=self.mlp.gate_up_proj,
        )


class AudioDecoder(nn.Module):
    """Frame embeddings and the codebook tower of Drama 3.

    A frame row is [main token id, code 0, ..., code num_codebooks - 1]; code 0
    is the main token's offset in the semantic band.
    """

    def __init__(self, config: FishOmniConfig) -> None:
        super().__init__()
        decoder_config = config.audio_decoder_config
        self.semantic_begin_id = config.semantic_start_token_id
        self.semantic_end_id = config.semantic_end_token_id
        self.codebook_size = decoder_config.vocab_size
        self.num_codebooks = decoder_config.num_codebooks
        self.backbone_codebook_count = decoder_config.vq_backbone_codebooks
        rotary_embedding = InterleavedRotaryEmbedding(
            decoder_config, config.text_config.max_position_embeddings
        )
        self.codebook_embeddings = nn.Embedding(
            self.codebook_size * self.num_codebooks, decoder_config.dim
        )
        self.register_buffer(
            "codebook_row_offsets",
            torch.arange(self.num_codebooks) * self.codebook_size,
            persistent=False,
        )
        self.shared_layers = nn.ModuleList(
            DecoderLayer(decoder_config, rotary_embedding)
            for _ in range(decoder_config.shared_enrichment_layers)
        )
        self.residual_layers = nn.ModuleList(
            DecoderLayer(decoder_config, rotary_embedding)
            for _ in range(self.num_codebooks - 1)
        )
        self.codebook_heads = nn.ModuleList(
            nn.Linear(decoder_config.dim, self.codebook_size, bias=False)
            for _ in range(self.num_codebooks - 1)
        )
        self.register_buffer(
            "main_token_candidate_ids",
            torch.tensor(
                sorted(
                    {
                        config.eos_token_id,
                        *range(self.semantic_begin_id, self.semantic_end_id + 1),
                    }
                )
            ),
            persistent=False,
        )
        self.register_buffer(
            "sampling_step", torch.zeros((), dtype=torch.int64), persistent=False
        )
        self.sampler: CodebookSampler | None = None
        self.fused_layers: FusedTowerLayers | None = None

    def fused_layers_unsupported_reason(
        self,
        *,
        kv_pool: HybridLinearKVPool,
        tower_layer_ids: list[int],
        context_length: int,
        device: torch.device,
    ) -> str | None:
        attention = self.shared_layers[0].self_attn
        return fused_tower_layers_unsupported_reason(
            layers=[
                layer.tower_layer_weights()
                for layer in (*self.shared_layers, *self.residual_layers)
            ],
            head_count=attention.head_count,
            kv_head_count=attention.kv_head_count,
            rotary_dim=attention.rotary_embedding.rotary_dim,
            rotary_cos_sin=attention.rotary_embedding.cos_sin_cache,
            context_length=context_length,
            kv_pool=kv_pool,
            tower_layer_ids=tower_layer_ids,
            device=device,
        )

    def enable_fused_layers(self, device: torch.device) -> None:
        layer = self.shared_layers[0]
        attention = layer.self_attn
        self.fused_layers = FusedTowerLayers(
            shape=tower_layer_shape(
                weights=layer.tower_layer_weights(),
                head_count=attention.head_count,
                kv_head_count=attention.kv_head_count,
                rotary_dim=attention.rotary_embedding.rotary_dim,
                multiprocessor_count=torch.cuda.get_device_properties(
                    device
                ).multi_processor_count,
            ),
            rotary_cos_sin=attention.rotary_embedding.cos_sin_cache,
            device=device,
        )

    def is_semantic(self, token_ids: torch.Tensor) -> torch.Tensor:
        return (token_ids >= self.semantic_begin_id) & (
            token_ids <= self.semantic_end_id
        )

    def codebook_embedding_sum(
        self, frames: torch.Tensor, first_codebook: int, end_codebook: int
    ) -> torch.Tensor:
        """Summed embeddings of codes first_codebook..end_codebook - 1 per frame."""
        if first_codebook == end_codebook:
            return self.codebook_embeddings.weight.new_zeros(
                frames.shape[0], self.codebook_embeddings.embedding_dim
            )
        else:
            table_rows = (
                frames[:, 1 + first_codebook : 1 + end_codebook].clamp(
                    0, self.codebook_size - 1
                )
                + self.codebook_row_offsets[first_codebook:end_codebook]
            )
            return F.embedding_bag(
                table_rows, self.codebook_embeddings.weight, mode="sum"
            )

    def embed_backbone(
        self, frames: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Backbone input: semantic frames add codes 1..backbone_codebook_count to the
        token embedding."""
        mixed = (
            text_embeddings
            + self.codebook_embedding_sum(
                frames, 1, 1 + self.backbone_codebook_count
            ).to(text_embeddings.dtype)
        ) * (1.0 / math.sqrt(1.0 + self.backbone_codebook_count))
        return torch.where(
            self.is_semantic(frames[:, 0]).unsqueeze(-1), mixed, text_embeddings
        )

    def shared_input(self, frames: torch.Tensor) -> torch.Tensor:
        """Extra input of the first shared layer: codes backbone_codebook_count + 1.. of
        semantic frames."""
        return torch.where(
            self.is_semantic(frames[:, 0]).unsqueeze(-1),
            self.codebook_embedding_sum(
                frames, 1 + self.backbone_codebook_count, self.num_codebooks
            ),
            0.0,
        )

    def run_layer(
        self,
        layer: DecoderLayer,
        layer_index: int,
        hidden_states: torch.Tensor,
        code_ids: torch.Tensor,
        label_is_semantic: torch.Tensor,
        shared_input: torch.Tensor | None,
        positions: torch.Tensor,
        attention: DecoderAttention,
        decode_pass: bool,
    ) -> torch.Tensor:
        fused_layers = self.fused_layers
        if (
            fused_layers is not None
            and decode_pass
            and isinstance(attention, PagedDecoderAttention)
            and attention.forward_mode() == ForwardMode.DECODE
            and fused_layers.covers(hidden_states.shape[0])
        ):
            activated, residual = fused_layers.forward_layer(
                weights=layer.tower_layer_weights(),
                layer_index=layer_index,
                hidden_states=hidden_states,
                code_ids=code_ids,
                label_is_semantic=label_is_semantic,
                shared_input=shared_input,
                positions=positions,
                cache_slots=attention.cache_slots(),
                key_cache=attention.key_buffer(layer_index),
                value_cache=attention.value_buffer(layer_index),
                attention=attention.attend,
            )
            normed, _ = layer.final_layernorm(layer.mlp.down_proj(activated), residual)
            return normed
        else:
            return layer(
                hidden_states,
                code_ids,
                label_is_semantic,
                shared_input,
                positions,
                attention,
                layer_index,
            )

    def run_shared_layers(
        self,
        hidden_states: torch.Tensor,
        codebook_zero: torch.Tensor,
        label_is_semantic: torch.Tensor,
        shared_input: torch.Tensor,
        positions: torch.Tensor,
        attention: DecoderAttention,
        decode_pass: bool = False,
    ) -> torch.Tensor:
        for layer_index, layer in enumerate(self.shared_layers):
            hidden_states = self.run_layer(
                layer,
                layer_index,
                hidden_states,
                codebook_zero,
                label_is_semantic,
                shared_input if layer_index == 0 else None,
                positions,
                attention,
                decode_pass,
            )
        return hidden_states

    def prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_frames: torch.Tensor,
        next_frames: torch.Tensor,
        attention: DecoderAttention,
    ) -> None:
        """Fill the tower cache of prompt rows; row p gets frame p + 1 as next frame.

        hidden_states is the backbone's final-norm output (the lm_head input).
        """
        label_is_semantic = self.is_semantic(next_frames[:, 0])
        hidden_states = self.run_shared_layers(
            hidden_states,
            next_frames[:, 1],
            label_is_semantic,
            self.shared_input(input_frames),
            positions,
            attention,
        )
        for residual_index, layer in enumerate(self.residual_layers):
            hidden_states = self.run_layer(
                layer,
                len(self.shared_layers) + residual_index,
                hidden_states,
                next_frames[:, 1 + residual_index],
                label_is_semantic,
                None,
                positions,
                attention,
                False,
            )

    def decode_step(
        self,
        semantic_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_frames: torch.Tensor,
        previous_main_tokens: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        attention: DecoderAttention,
    ) -> torch.Tensor:
        """Sample the next int64 frame [rows, 1 + num_codebooks] per backbone row,
        filling the tower cache at its position.

        semantic_logits are the raw lm_head logits (no bias or temperature) of
        hidden_states, the backbone's final-norm output; an end-of-speech frame still
        runs every layer.
        """
        sampler = self.sampler
        if sampler is not None:
            sampler.step.add_(1)
        else:
            pass
        main_tokens, codebook_zero = sample_semantic_with_ras(
            semantic_logits,
            self.main_token_candidate_ids,
            previous_main_tokens,
            temperature,
            top_k,
            top_p,
            self.semantic_begin_id,
            self.codebook_size,
            sampler,
        )
        label_is_semantic = self.is_semantic(main_tokens)
        hidden_states = self.run_shared_layers(
            hidden_states,
            codebook_zero,
            label_is_semantic,
            self.shared_input(input_frames),
            positions,
            attention,
            True,
        )
        codes = [codebook_zero]
        for residual_index, (layer, head) in enumerate(
            zip(self.residual_layers, self.codebook_heads)
        ):
            hidden_states = self.run_layer(
                layer,
                len(self.shared_layers) + residual_index,
                hidden_states,
                codes[-1],
                label_is_semantic,
                None,
                positions,
                attention,
                True,
            )
            codes.append(
                sample_top_k_top_p(
                    head(hidden_states),
                    temperature,
                    top_k,
                    top_p,
                    sampler,
                    FIRST_CODEBOOK_HEAD_STREAM + residual_index,
                )
            )
        return torch.stack([main_tokens, *codes], dim=-1)
