# SPDX-License-Identifier: Apache-2.0
"""S3-drama codebook decoder: frame embeddings and the tower predicting codes 1..N-1.

The tower's key/value cache stays with the caller: every layer attends through
a DecoderAttention callback. The cache slot of backbone row p is computed from
that row's hidden state and the codes of frame p + 1.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch
import torch.nn.functional as F
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import GemmaRMSNorm
from torch import nn

from sglang_omni.models.fishaudio_s3_drama.hf_config import (
    FishOmniAudioDecoderConfig,
    FishOmniConfig,
)
from sglang_omni.models.fishaudio_s3_drama.sampling import (
    sample_semantic_with_ras,
    sample_top_k_top_p,
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
        """Store this call's key/value rows in tower layer layer_index, then attend.

        layer_index counts tower layers from 0, shared layers first. query
        [rows, heads * head_dim] and key [rows, kv_heads * head_dim] are already
        normed and rotated; value is [rows, kv_heads * head_dim]. Each row attends
        causally over its own request's cached rows up to its position, with
        grouped kv heads and a 1 / sqrt(head_dim) scale. The result
        [rows, heads * head_dim] is returned before the output gate. The callee
        owns the rows' requests and positions, which must match the positions
        given to the tower.
        """


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


class AudioDecoder(nn.Module):
    """Frame embeddings and the codebook tower of S3-drama.

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
            torch.unique(
                torch.cat(
                    (
                        torch.tensor([config.eos_token_id]),
                        torch.arange(self.semantic_begin_id, self.semantic_end_id + 1),
                    )
                )
            ),
            persistent=False,
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
        """Backbone input: semantic frames add codes 1..N to the token embedding."""
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
        """Extra input of the first shared layer: codes N+1.. of semantic frames."""
        return torch.where(
            self.is_semantic(frames[:, 0]).unsqueeze(-1),
            self.codebook_embedding_sum(
                frames, 1 + self.backbone_codebook_count, self.num_codebooks
            ),
            0.0,
        )

    def run_shared_layers(
        self,
        hidden_states: torch.Tensor,
        codebook_zero: torch.Tensor,
        label_is_semantic: torch.Tensor,
        shared_input: torch.Tensor,
        positions: torch.Tensor,
        attention: DecoderAttention,
    ) -> torch.Tensor:
        for layer_index, layer in enumerate(self.shared_layers):
            hidden_states = layer(
                hidden_states,
                codebook_zero,
                label_is_semantic,
                shared_input if layer_index == 0 else None,
                positions,
                attention,
                layer_index,
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
            hidden_states = layer(
                hidden_states,
                next_frames[:, 1 + residual_index],
                label_is_semantic,
                None,
                positions,
                attention,
                len(self.shared_layers) + residual_index,
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
        """Sample the frame after each backbone row, filling the cache at its position.

        hidden_states is the backbone's final-norm output (the lm_head input) and
        semantic_logits its raw lm_head logits, before bias or temperature.
        Returns int64 frames [rows, 1 + num_codebooks]. A frame whose main token
        is not semantic (end of speech) still runs every layer.
        """
        main_tokens, codebook_zero = sample_semantic_with_ras(
            semantic_logits,
            self.main_token_candidate_ids,
            previous_main_tokens,
            temperature,
            top_k,
            top_p,
            self.semantic_begin_id,
            self.codebook_size,
        )
        label_is_semantic = self.is_semantic(main_tokens)
        hidden_states = self.run_shared_layers(
            hidden_states,
            codebook_zero,
            label_is_semantic,
            self.shared_input(input_frames),
            positions,
            attention,
        )
        codes = [codebook_zero]
        for residual_index, (layer, head) in enumerate(
            zip(self.residual_layers, self.codebook_heads)
        ):
            hidden_states = layer(
                hidden_states,
                codes[-1],
                label_is_semantic,
                None,
                positions,
                attention,
                len(self.shared_layers) + residual_index,
            )
            codes.append(
                sample_top_k_top_p(head(hidden_states), temperature, top_k, top_p)
            )
        return torch.stack([main_tokens, *codes], dim=-1)
