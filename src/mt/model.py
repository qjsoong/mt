"""A small Transformer encoder--decoder, implemented from scratch.

The architecture is a pre-norm Transformer with optional RMSNorm and SwiGLU
feed-forward blocks. Positional information is either fixed sinusoidal or a
learned embedding table. Encoder input embeddings, decoder input embeddings
and the output projection are tied by default.

Design notes
------------
* Attention uses :func:`torch.nn.functional.scaled_dot_product_attention`,
  so flash-attention/memory-efficient kernels are used automatically when
  available.
* Masks follow the PyTorch convention: a boolean attention mask is ``True``
  for positions that *participate* in attention; a boolean key padding mask
  is ``True`` for padded positions that must be *ignored*.
* Incremental decoding (for greedy/beam search) is implemented next to the
  teacher-forced forward pass and shares all parameters. ``tests/`` checks
  that incremental logits match full forward logits.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

# A KV cache entry for one attention module: (keys, values) of shape
# (batch, heads, length, head_dim).
KVCache = tuple[torch.Tensor, torch.Tensor]


def sinusoidal_positions(
    max_len: int,
    d_model: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Classic fixed sinusoidal position encoding, shape (max_len, d_model)."""
    position = torch.arange(max_len, device=device, dtype=torch.float32)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    encoding = torch.zeros(max_len, d_model, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(position[:, None] * div_term)
    encoding[:, 1::2] = torch.cos(position[:, None] * div_term)
    return encoding.to(dtype)


class RMSNorm(nn.Module):
    """Root mean square layer normalization."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def make_norm(name: str, dim: int) -> nn.Module:
    if name == "rmsnorm":
        return RMSNorm(dim)
    if name == "layernorm":
        return nn.LayerNorm(dim, eps=1e-5)
    raise ValueError(f"unknown norm: {name}")


def _valid_mask(
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """Convert a (B, L) padding mask into a broadcastable keep mask."""
    if key_padding_mask is None:
        return None
    return (~key_padding_mask).unsqueeze(1).unsqueeze(2)


def _repeat_batch(
    tensor: torch.Tensor | None, batch_size: int
) -> torch.Tensor | None:
    """Repeat a per-sequence tensor along the batch dimension.

    Beam search runs a batch of ``B * beam`` sequences while encoder-side
    masks and caches are computed for ``B`` sequences. Broadcasting cannot
    do that reduction for arbitrary batch sizes, so expand explicitly.
    """
    if tensor is None or tensor.size(0) == batch_size:
        return tensor
    if batch_size % tensor.size(0) != 0:
        raise ValueError(
            f"cannot expand batch dimension {tensor.size(0)} to {batch_size}"
        )
    return tensor.repeat_interleave(batch_size // tensor.size(0), dim=0)


class MultiHeadAttention(nn.Module):
    """Self- or cross-attention with optional key/value caches."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.head_dim).transpose(
            1, 2
        )

    def project_kv(self, x: torch.Tensor) -> KVCache:
        """Project a memory tensor into cached (key, value) pairs."""
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))
        return key, value

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        kv: KVCache | None = None,
        past_kv: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """Attend.

        Args:
            query: (B, Lq, D) tensor.
            key/value: (B, Lk, D) tensors; if omitted, query is used (self
                attention). Ignored when ``kv`` is given.
            attn_mask: boolean keep-mask broadcastable to
                (B, n_heads, Lq, Lk).
            kv: pre-projected ``(key, value)`` pair; when provided the
                projections are skipped.
            past_kv: previously computed key/value pairs for incremental
                decoding; the current step's keys/values are appended.

        Returns:
            The attention output (B, Lq, D) and the updated ``(key, value)``
            pair, or ``None`` when caching was not requested.
        """
        batch = query.shape[0]
        q = self._split_heads(self.q_proj(query))
        if kv is not None:
            k, v = kv
            new_kv: KVCache | None = None
        else:
            if key is None:
                key = query
            if value is None:
                value = key
            k = self._split_heads(self.k_proj(key))
            v = self._split_heads(self.v_proj(value))
            new_kv = (k, v)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
            new_kv = (k, v)

        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, -1, self.d_model)
        return self.out_proj(attended), new_kv


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str,
        dropout: float,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.activation = activation
        if activation == "swiglu":
            self.w_in = nn.Linear(d_model, 2 * d_ff, bias=bias)
        else:
            self.w_in = nn.Linear(d_model, d_ff, bias=bias)
        self.w_out = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w_in(x)
        if self.activation == "swiglu":
            gate, value = x.chunk(2, dim=-1)
            x = F.silu(gate) * value
        elif self.activation == "gelu":
            x = F.gelu(x)
        else:
            x = F.relu(x)
        return self.dropout(self.w_out(x))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            cfg.d_model, cfg.n_heads, cfg.attention_dropout, cfg.bias
        )
        self.ffn = FeedForward(
            cfg.d_model, cfg.d_ff, cfg.activation, cfg.dropout, cfg.bias
        )
        self.norm1 = make_norm(cfg.norm, cfg.d_model)
        self.norm2 = make_norm(cfg.norm, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        attention, _ = self.self_attn(
            self.norm1(x), attn_mask=_valid_mask(key_padding_mask)
        )
        x = x + self.dropout(attention)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(
            cfg.d_model, cfg.n_heads, cfg.attention_dropout, cfg.bias
        )
        self.cross_attn = MultiHeadAttention(
            cfg.d_model, cfg.n_heads, cfg.attention_dropout, cfg.bias
        )
        self.ffn = FeedForward(
            cfg.d_model, cfg.d_ff, cfg.activation, cfg.dropout, cfg.bias
        )
        self.norm1 = make_norm(cfg.norm, cfg.d_model)
        self.norm2 = make_norm(cfg.norm, cfg.d_model)
        self.norm3 = make_norm(cfg.norm, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        memory_padding_mask: torch.Tensor | None = None,
        self_past_kv: KVCache | None = None,
        cross_kv: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        self_attention, new_self_kv = self.self_attn(
            self.norm1(x),
            attn_mask=self_attn_mask,
            past_kv=self_past_kv,
        )
        x = x + self.dropout(self_attention)

        cross_attention, _ = self.cross_attn(
            self.norm2(x),
            key=memory,
            value=memory,
            attn_mask=_valid_mask(memory_padding_mask),
            kv=cross_kv,
        )
        x = x + self.dropout(cross_attention)

        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x, new_self_kv


def _self_attention_mask(
    target_tokens: torch.Tensor,
    target_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Causal mask optionally combined with a target padding mask."""
    length = target_tokens.size(1)
    causal = torch.ones(
        length, length, dtype=torch.bool, device=target_tokens.device
    ).tril()
    mask = causal.unsqueeze(0).unsqueeze(0)
    if target_padding_mask is not None:
        mask = mask & _valid_mask(target_padding_mask)
    return mask


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embedding_scale = math.sqrt(cfg.d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(cfg) for _ in range(cfg.n_encoder_layers)]
        )
        self.norm = make_norm(cfg.norm, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        embeddings: nn.Embedding,
        positions: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = embeddings(inputs) * self.embedding_scale
        x = x + positions[: inputs.size(1)].unsqueeze(0).to(x.dtype)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.embedding_scale = math.sqrt(cfg.d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg) for _ in range(cfg.n_decoder_layers)]
        )
        self.norm = make_norm(cfg.norm, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        memory: torch.Tensor,
        embeddings: nn.Embedding,
        positions: torch.Tensor,
        target_padding_mask: torch.Tensor | None = None,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        length = inputs.size(1)
        x = embeddings(inputs) * self.embedding_scale
        x = x + positions[:length].unsqueeze(0).to(x.dtype)
        x = self.dropout(x)

        mask = _self_attention_mask(inputs, target_padding_mask)
        for layer in self.layers:
            x, _ = layer(
                x,
                memory,
                self_attn_mask=mask,
                memory_padding_mask=memory_padding_mask,
            )
        return self.norm(x)


class Transformer(nn.Module):
    """Encoder--decoder translation model."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(
            cfg.vocab_size, cfg.d_model, padding_idx=cfg.padding_idx
        )
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.output_projection = nn.Linear(
            cfg.d_model, cfg.vocab_size, bias=False
        )
        if cfg.tie_embeddings:
            self.output_projection.weight = self.embedding.weight
        if cfg.positional == "sinusoidal":
            self.register_buffer(
                "positions",
                sinusoidal_positions(cfg.max_positions, cfg.d_model),
                persistent=False,
            )
        else:
            self.positional_embedding = nn.Embedding(
                cfg.max_positions, cfg.d_model
            )
        self.reset_parameters()

    # ------------------------------------------------------------------
    # Parameter initialization
    # ------------------------------------------------------------------
    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            if (
                isinstance(module, nn.Embedding)
                and module.padding_idx is not None
            ):
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def _positions_for(
        self, offset: int, length: int, device: torch.device
    ) -> torch.Tensor:
        """Positions for ``[offset, offset + length)``; shape (length, D)."""
        if self.cfg.positional == "sinusoidal":
            return self.positions[offset : offset + length].to(device)
        index = torch.arange(offset, offset + length, device=device)
        return self.positional_embedding(index)

    def num_parameters(self) -> int:
        """Count model parameters; tied weights are counted once."""
        return sum(parameter.numel() for parameter in self.parameters())

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------
    def encode(
        self,
        source_tokens: torch.Tensor,
        source_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        positions = self._positions_for(
            0, source_tokens.size(1), source_tokens.device
        )
        return self.encoder(
            source_tokens,
            self.embedding,
            positions,
            key_padding_mask=source_padding_mask,
        )

    def decode(
        self,
        target_tokens: torch.Tensor,
        memory: torch.Tensor,
        target_padding_mask: torch.Tensor | None = None,
        memory_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        positions = self._positions_for(
            0, target_tokens.size(1), target_tokens.device
        )
        return self.decoder(
            target_tokens,
            memory,
            self.embedding,
            positions,
            target_padding_mask=target_padding_mask,
            memory_padding_mask=memory_padding_mask,
        )

    def forward(
        self,
        source_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        source_padding_mask: torch.Tensor | None = None,
        target_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced forward pass; returns logits (B, T, V)."""
        memory = self.encode(source_tokens, source_padding_mask)
        hidden = self.decode(
            target_tokens,
            memory,
            target_padding_mask=target_padding_mask,
            memory_padding_mask=source_padding_mask,
        )
        return self.output_projection(hidden)

    # ------------------------------------------------------------------
    # Incremental decoding
    # ------------------------------------------------------------------
    def prepare_cross_cache(self, memory: torch.Tensor) -> list[KVCache]:
        """Pre-project encoder states for all decoder cross-attention layers."""
        return [
            layer.cross_attn.project_kv(memory) for layer in self.decoder.layers
        ]

    def decode_step(
        self,
        target_tokens: torch.Tensor,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor | None,
        cross_cache: list[KVCache],
        self_cache: list[KVCache] | None = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, list[KVCache]]:
        """Decode one token per sequence, updating the self-attention cache.

        ``target_tokens`` has shape (B, 1). Returns logits (B, V) and the
        updated per-layer self-attention cache. Cross-attention keys/values
        come from ``cross_cache`` and never change. ``memory_padding_mask``
        and ``cross_cache`` may describe fewer sequences than
        ``target_tokens`` (during beam search); they are repeated along the
        batch dimension to match.
        """
        batch_size = target_tokens.size(0)
        memory_padding_mask = _repeat_batch(memory_padding_mask, batch_size)
        if cross_cache and cross_cache[0][0].size(0) != batch_size:
            cross_cache = [
                (
                    _repeat_batch(key, batch_size),
                    _repeat_batch(value, batch_size),
                )
                for key, value in cross_cache
            ]
        positions = self._positions_for(
            position_offset, 1, target_tokens.device
        )
        x = self.embedding(target_tokens) * self.decoder.embedding_scale
        x = x + positions.unsqueeze(0).to(x.dtype)
        x = self.decoder.dropout(x)

        new_cache: list[KVCache] = []
        for index, layer in enumerate(self.decoder.layers):
            past = None if self_cache is None else self_cache[index]
            x, updated = layer(
                x,
                memory,
                self_attn_mask=None,
                memory_padding_mask=memory_padding_mask,
                self_past_kv=past,
                cross_kv=cross_cache[index],
            )
            assert updated is not None
            new_cache.append(updated)
        hidden = self.decoder.norm(x)
        logits = self.output_projection(hidden[:, -1])
        return logits, new_cache

    @staticmethod
    def cache_length(self_cache: list[KVCache] | None) -> int:
        if not self_cache:
            return 0
        return self_cache[0][0].size(2)
