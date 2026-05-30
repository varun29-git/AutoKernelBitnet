from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from test2_bitnet_kernels import choose_attention_backend, default_attention_backend, flash_attention


@dataclass(frozen=True)
class BitNetLlamaConfig:
    vocab_size: int = 32_000
    dim: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    n_kv_heads: int = 4
    ffn_dim: int = 2048
    max_seq_len: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    pad_token_id: int = 0
    activation_bits: int = 8
    weight_init: str = "orthogonal"
    alpha_init: str = "absmean"
    attention_backend: str = "auto"
    use_subln: bool = True
    use_qk_norm: bool = True
    loss_chunk_size: int = 0
    quantization_strength: float = 1.0
    residual_init_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.activation_bits < 2:
            raise ValueError("activation_bits must be at least 2")
        if self.weight_init not in {"orthogonal", "normal"}:
            raise ValueError("weight_init must be 'orthogonal' or 'normal'")
        if self.alpha_init not in {"absmean", "one"}:
            raise ValueError("alpha_init must be 'absmean' or 'one'")
        if self.attention_backend not in {"auto", "sdpa", "flash"}:
            raise ValueError("attention_backend must be 'auto', 'sdpa', or 'flash'")
        if self.loss_chunk_size < 0:
            raise ValueError("loss_chunk_size must be non-negative")
        if not 0.0 <= self.quantization_strength <= 1.0:
            raise ValueError("quantization_strength must be in [0, 1]")
        if self.residual_init_scale <= 0.0:
            raise ValueError("residual_init_scale must be positive")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        normed = x_float * torch.rsqrt((x_float * x_float).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(dtype=dtype)


class BitLinear(nn.Module):
    """BitNet 1.58b-style linear layer with latent full-precision weights.

    The forward pass uses absmean ternarization to map latent weights to
    {-1, 0, +1} and absmax quantization to map activations to int8-like values.
    Gradients flow through both quantizers with straight-through estimators.
    The BitNet scale alpha is deterministic, not learned: by default it is the
    current latent-weight absmean for the layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        eps: float = 1e-6,
        activation_bits: int = 8,
        weight_init: str = "orthogonal",
        alpha_init: str = "absmean",
        quantization_strength: float = 1.0,
        init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if activation_bits < 2:
            raise ValueError("activation_bits must be at least 2")
        if weight_init not in {"orthogonal", "normal"}:
            raise ValueError("weight_init must be 'orthogonal' or 'normal'")
        if alpha_init not in {"absmean", "one"}:
            raise ValueError("alpha_init must be 'absmean' or 'one'")
        if not 0.0 <= quantization_strength <= 1.0:
            raise ValueError("quantization_strength must be in [0, 1]")
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.activation_bits = activation_bits
        self.activation_qmax = float(2 ** (activation_bits - 1) - 1)
        self.weight_init = weight_init
        self.alpha_init = alpha_init
        self.init_scale = init_scale
        self._quantization_strength = float(quantization_strength)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer(
            "quantization_strength",
            torch.tensor(float(quantization_strength), dtype=torch.float32),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.weight_init == "orthogonal":
            nn.init.orthogonal_(self.weight)
        else:
            nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(self.in_features))
        with torch.no_grad():
            if self.init_scale != 1.0:
                self.weight.mul_(self.init_scale)

    def alpha_scale(self) -> torch.Tensor:
        if self.alpha_init == "one":
            return torch.ones((), device=self.weight.device, dtype=torch.float32)
        return self.weight.float().abs().mean().clamp_min(self.eps)

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_scale()

    def ternary_weight(self) -> torch.Tensor:
        weight = self.weight.float()
        absmean = weight.abs().mean().clamp_min(self.eps)
        ternary = torch.round(weight / absmean).clamp_(-1, 1)
        return ternary

    def quantized_weight(self) -> torch.Tensor:
        ternary = self.ternary_weight()
        latent = self.weight.float()
        ste_weight = latent + (ternary - latent).detach()
        bitnet_weight = self.alpha_scale() * ste_weight
        if self._quantization_strength >= 1.0:
            return bitnet_weight
        strength = self.quantization_strength.to(device=latent.device, dtype=latent.dtype)
        return torch.lerp(latent, bitnet_weight, strength)

    def integer_weight(self) -> torch.Tensor:
        ternary = self.ternary_weight()
        latent = self.weight.float()
        return latent + (ternary - latent).detach()

    def quantized_activation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_float = x.float()
        absmax = x_float.abs().amax(dim=-1, keepdim=True).clamp_min(self.eps)
        scale = absmax / self.activation_qmax
        x_int = torch.round(x_float / scale).clamp(-self.activation_qmax, self.activation_qmax)
        x_dequant = x_int * scale
        x_ste = x_float + (x_dequant - x_float).detach()
        if self._quantization_strength >= 1.0:
            return x_ste.to(dtype=x.dtype), scale
        strength = self.quantization_strength.to(device=x.device, dtype=torch.float32)
        x_quant = torch.lerp(x_float, x_ste, strength)
        return x_quant.to(dtype=x.dtype), scale

    def quantized_integer_activation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_float = x.float()
        absmax = x_float.abs().amax(dim=-1, keepdim=True).clamp_min(self.eps)
        scale = absmax / self.activation_qmax
        x_int = torch.round(x_float / scale).clamp(-self.activation_qmax, self.activation_qmax)
        x_ste = x_float + (x_int - x_float).detach()
        return x_ste.to(dtype=x.dtype), scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._quantization_strength >= 1.0:
            x_int, activation_scale = self.quantized_integer_activation(x)
            ternary_weight = self.integer_weight().to(dtype=x.dtype)
            out = F.linear(x_int, ternary_weight, None)
            out = out.float() * (activation_scale * self.alpha_scale())
            return out.to(dtype=x.dtype)
        x_quant, _ = self.quantized_activation(x)
        weight = self.quantized_weight().to(dtype=x.dtype)
        return F.linear(x_quant, weight, None)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len = max_seq_len
        self.dim = dim

    def forward(
        self,
        seq_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if offset + seq_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {offset + seq_len} exceeds max_seq_len {self.max_seq_len}"
            )
        positions = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        cos = freqs.cos().to(dtype=dtype)[None, None, :, :]
        sin = freqs.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    out = torch.empty_like(x)
    out[..., :half] = (x1 * cos) - (x2 * sin)
    out[..., half:] = (x2 * cos) + (x1 * sin)
    return out


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    batch, n_kv_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, repeats, seq_len, head_dim)
    return x.reshape(batch, n_kv_heads * repeats, seq_len, head_dim)


class BitNetAttention(nn.Module):
    def __init__(self, config: BitNetLlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.kv_repeats = config.n_heads // config.n_kv_heads
        self.attention_backend = (
            default_attention_backend()
            if config.attention_backend == "auto"
            else config.attention_backend
        )

        self.q_proj = BitLinear(
            config.dim,
            config.n_heads * self.head_dim,
            activation_bits=config.activation_bits,
            weight_init=config.weight_init,
            alpha_init=config.alpha_init,
            quantization_strength=config.quantization_strength,
        )
        self.k_proj = BitLinear(
            config.dim,
            config.n_kv_heads * self.head_dim,
            activation_bits=config.activation_bits,
            weight_init=config.weight_init,
            alpha_init=config.alpha_init,
            quantization_strength=config.quantization_strength,
        )
        self.v_proj = BitLinear(
            config.dim,
            config.n_kv_heads * self.head_dim,
            activation_bits=config.activation_bits,
            weight_init=config.weight_init,
            alpha_init=config.alpha_init,
            quantization_strength=config.quantization_strength,
        )
        self.o_proj = BitLinear(
            config.n_heads * self.head_dim,
            config.dim,
            activation_bits=config.activation_bits,
            weight_init=config.weight_init,
            alpha_init=config.alpha_init,
            quantization_strength=config.quantization_strength,
            init_scale=config.residual_init_scale,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
        self.attn_sub_norm = RMSNorm(config.dim, eps=config.rms_norm_eps) if config.use_subln else nn.Identity()
        self.dropout = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        *,
        collect_attention_stats: bool = False,
        attention_stats: Optional[List[Dict[str, object]]] = None,
        layer_index: Optional[int] = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if collect_attention_stats and attention_stats is not None:
            attention_stats.append(self._attention_logit_stats(q, k, layer_index=layer_index))
        dropout_p = self.dropout if self.training else 0.0

        q_flash = q.transpose(1, 2).contiguous()
        k_flash = k.transpose(1, 2).contiguous()
        v_flash = v.transpose(1, 2).contiguous()
        backend = choose_attention_backend(
            self.attention_backend,
            q=q_flash,
            k=k_flash,
            attn_mask=attn_mask,
        )

        if backend == "flash":
            out = flash_attention(q_flash, k_flash, v_flash, dropout_p=dropout_p, causal=True)
            out = out.contiguous().view(batch, seq_len, self.config.dim)
            out = self.attn_sub_norm(out)
            return self.o_proj(out)

        k = repeat_kv(k, self.kv_repeats)
        v = repeat_kv(v, self.kv_repeats)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=attn_mask is None,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.config.dim)
        out = self.attn_sub_norm(out)
        return self.o_proj(out)

    @torch.no_grad()
    def _attention_logit_stats(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        *,
        layer_index: Optional[int],
        sample_seq_len: int = 256,
    ) -> Dict[str, object]:
        sample_len = min(int(sample_seq_len), int(q.shape[2]))
        if sample_len <= 0:
            return {
                "layer": int(layer_index) if layer_index is not None else -1,
                "sample_seq_len": 0,
                "max_logit": 0.0,
                "min_logit": 0.0,
                "mean_logit": 0.0,
                "std_logit": 0.0,
                "mean_entropy": 0.0,
                "normalized_entropy": 0.0,
                "top_heads": [],
            }

        if q.shape[2] == sample_len:
            positions = torch.arange(sample_len, device=q.device)
        else:
            positions = torch.linspace(0, q.shape[2] - 1, sample_len, device=q.device).long()

        q_sample = q[:1, :, positions, :].detach().float()
        k_sample = k[:1, :, positions, :].detach().float()
        k_sample = repeat_kv(k_sample, self.kv_repeats)
        logits = torch.matmul(q_sample, k_sample.transpose(-2, -1)) / math.sqrt(self.head_dim)

        causal = torch.ones(sample_len, sample_len, dtype=torch.bool, device=logits.device).tril()
        finite_logits = logits.masked_select(causal.view(1, 1, sample_len, sample_len))
        masked_logits = logits.masked_fill(~causal.view(1, 1, sample_len, sample_len), float("-inf"))
        probs = torch.softmax(masked_logits, dim=-1)
        entropy = -(probs * probs.clamp_min(1e-20).log()).sum(dim=-1)
        normalizer = math.log(max(2, sample_len))

        head_max = masked_logits.amax(dim=(-1, -2)).squeeze(0)
        head_entropy = entropy.mean(dim=(0, 2))
        top_count = min(4, int(head_max.numel()))
        top_values, top_indices = torch.topk(head_max, k=top_count)
        top_heads = [
            {
                "head": int(head_index),
                "max_logit": float(max_value),
                "mean_entropy": float(head_entropy[head_index]),
            }
            for max_value, head_index in zip(top_values.cpu(), top_indices.cpu())
        ]

        return {
            "layer": int(layer_index) if layer_index is not None else -1,
            "sample_seq_len": int(sample_len),
            "max_logit": float(finite_logits.max()),
            "min_logit": float(finite_logits.min()),
            "mean_logit": float(finite_logits.mean()),
            "std_logit": float(finite_logits.std(unbiased=False)),
            "mean_entropy": float(entropy.mean()),
            "normalized_entropy": float(entropy.mean() / normalizer),
            "top_heads": top_heads,
        }


class BitNetFeedForward(nn.Module):
    def __init__(self, config: BitNetLlamaConfig) -> None:
        super().__init__()
        self.gate_proj = BitLinear(
            config.dim,
            config.ffn_dim,
            activation_bits=config.activation_bits,
            weight_init=config.weight_init,
            alpha_init=config.alpha_init,
            quantization_strength=config.quantization_strength,
        )
        self.up_proj = BitLinear(
            config.dim,
            config.ffn_dim,
            activation_bits=config.activation_bits,
            weight_init=config.weight_init,
            alpha_init=config.alpha_init,
            quantization_strength=config.quantization_strength,
        )
        self.down_proj = BitLinear(
            config.ffn_dim,
            config.dim,
            activation_bits=config.activation_bits,
            weight_init=config.weight_init,
            alpha_init=config.alpha_init,
            quantization_strength=config.quantization_strength,
            init_scale=config.residual_init_scale,
        )
        self.ffn_sub_norm = RMSNorm(config.ffn_dim, eps=config.rms_norm_eps) if config.use_subln else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.ffn_sub_norm(hidden))


class BitNetBlock(nn.Module):
    def __init__(self, config: BitNetLlamaConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.attn = BitNetAttention(config)
        self.ffn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.ffn = BitNetFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        *,
        collect_attention_stats: bool = False,
        attention_stats: Optional[List[Dict[str, object]]] = None,
        layer_index: Optional[int] = None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.attn_norm(x),
            cos,
            sin,
            attn_mask=attn_mask,
            collect_attention_stats=collect_attention_stats,
            attention_stats=attention_stats,
            layer_index=layer_index,
        )
        x = x + self.ffn(self.ffn_norm(x))
        return x


class BitNetLlama(nn.Module):
    def __init__(self, config: Optional[BitNetLlamaConfig] = None) -> None:
        super().__init__()
        self.config = config or BitNetLlamaConfig()
        self.tok_embeddings = nn.Embedding(
            self.config.vocab_size,
            self.config.dim,
            padding_idx=self.config.pad_token_id,
        )
        self.dropout = nn.Dropout(self.config.dropout)
        self.rope = RotaryEmbedding(
            self.config.head_dim,
            self.config.max_seq_len,
            theta=self.config.rope_theta,
        )
        self.layers = nn.ModuleList(BitNetBlock(self.config) for _ in range(self.config.n_layers))
        self.norm = RMSNorm(self.config.dim, eps=self.config.rms_norm_eps)
        self.reset_parameters()
        self.assert_bias_free()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=1.0 / math.sqrt(self.config.dim))
        if self.config.pad_token_id is not None:
            with torch.no_grad():
                self.tok_embeddings.weight[self.config.pad_token_id].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        return_logits: bool = True,
        return_activation_stats: bool = False,
        return_attention_stats: bool = False,
    ) -> Dict[str, object]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq_len]")
        batch, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"seq_len {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        if targets is not None and targets.shape != input_ids.shape:
            raise ValueError("targets must have the same shape as input_ids")
        if attention_mask is None and self.config.pad_token_id is not None:
            maybe_padding = input_ids.eq(self.config.pad_token_id)
            if maybe_padding.any():
                attention_mask = ~maybe_padding

        x = self.tok_embeddings(input_ids)
        x = self.dropout(x)
        cos, sin = self.rope(seq_len, device=input_ids.device, dtype=x.dtype)
        attn_mask = self._prepare_attention_mask(
            attention_mask,
            batch=batch,
            seq_len=seq_len,
            device=input_ids.device,
        )

        attention_stats: Optional[List[Dict[str, object]]] = [] if return_attention_stats else None
        for layer_index, layer in enumerate(self.layers):
            x = layer(
                x,
                cos,
                sin,
                attn_mask=attn_mask,
                collect_attention_stats=return_attention_stats,
                attention_stats=attention_stats,
                layer_index=layer_index,
            )

        x = self.norm(x)
        activation_stats = self._activation_stats(x) if return_activation_stats else None
        logits = (
            F.linear(x, self.tok_embeddings.weight)
            if return_logits or targets is None or self.config.loss_chunk_size == 0
            else None
        )
        loss = None
        loss_tokens = None
        if targets is not None:
            loss_targets = targets
            if attention_mask is not None and attention_mask.ndim == 2:
                loss_targets = targets.masked_fill(~attention_mask.to(torch.bool), -100)
            loss, loss_tokens = self._language_model_loss(x, loss_targets, logits)

        return {
            "loss": loss,
            "loss_tokens": loss_tokens,
            "logits": logits if return_logits else None,
            "activation_stats": activation_stats,
            "attention_stats": attention_stats,
        }

    @staticmethod
    def _activation_stats(hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        values = hidden.detach().float()
        squared = values * values
        return {
            "mean_abs": values.abs().mean(),
            "rms": torch.sqrt(squared.mean()),
            "max_abs": values.abs().amax(),
        }

    def _language_model_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        logits: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flat_targets = targets.reshape(-1)
        loss_tokens = flat_targets.ne(-100).sum()
        loss_denominator = loss_tokens.clamp_min(1)

        if logits is not None:
            loss_sum = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size).float(),
                flat_targets,
                ignore_index=-100,
                reduction="sum",
            )
            return loss_sum / loss_denominator, loss_tokens

        flat_hidden = hidden.reshape(-1, self.config.dim)
        loss_sum = torch.zeros((), device=hidden.device, dtype=torch.float32)
        if self.config.loss_chunk_size <= 0:
            raise ValueError("loss_chunk_size must be positive when logits are not materialized")
        chunk_size = min(self.config.loss_chunk_size, flat_hidden.shape[0])
        for start in range(0, flat_hidden.shape[0], chunk_size):
            end = min(start + chunk_size, flat_hidden.shape[0])
            chunk_targets = flat_targets[start:end]
            chunk_logits = F.linear(flat_hidden[start:end], self.tok_embeddings.weight).float()
            loss_sum = loss_sum + F.cross_entropy(
                chunk_logits,
                chunk_targets,
                ignore_index=-100,
                reduction="sum",
            )
        return loss_sum / loss_denominator, loss_tokens

    @staticmethod
    def _prepare_attention_mask(
        attention_mask: Optional[torch.Tensor],
        *,
        batch: int,
        seq_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None:
            return None

        mask = attention_mask.to(device=device)
        if mask.ndim == 2:
            if mask.shape != (batch, seq_len):
                raise ValueError("2D attention_mask must have shape [batch, seq_len]")
            key_is_valid = mask.to(torch.bool)
            causal = torch.ones(seq_len, seq_len, device=device, dtype=torch.bool).tril()
            allowed = causal[None, None, :, :] & key_is_valid[:, None, None, :]

            # Avoid fully masked rows for padded query positions. Those query
            # outputs are ignored in the loss, but SDPA still needs a finite row.
            empty_rows = ~allowed.any(dim=-1, keepdim=True)
            diagonal = torch.eye(seq_len, device=device, dtype=torch.bool)[None, None, :, :]
            return allowed | (empty_rows & diagonal)

        if mask.ndim == 3:
            if mask.shape != (batch, seq_len, seq_len):
                raise ValueError("3D attention_mask must have shape [batch, seq_len, seq_len]")
            return mask[:, None, :, :]

        if mask.ndim == 4:
            if mask.shape[0] != batch or mask.shape[-2:] != (seq_len, seq_len):
                raise ValueError(
                    "4D attention_mask must have shape [batch, heads_or_1, seq_len, seq_len]"
                )
            return mask

        raise ValueError("attention_mask must be 2D, 3D, or 4D")

    def bitlinear_modules(self) -> Iterable[BitLinear]:
        for module in self.modules():
            if isinstance(module, BitLinear):
                yield module

    def bitlinear_named_modules(self) -> Iterable[Tuple[str, BitLinear]]:
        for name, module in self.named_modules():
            if isinstance(module, BitLinear):
                yield name, module

    @torch.no_grad()
    def set_quantization_strength_(self, strength: float) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError("quantization strength must be in [0, 1]")
        for module in self.bitlinear_modules():
            module._quantization_strength = float(strength)
            module.quantization_strength.fill_(float(strength))

    def muon_parameters(self) -> List[nn.Parameter]:
        return [module.weight for module in self.bitlinear_modules()]

    def alpha_parameters(self) -> List[nn.Parameter]:
        return []

    def adamw_parameters(self) -> List[nn.Parameter]:
        muon_ids = {id(param) for param in self.muon_parameters()}
        return [param for param in self.parameters() if id(param) not in muon_ids]

    def adamw_decay_parameters(self) -> List[nn.Parameter]:
        muon_ids = {id(param) for param in self.muon_parameters()}
        alpha_ids = {id(param) for param in self.alpha_parameters()}
        return [
            param
            for param in self.parameters()
            if id(param) not in muon_ids and id(param) not in alpha_ids and param.ndim >= 2
        ]

    def adamw_no_decay_parameters(self) -> List[nn.Parameter]:
        muon_ids = {id(param) for param in self.muon_parameters()}
        alpha_ids = {id(param) for param in self.alpha_parameters()}
        return [
            param
            for param in self.parameters()
            if id(param) not in muon_ids and id(param) not in alpha_ids and param.ndim < 2
        ]

    def optimizer_parameter_groups(
        self,
        *,
        muon_lr: float = 0.02,
        adamw_lr: float = 3e-4,
        alpha_lr: float = 1e-4,
        adamw_weight_decay: float = 0.1,
    ) -> List[Dict[str, object]]:
        return [
            {
                "name": "muon",
                "params": self.muon_parameters(),
                "lr": muon_lr,
                "weight_decay": 0.0,
            },
            {
                "name": "adamw_decay",
                "params": self.adamw_decay_parameters(),
                "lr": adamw_lr,
                "betas": (0.9, 0.95),
                "weight_decay": adamw_weight_decay,
            },
            {
                "name": "adamw_no_decay",
                "params": self.adamw_no_decay_parameters(),
                "lr": adamw_lr,
                "betas": (0.9, 0.95),
                "weight_decay": 0.0,
            },
        ]

    def clip_alpha_grad_norm(self, max_norm: float = 0.1) -> torch.Tensor:
        return torch.tensor(0.0)

    @torch.no_grad()
    def clamp_alpha_(self, min_value: float = 1e-5, max_value: float = 2.0) -> None:
        return None

    @torch.no_grad()
    def alpha_magnitude(self) -> torch.Tensor:
        alphas = [module.alpha.detach().float().abs() for module in self.bitlinear_modules()]
        if not alphas:
            return torch.tensor(0.0)
        return torch.stack(alphas).mean()

    @torch.no_grad()
    def alpha_stats(self) -> Dict[str, torch.Tensor]:
        alphas = [module.alpha.detach().float() for module in self.bitlinear_modules()]
        if not alphas:
            zero = torch.tensor(0.0)
            return {"mean_abs": zero, "min": zero, "max": zero}
        stacked = torch.stack(alphas)
        return {
            "mean_abs": stacked.abs().mean(),
            "min": stacked.min(),
            "max": stacked.max(),
        }

    @torch.no_grad()
    def ternary_weight_histogram(self) -> Dict[int, int]:
        hist = {-1: 0, 0: 0, 1: 0}
        for module in self.bitlinear_modules():
            values, counts = module.ternary_weight().to(torch.int8).unique(return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist()):
                hist[int(value)] = hist.get(int(value), 0) + int(count)
        return hist

    @torch.no_grad()
    def ternary_weight_snapshot(self) -> List[torch.Tensor]:
        return [module.ternary_weight().to(torch.int8).detach().cpu() for module in self.bitlinear_modules()]

    @torch.no_grad()
    def weight_flip_rate(self, previous_snapshot: List[torch.Tensor]) -> float:
        report, _ = self.weight_flip_report(previous_snapshot, top_k=0)
        return float(report["total_rate"])

    @torch.no_grad()
    def weight_flip_report(
        self,
        previous_snapshot: List[torch.Tensor],
        *,
        top_k: int = 5,
    ) -> Tuple[Dict[str, object], List[torch.Tensor]]:
        named_modules = list(self.bitlinear_named_modules())
        current = [module.ternary_weight().to(torch.int8).detach().cpu() for _, module in named_modules]
        if len(current) != len(previous_snapshot):
            raise ValueError("snapshot does not match the current BitLinear module count")

        changed = 0
        total = 0
        transition_counts = torch.zeros(9, dtype=torch.long)
        layers = []
        for (name, _), old, new in zip(named_modules, previous_snapshot, current):
            if old.shape != new.shape:
                raise ValueError("snapshot tensor shape does not match current ternary weight shape")
            layer_changed = int((old != new).sum().item())
            layer_total = int(new.numel())
            layer_transition_counts = torch.bincount(
                ((old.reshape(-1).to(torch.long) + 1) * 3 + (new.reshape(-1).to(torch.long) + 1)),
                minlength=9,
            )
            transition_counts += layer_transition_counts
            changed += layer_changed
            total += layer_total
            layers.append(
                {
                    "name": name,
                    "rate": float(layer_changed / layer_total) if layer_total else 0.0,
                    "changed": layer_changed,
                    "total": layer_total,
                    "transition_counts": self._format_transition_counts(layer_transition_counts),
                    "transition_rates": self._format_transition_rates(layer_transition_counts),
                }
            )

        layers.sort(key=lambda item: item["rate"], reverse=True)
        report = {
            "total_rate": float(changed / total) if total else 0.0,
            "changed": changed,
            "total": total,
            "transition_counts": self._format_transition_counts(transition_counts),
            "transition_rates": self._format_transition_rates(transition_counts),
            "top_layers": layers[:max(0, top_k)],
        }
        return report, current

    @staticmethod
    def _format_transition_counts(counts: torch.Tensor) -> Dict[str, int]:
        states = (-1, 0, 1)
        flat = counts.reshape(3, 3).tolist()
        return {
            f"{old}->{new}": int(flat[old_index][new_index])
            for old_index, old in enumerate(states)
            for new_index, new in enumerate(states)
        }

    @staticmethod
    def _format_transition_rates(counts: torch.Tensor) -> Dict[str, float]:
        total = int(counts.sum().item())
        if total <= 0:
            return {key: 0.0 for key in BitNetLlama._format_transition_counts(counts)}
        return {
            key: value / total
            for key, value in BitNetLlama._format_transition_counts(counts).items()
        }

    @torch.no_grad()
    def quantization_error_report(self, *, top_k: int = 5) -> Dict[str, object]:
        layers = []
        weighted_error = 0.0
        weighted_total = 0
        for name, module in self.bitlinear_named_modules():
            latent = module.weight.detach().float()
            quantized = module.quantized_weight().detach().float()
            denom = torch.mean(latent * latent).clamp_min(module.eps)
            relative_mse = torch.mean((latent - quantized) * (latent - quantized)) / denom
            layer_total = latent.numel()
            value = float(relative_mse)
            weighted_error += value * layer_total
            weighted_total += layer_total
            layers.append({"name": name, "relative_mse": value, "total": layer_total})

        layers.sort(key=lambda item: item["relative_mse"], reverse=True)
        return {
            "mean_relative_mse": float(weighted_error / weighted_total) if weighted_total else 0.0,
            "top_layers": layers[:max(0, top_k)],
        }

    @torch.no_grad()
    def module_health_report(self, *, top_k: int = 5) -> Dict[str, object]:
        layers = []
        for name, module in self.bitlinear_named_modules():
            latent = module.weight.detach().float()
            ternary = module.ternary_weight().to(torch.int8)
            total = ternary.numel()
            zeros = int(ternary.eq(0).sum().item())
            neg = int(ternary.eq(-1).sum().item())
            pos = int(ternary.eq(1).sum().item())
            layers.append(
                {
                    "name": name,
                    "alpha": float(module.alpha.detach().float()),
                    "latent_mean": float(latent.mean()),
                    "latent_absmean": float(latent.abs().mean()),
                    "latent_rms": float(torch.sqrt((latent * latent).mean())),
                    "latent_max_abs": float(latent.abs().amax()),
                    "p_zero": zeros / total if total else 0.0,
                    "p_neg": neg / total if total else 0.0,
                    "p_pos": pos / total if total else 0.0,
                }
            )

        count = max(0, top_k)
        return {
            "highest_alpha": sorted(layers, key=lambda item: item["alpha"], reverse=True)[:count],
            "lowest_alpha": sorted(layers, key=lambda item: item["alpha"])[:count],
            "highest_latent_rms": sorted(
                layers,
                key=lambda item: item["latent_rms"],
                reverse=True,
            )[:count],
            "highest_zero_fraction": sorted(
                layers,
                key=lambda item: item["p_zero"],
                reverse=True,
            )[:count],
        }

    @torch.no_grad()
    def initialization_report(self, *, top_k: int = 5) -> Dict[str, object]:
        layers = []
        total_zero_rows = 0
        total_rows = 0
        for name, module in self.bitlinear_named_modules():
            latent = module.weight.detach().float()
            ternary = module.ternary_weight().to(torch.int8)
            rows = ternary.shape[0]
            zero_rows = int(ternary.abs().sum(dim=1).eq(0).sum().item())
            total_zero_rows += zero_rows
            total_rows += rows
            hist = {-1: 0, 0: 0, 1: 0}
            values, counts = ternary.unique(return_counts=True)
            for value, count in zip(values.tolist(), counts.tolist()):
                hist[int(value)] = int(count)
            total = ternary.numel()
            layers.append(
                {
                    "name": name,
                    "latent_absmean": float(latent.abs().mean()),
                    "latent_rms": float(torch.sqrt((latent * latent).mean())),
                    "alpha": float(module.alpha.detach().float()),
                    "init_scale": float(module.init_scale),
                    "p_zero": hist[0] / total if total else 0.0,
                    "p_neg": hist[-1] / total if total else 0.0,
                    "p_pos": hist[1] / total if total else 0.0,
                    "zero_rows": zero_rows,
                    "rows": rows,
                }
            )
        layers.sort(key=lambda item: (item["zero_rows"], item["p_zero"]), reverse=True)
        return {
            "weight_init": self.config.weight_init,
            "alpha_init": self.config.alpha_init,
            "residual_init_scale": self.config.residual_init_scale,
            "alpha_stats": tensor_stats_to_python(self.alpha_stats()),
            "ternary_histogram": self.ternary_weight_histogram(),
            "zero_row_fraction": total_zero_rows / total_rows if total_rows else 0.0,
            "highest_zero_layers": layers[:max(0, top_k)],
        }

    def count_parameters(self) -> Dict[str, int]:
        muon_ids = {id(param) for param in self.muon_parameters()}
        muon = 0
        adamw = 0
        for param in self.parameters():
            if id(param) in muon_ids:
                muon += param.numel()
            else:
                adamw += param.numel()
        return {"total": muon + adamw, "muon": muon, "adamw": adamw}

    def assert_bias_free(self) -> None:
        for module_name, module in self.named_modules():
            bias = getattr(module, "bias", None)
            if bias is not None:
                name = module_name or module.__class__.__name__
                raise ValueError(f"bias parameter found in bias-free BitNet model: {name}")
        for param_name, _ in self.named_parameters():
            if param_name.endswith(".bias") or ".bias." in param_name:
                raise ValueError(f"bias parameter found in bias-free BitNet model: {param_name}")


def build_model(config: Optional[BitNetLlamaConfig] = None) -> BitNetLlama:
    return BitNetLlama(config)


def tensor_stats_to_python(stats: Dict[str, torch.Tensor]) -> Dict[str, float]:
    return {key: float(value) for key, value in stats.items()}


if __name__ == "__main__":
    smoke_config = BitNetLlamaConfig(
        vocab_size=256,
        dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        ffn_dim=344,
        max_seq_len=64,
    )
    model = build_model(smoke_config)
    input_ids = torch.randint(1, smoke_config.vocab_size, (2, 16))
    targets = torch.randint(1, smoke_config.vocab_size, (2, 16))
    input_ids[1, -4:] = smoke_config.pad_token_id
    targets[1, -4:] = smoke_config.pad_token_id
    attention_mask = input_ids.ne(smoke_config.pad_token_id)
    output = model(input_ids, targets, attention_mask=attention_mask)
    output["loss"].backward()
    print("loss", float(output["loss"]))
    print("params", model.count_parameters())
    print("hist", model.ternary_weight_histogram())
