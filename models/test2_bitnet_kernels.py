from __future__ import annotations

import os
from functools import lru_cache
from typing import Callable, Dict, Optional

import torch


ATTENTION_BACKEND_ENV = "BITNET_ATTENTION_BACKEND"
VALID_ATTENTION_BACKENDS = {"auto", "sdpa", "flash"}


class KernelBackendError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def flash_attn_func() -> Optional[Callable]:
    try:
        from flash_attn import flash_attn_func as func
    except Exception:
        return None
    return func


def flash_attention_available() -> bool:
    return flash_attn_func() is not None


def kernel_policy() -> Dict[str, object]:
    """Describe the allowed accelerator kernels for Test 2.

    This experiment intentionally relies on maintained PyTorch/FlashAttention
    kernels and Keller Jordan's Muon optimizer package. It does not include
    custom CUDA/Triton kernels for ternary matmul, RoPE, Mamba/state-space
    layers, or any other non-attention operator.
    """

    return {
        "attention": {
            "allowed_backends": sorted(VALID_ATTENTION_BACKENDS),
            "auto_policy": "FlashAttention for compatible fixed-length CUDA BF16/FP16 causal batches; otherwise PyTorch SDPA",
            "flash_attention_installed": flash_attention_available(),
        },
        "optimizer": {
            "muon": "external KellerJordan/Muon import via BITNET_MUON_IMPORT",
            "local_foundations_muon_used": False,
        },
        "custom_kernels": {
            "custom_ternary_matmul": False,
            "custom_rope": False,
            "custom_mamba_or_state_space": False,
            "custom_attention": False,
        },
    }


def default_attention_backend() -> str:
    backend = os.environ.get(ATTENTION_BACKEND_ENV, "auto").lower()
    if backend not in VALID_ATTENTION_BACKENDS:
        raise KernelBackendError(
            f"{ATTENTION_BACKEND_ENV} must be one of {sorted(VALID_ATTENTION_BACKENDS)}, got {backend!r}"
        )
    return backend


def backend_runtime_summary() -> Dict[str, object]:
    requested = default_attention_backend()
    return {
        "requested_attention_backend": requested,
        "flash_attention_available": flash_attention_available(),
        "kernel_policy": kernel_policy(),
    }


def choose_attention_backend(
    requested: str,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
) -> str:
    requested = requested.lower()
    if requested not in VALID_ATTENTION_BACKENDS:
        raise KernelBackendError(f"unknown attention backend: {requested!r}")

    can_flash = (
        attn_mask is None
        and q.is_cuda
        and k.is_cuda
        and q.dtype in {torch.float16, torch.bfloat16}
        and k.dtype == q.dtype
        and q.shape[-1] in {16, 32, 64, 128, 256}
        and q.shape[2] % k.shape[2] == 0
        and flash_attention_available()
    )

    if requested == "flash" and not can_flash:
        reasons = []
        if attn_mask is not None:
            reasons.append("attention masks require the SDPA fallback")
        if not q.is_cuda or not k.is_cuda:
            reasons.append("FlashAttention requires CUDA tensors")
        if q.dtype not in {torch.float16, torch.bfloat16}:
            reasons.append("FlashAttention requires fp16 or bf16 tensors")
        if k.dtype != q.dtype:
            reasons.append("Q/K/V dtypes must match")
        if q.shape[-1] not in {16, 32, 64, 128, 256}:
            reasons.append(f"unsupported head_dim={q.shape[-1]}")
        if q.shape[2] % k.shape[2] != 0:
            reasons.append("query heads must be divisible by KV heads")
        if not flash_attention_available():
            reasons.append("flash_attn is not installed")
        raise KernelBackendError("Cannot use FlashAttention: " + "; ".join(reasons))

    if requested == "flash" or (requested == "auto" and can_flash):
        return "flash"
    return "sdpa"


def flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
    causal: bool,
) -> torch.Tensor:
    func = flash_attn_func()
    if func is None:
        raise KernelBackendError("flash_attn is not installed")
    return func(q, k, v, dropout_p=dropout_p, causal=causal)
