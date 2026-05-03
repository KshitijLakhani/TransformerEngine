# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Shared utilities for the fused attention examples"""
import argparse
from functools import lru_cache
from math import sqrt

import jax
import jax.numpy as jnp
import numpy as np
from transformer_engine_jax import get_device_compute_capability

from transformer_engine.jax.attention import (
    AttnSoftmaxType,
    is_fused_attn_kernel_available,
)


@lru_cache
def is_bf16_supported():
    """Return if BF16 has hardware support"""
    gpu_arch = get_device_compute_capability(0)
    return gpu_arch >= 80


@lru_cache
def is_fp16_supported():
    """Return if FP16 fused attention has hardware support"""
    gpu_arch = get_device_compute_capability(0)
    return gpu_arch >= 80


def check_fused_attn_kernel_available(
    dtype,
    qkv_layout,
    attn_bias_type,
    attn_mask_type,
    num_heads_q,
    num_heads_kv,
    max_seqlen_q,
    max_seqlen_kv,
    head_dim,
    dropout_probability=0.0,
    is_training=True,
    window_size=None,
):
    """Check whether a fused attention kernel is available for the given config."""
    return is_fused_attn_kernel_available(
        is_training=is_training,
        q_dtype=dtype,
        kv_dtype=dtype,
        qkv_layout=qkv_layout,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        softmax_type=AttnSoftmaxType.VANILLA_SOFTMAX,
        dropout_probability=dropout_probability,
        q_num_heads=num_heads_q,
        kv_num_heads=num_heads_kv,
        q_max_seqlen=max_seqlen_q,
        kv_max_seqlen=max_seqlen_kv,
        head_dim_qk=head_dim,
        head_dim_v=head_dim,
        window_size=window_size,
    )


def generate_qkv(
    batch, max_seqlen_q, max_seqlen_kv, num_heads_q, num_heads_kv, head_dim, dtype, rng_key
):
    """Generate random Q, K, V tensors in BSHD format."""
    k1, k2, k3 = jax.random.split(rng_key, 3)
    q = jax.random.normal(k1, (batch, max_seqlen_q, num_heads_q, head_dim), dtype=dtype)
    k = jax.random.normal(k2, (batch, max_seqlen_kv, num_heads_kv, head_dim), dtype=dtype)
    v = jax.random.normal(k3, (batch, max_seqlen_kv, num_heads_kv, head_dim), dtype=dtype)
    return q, k, v


def reference_attention(query, key, value, bias=None, mask=None, scale=None, dtype=jnp.float32):
    """
    Simple unfused multi-head attention reference in JAX.

    Supports GQA by repeating KV heads to match Q heads.

    Args:
        query: [b, s_q, h_q, d]
        key:   [b, s_kv, h_kv, d]
        value: [b, s_kv, h_kv, d]
        bias:  broadcastable to [b, h_q, s_q, s_kv] or None
        mask:  [b, 1, s_q, s_kv] boolean where True = masked out, or None
        scale: scaling factor (default: 1/sqrt(d))
        dtype: computation dtype
    """
    d = query.shape[-1]
    if scale is None:
        scale = 1.0 / sqrt(d)

    q = query.astype(dtype)
    k = key.astype(dtype)
    v = value.astype(dtype)

    h_q = q.shape[2]
    h_kv = k.shape[2]
    if h_q != h_kv:
        num_groups = h_q // h_kv
        k = jnp.repeat(k, num_groups, axis=2)
        v = jnp.repeat(v, num_groups, axis=2)

    # [b, h, s_q, d] x [b, h, d, s_kv] -> [b, h, s_q, s_kv]
    q = jnp.transpose(q, (0, 2, 1, 3))
    k = jnp.transpose(k, (0, 2, 1, 3))
    v = jnp.transpose(v, (0, 2, 1, 3))

    logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale

    if bias is not None:
        logits = logits + bias.astype(dtype)

    if mask is not None:
        logits = jnp.where(mask, jnp.finfo(dtype).min, logits)

    weights = jax.nn.softmax(logits, axis=-1)

    # [b, h, s_q, s_kv] x [b, h, s_kv, d] -> [b, h, s_q, d]
    out = jnp.matmul(weights, v)

    # [b, h, s_q, d] -> [b, s_q, h, d]
    out = jnp.transpose(out, (0, 2, 1, 3))
    return out.astype(query.dtype)


def make_causal_mask(seq_q, seq_kv):
    """Create a causal mask where True = masked out (cannot attend).

    Returns shape [1, 1, seq_q, seq_kv].
    """
    row_idx = jnp.arange(seq_q)[:, None]
    col_idx = jnp.arange(seq_kv)[None, :]
    mask = col_idx > row_idx
    return mask[None, None, :, :]


def assert_allclose(actual, expected, dtype, rtol=None, atol=None):
    """Assert tensors are close with dtype-appropriate tolerances."""
    if rtol is None:
        rtol = 5e-2 if dtype in (jnp.bfloat16, jnp.float16) else 1e-5
    if atol is None:
        atol = 5e-2 if dtype in (jnp.bfloat16, jnp.float16) else 1e-5
    np.testing.assert_allclose(
        np.asarray(actual, dtype=np.float32),
        np.asarray(expected, dtype=np.float32),
        rtol=rtol,
        atol=atol,
    )


def attn_parser(args=None):
    """Argument parser with common attention example flags."""
    parser = argparse.ArgumentParser(description="JAX Fused Attention Example")
    parser.add_argument("--batch-size", type=int, default=2, help="batch size (default: 2)")
    parser.add_argument(
        "--max-seqlen-q", type=int, default=128, help="max query sequence length (default: 128)"
    )
    parser.add_argument(
        "--max-seqlen-kv",
        type=int,
        default=128,
        help="max key/value sequence length (default: 128)",
    )
    parser.add_argument(
        "--num-heads-q", type=int, default=12, help="number of query heads (default: 12)"
    )
    parser.add_argument(
        "--num-heads-kv",
        type=int,
        default=12,
        help="number of key/value heads (default: 12, set < num-heads-q for GQA)",
    )
    parser.add_argument("--head-dim", type=int, default=64, help="head dimension (default: 64)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    return parser.parse_args(args)
