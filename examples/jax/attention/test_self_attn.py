# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Self-attention with GQA, BSHD layout, and post-scale bias.

This example demonstrates two ways to call TE's fused attention:

1. **Low-level API** — ``transformer_engine.jax.attention.fused_attn``
   Gives full control over every parameter (layout, mask type, bias type,
   sequence descriptor, scaling factor, etc.).

2. **High-level Flax module** — ``transformer_engine.jax.flax.DotProductAttention``
   Wraps the fused kernel behind a familiar Flax ``nn.Module`` interface;
   automatically falls back to unfused attention when the kernel is unavailable.

Scenarios covered
-----------------
* Standard self-attention (all heads equal)
* Grouped-Query Attention (GQA) with ``num_heads_q > num_heads_kv``
* Post-scale additive bias (``AttnBiasType.POST_SCALE_BIAS``)
* Causal masking (``AttnMaskType.CAUSAL_MASK``)
* Forward **and** backward pass (via ``jax.value_and_grad``)

Run directly::

    python test_self_attn.py

Run via pytest::

    pytest -xvs test_self_attn.py
"""
import unittest
from math import sqrt

import jax
import jax.numpy as jnp

from common import (
    is_bf16_supported,
    generate_qkv,
    reference_attention,
    make_causal_mask,
    assert_allclose,
    check_fused_attn_kernel_available,
    attn_parser,
)
from transformer_engine.jax.attention import (
    AttnBiasType,
    AttnMaskType,
    AttnSoftmaxType,
    QKVLayout,
    SequenceDescriptor,
    fused_attn,
)
import transformer_engine.jax.flax as te_flax

# ---------------------------------------------------------------------------
# Low-level API helpers
# ---------------------------------------------------------------------------


def run_fused_attn_low_level(
    q,
    k,
    v,
    bias=None,
    attn_bias_type=AttnBiasType.NO_BIAS,
    attn_mask_type=AttnMaskType.CAUSAL_MASK,
    is_training=True,
):
    """Call the low-level ``fused_attn`` API with separate Q, K, V (BSHD layout)."""
    batch = q.shape[0]
    max_seqlen_q = q.shape[1]
    max_seqlen_kv = k.shape[1]
    head_dim = q.shape[-1]
    scaling_factor = 1.0 / sqrt(head_dim)

    q_seqlens = jnp.full((batch,), max_seqlen_q, dtype=jnp.int32)
    kv_seqlens = jnp.full((batch,), max_seqlen_kv, dtype=jnp.int32)
    seq_desc = SequenceDescriptor.from_seqlens((q_seqlens, kv_seqlens))

    return fused_attn(
        qkv=(q, k, v),
        bias=bias,
        sequence_descriptor=seq_desc,
        seed=None,
        attn_bias_type=attn_bias_type,
        attn_mask_type=attn_mask_type,
        qkv_layout=QKVLayout.BSHD_BSHD_BSHD,
        softmax_type=AttnSoftmaxType.VANILLA_SOFTMAX,
        scaling_factor=scaling_factor,
        dropout_probability=0.0,
        is_training=is_training,
    )


# ---------------------------------------------------------------------------
# High-level Flax module helpers
# ---------------------------------------------------------------------------


def run_flax_dpa(
    q,
    k,
    v,
    num_heads_q,
    num_heads_kv,
    bias=None,
    attn_mask_type="causal",
    attn_bias_type=None,
):
    """Call the high-level ``DotProductAttention`` Flax module."""
    head_dim = q.shape[-1]

    dpa = te_flax.DotProductAttention(
        head_dim=head_dim,
        num_attention_heads=num_heads_q,
        num_gqa_groups=num_heads_kv,
        attn_mask_type=attn_mask_type,
        attn_bias_type=attn_bias_type,
        transpose_batch_sequence=False,
    )

    variables = dpa.init(jax.random.PRNGKey(0), q, k, v, bias=bias, deterministic=True)
    return dpa.apply(variables, q, k, v, bias=bias, deterministic=True)


# ---------------------------------------------------------------------------
# Demonstration logic
# ---------------------------------------------------------------------------


def demo_self_attn(args):
    """Run a self-attention forward + backward pass and print results."""
    dtype = jnp.bfloat16
    rng = jax.random.PRNGKey(args.seed)
    rng, qkv_rng, bias_rng = jax.random.split(rng, 3)

    q, k, v = generate_qkv(
        args.batch_size,
        args.max_seqlen_q,
        args.max_seqlen_q,
        args.num_heads_q,
        args.num_heads_kv,
        args.head_dim,
        dtype,
        qkv_rng,
    )

    bias = jax.random.normal(
        bias_rng, (1, args.num_heads_q, args.max_seqlen_q, args.max_seqlen_q), dtype=dtype
    )

    print(f"Q shape: {q.shape}  K shape: {k.shape}  V shape: {v.shape}")
    print(f"Bias shape: {bias.shape}")
    print(f"GQA: num_heads_q={args.num_heads_q}, num_heads_kv={args.num_heads_kv}")

    # -- Forward (low-level) --
    out_fused = run_fused_attn_low_level(
        q,
        k,
        v,
        bias=bias,
        attn_bias_type=AttnBiasType.POST_SCALE_BIAS,
    )
    print(f"\n[Low-level fused_attn]  output shape: {out_fused.shape}")

    # -- Forward (Flax DPA) --
    out_flax = run_flax_dpa(
        q,
        k,
        v,
        args.num_heads_q,
        args.num_heads_kv,
        bias=bias,
        attn_bias_type="post_scale_bias",
    )
    print(f"[Flax DotProductAttention] output shape: {out_flax.shape}")

    # -- Reference --
    causal_mask = make_causal_mask(args.max_seqlen_q, args.max_seqlen_q)
    out_ref = reference_attention(q, k, v, bias=bias, mask=causal_mask)
    print(f"[Reference (unfused)]   output shape: {out_ref.shape}")

    # -- Backward (low-level) --
    def fwd_loss(q_, k_, v_, bias_):
        out = run_fused_attn_low_level(
            q_, k_, v_, bias=bias_, attn_bias_type=AttnBiasType.POST_SCALE_BIAS
        )
        return jnp.mean(out)

    loss, grads = jax.value_and_grad(fwd_loss, argnums=(0, 1, 2, 3))(q, k, v, bias)
    dq, dk, dv, dbias = grads
    print(f"\nBackward pass — loss: {loss:.6f}")
    print(f"  dQ shape: {dq.shape}  dK shape: {dk.shape}  dV shape: {dv.shape}")
    print(f"  dBias shape: {dbias.shape}")
    print("PASSED")


# ---------------------------------------------------------------------------
# Test cases (pytest / unittest)
# ---------------------------------------------------------------------------

BATCH = 2
MAX_SEQLEN = 128
NUM_HEADS_Q = 12
NUM_HEADS_KV = 4
HEAD_DIM = 64
DTYPE = jnp.bfloat16


class TestSelfAttn(unittest.TestCase):
    """Self-attention fused attention examples as tests."""

    @unittest.skipIf(not is_bf16_supported(), "Device compute capability 8.0+ is required for BF16")
    def test_self_attn_bf16_with_bias(self):
        """Self-attention with causal mask and post-scale bias (BF16, MHA)."""
        num_heads = 12
        if not check_fused_attn_kernel_available(
            DTYPE,
            QKVLayout.BSHD_BSHD_BSHD,
            AttnBiasType.POST_SCALE_BIAS,
            AttnMaskType.CAUSAL_MASK,
            num_heads,
            num_heads,
            MAX_SEQLEN,
            MAX_SEQLEN,
            HEAD_DIM,
        ):
            self.skipTest("Fused attention kernel not available for this config")

        rng = jax.random.PRNGKey(0)
        rng, qkv_rng, bias_rng = jax.random.split(rng, 3)
        q, k, v = generate_qkv(
            BATCH, MAX_SEQLEN, MAX_SEQLEN, num_heads, num_heads, HEAD_DIM, DTYPE, qkv_rng
        )
        bias = jax.random.normal(bias_rng, (1, num_heads, MAX_SEQLEN, MAX_SEQLEN), dtype=DTYPE)

        out_fused = run_fused_attn_low_level(
            q, k, v, bias=bias, attn_bias_type=AttnBiasType.POST_SCALE_BIAS
        )
        causal_mask = make_causal_mask(MAX_SEQLEN, MAX_SEQLEN)
        out_ref = reference_attention(q, k, v, bias=bias, mask=causal_mask)
        assert_allclose(out_fused, out_ref, DTYPE)

        out_flax = run_flax_dpa(
            q, k, v, num_heads, num_heads, bias=bias, attn_bias_type="post_scale_bias"
        )
        assert_allclose(out_flax, out_ref, DTYPE)

    @unittest.skipIf(not is_bf16_supported(), "Device compute capability 8.0+ is required for BF16")
    def test_self_attn_bf16_no_bias(self):
        """Self-attention with causal mask, no bias (BF16, MHA)."""
        num_heads = 12
        if not check_fused_attn_kernel_available(
            DTYPE,
            QKVLayout.BSHD_BSHD_BSHD,
            AttnBiasType.NO_BIAS,
            AttnMaskType.CAUSAL_MASK,
            num_heads,
            num_heads,
            MAX_SEQLEN,
            MAX_SEQLEN,
            HEAD_DIM,
        ):
            self.skipTest("Fused attention kernel not available for this config")

        rng = jax.random.PRNGKey(1)
        q, k, v = generate_qkv(
            BATCH, MAX_SEQLEN, MAX_SEQLEN, num_heads, num_heads, HEAD_DIM, DTYPE, rng
        )

        out_fused = run_fused_attn_low_level(q, k, v)
        causal_mask = make_causal_mask(MAX_SEQLEN, MAX_SEQLEN)
        out_ref = reference_attention(q, k, v, mask=causal_mask)
        assert_allclose(out_fused, out_ref, DTYPE)

        out_flax = run_flax_dpa(q, k, v, num_heads, num_heads)
        assert_allclose(out_flax, out_ref, DTYPE)

    @unittest.skipIf(not is_bf16_supported(), "Device compute capability 8.0+ is required for BF16")
    def test_self_attn_gqa(self):
        """Self-attention with GQA (num_heads_q=12, num_heads_kv=4)."""
        if not check_fused_attn_kernel_available(
            DTYPE,
            QKVLayout.BSHD_BSHD_BSHD,
            AttnBiasType.POST_SCALE_BIAS,
            AttnMaskType.CAUSAL_MASK,
            NUM_HEADS_Q,
            NUM_HEADS_KV,
            MAX_SEQLEN,
            MAX_SEQLEN,
            HEAD_DIM,
        ):
            self.skipTest("Fused attention kernel not available for this config")

        rng = jax.random.PRNGKey(2)
        rng, qkv_rng, bias_rng = jax.random.split(rng, 3)
        q, k, v = generate_qkv(
            BATCH, MAX_SEQLEN, MAX_SEQLEN, NUM_HEADS_Q, NUM_HEADS_KV, HEAD_DIM, DTYPE, qkv_rng
        )
        bias = jax.random.normal(bias_rng, (1, NUM_HEADS_Q, MAX_SEQLEN, MAX_SEQLEN), dtype=DTYPE)

        out_fused = run_fused_attn_low_level(
            q, k, v, bias=bias, attn_bias_type=AttnBiasType.POST_SCALE_BIAS
        )
        causal_mask = make_causal_mask(MAX_SEQLEN, MAX_SEQLEN)
        out_ref = reference_attention(q, k, v, bias=bias, mask=causal_mask)
        assert_allclose(out_fused, out_ref, DTYPE)

        out_flax = run_flax_dpa(
            q,
            k,
            v,
            NUM_HEADS_Q,
            NUM_HEADS_KV,
            bias=bias,
            attn_bias_type="post_scale_bias",
        )
        assert_allclose(out_flax, out_ref, DTYPE)

    @unittest.skipIf(not is_bf16_supported(), "Device compute capability 8.0+ is required for BF16")
    def test_self_attn_backward(self):
        """Backward pass: gradient shapes and finiteness for fused self-attention with GQA."""
        if not check_fused_attn_kernel_available(
            DTYPE,
            QKVLayout.BSHD_BSHD_BSHD,
            AttnBiasType.POST_SCALE_BIAS,
            AttnMaskType.CAUSAL_MASK,
            NUM_HEADS_Q,
            NUM_HEADS_KV,
            MAX_SEQLEN,
            MAX_SEQLEN,
            HEAD_DIM,
        ):
            self.skipTest("Fused attention kernel not available for this config")

        rng = jax.random.PRNGKey(3)
        rng, qkv_rng, bias_rng = jax.random.split(rng, 3)
        q, k, v = generate_qkv(
            BATCH, MAX_SEQLEN, MAX_SEQLEN, NUM_HEADS_Q, NUM_HEADS_KV, HEAD_DIM, DTYPE, qkv_rng
        )
        bias = jax.random.normal(bias_rng, (1, NUM_HEADS_Q, MAX_SEQLEN, MAX_SEQLEN), dtype=DTYPE)

        def fwd(q_, k_, v_, bias_):
            return jnp.mean(
                run_fused_attn_low_level(
                    q_, k_, v_, bias=bias_, attn_bias_type=AttnBiasType.POST_SCALE_BIAS
                )
            )

        loss, grads = jax.value_and_grad(fwd, argnums=(0, 1, 2, 3))(q, k, v, bias)
        dq, dk, dv, dbias = grads

        self.assertEqual(dq.shape, q.shape)
        self.assertEqual(dk.shape, k.shape)
        self.assertEqual(dv.shape, v.shape)
        self.assertEqual(dbias.shape, bias.shape)
        self.assertTrue(jnp.isfinite(loss))
        self.assertTrue(jnp.all(jnp.isfinite(dq)))
        self.assertTrue(jnp.all(jnp.isfinite(dk)))
        self.assertTrue(jnp.all(jnp.isfinite(dv)))


if __name__ == "__main__":
    demo_self_attn(attn_parser(["--num-heads-kv", "4"]))
