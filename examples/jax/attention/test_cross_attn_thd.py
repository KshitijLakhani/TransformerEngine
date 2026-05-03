# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Cross-attention with THD (packed) sequences and sliding window.

This example demonstrates:

1. **THD format** — Multiple variable-length segments packed into a single
   ``[batch, max_seqlen, heads, dim]`` tensor, identified by segment IDs,
   positions, sequence lengths, and offsets.

2. **Cross-attention** — Query and Key/Value come from different sources
   and may have different sequence lengths (``max_seqlen_q != max_seqlen_kv``).

3. **Sliding window attention (SWA)** — Restrict each query to attend only to
   a local window of keys via ``window_size=(left, 0)`` (causal + limited left
   context).

Both the low-level ``fused_attn()`` and the high-level Flax
``DotProductAttention`` are shown.

Run directly::

    python test_cross_attn_thd.py

Run via pytest::

    pytest -xvs test_cross_attn_thd.py
"""
import unittest
from math import sqrt

import jax
import jax.numpy as jnp
import numpy as np

from common import (
    is_bf16_supported,
    assert_allclose,
    check_fused_attn_kernel_available,
    reference_attention,
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
# THD helpers
# ---------------------------------------------------------------------------


def build_thd_segments(batch_size, max_seqlen, num_segments, seed=42):
    """Build simple THD segment metadata.

    Returns:
        segment_ids: [batch, max_seqlen] — 1-indexed segment IDs, 0 = padding
        segment_pos: [batch, max_seqlen] — position within each segment
        seqlens:     [batch, max_seqlen] — per-segment lengths, unused slots = -1
        offsets:     [batch, max_seqlen + 1] — per-segment start offsets, unused = -1
    """
    rng = np.random.default_rng(seed=seed)
    segment_ids = np.zeros((batch_size, max_seqlen), dtype=np.int32)
    segment_pos = np.zeros((batch_size, max_seqlen), dtype=np.int32)

    max_segment_size = max_seqlen // num_segments
    all_seqlens = np.full((batch_size, max_seqlen), -1, dtype=np.int32)
    all_offsets = np.full((batch_size, max_seqlen + 1), -1, dtype=np.int32)

    for i in range(batch_size):
        pos = 0
        seg_id = 1
        seg_lens = []
        seg_offs = []
        for _ in range(num_segments):
            seg_size = rng.integers(1, max_segment_size + 1)
            if pos + seg_size > max_seqlen:
                break
            segment_ids[i, pos : pos + seg_size] = seg_id
            segment_pos[i, pos : pos + seg_size] = np.arange(seg_size)
            seg_offs.append(pos)
            seg_lens.append(seg_size)
            pos += seg_size
            seg_id += 1
        seg_offs.append(pos)
        for j, sl in enumerate(seg_lens):
            all_seqlens[i, j] = sl
        for j, off in enumerate(seg_offs):
            all_offsets[i, j] = off

    return (
        jnp.asarray(segment_ids),
        jnp.asarray(segment_pos),
        jnp.asarray(all_seqlens),
        jnp.asarray(all_offsets),
    )


def thd_reference_mask(segment_ids_q, segment_pos_q, segment_ids_kv, segment_pos_kv, is_causal):
    """Build a dense boolean mask from THD segment metadata.

    Returns mask where True = masked out (cannot attend).
    Shape: [batch, 1, max_seqlen_q, max_seqlen_kv]
    """
    # Tokens attend only within the same segment (same nonzero segment ID).
    seg_match = (segment_ids_q[:, :, None] == segment_ids_kv[:, None, :]) & (
        segment_ids_q[:, :, None] != 0
    )
    if is_causal:
        causal = segment_pos_q[:, :, None] >= segment_pos_kv[:, None, :]
        attend = seg_match & causal
    else:
        attend = seg_match
    # True = masked out
    return ~attend[:, None, :, :]


# ---------------------------------------------------------------------------
# Low-level API
# ---------------------------------------------------------------------------


def run_fused_attn_thd(
    q,
    k,
    v,
    seq_desc,
    attn_mask_type=AttnMaskType.PADDING_CAUSAL_MASK,
    max_segments_per_seq=4,
    window_size=None,
    is_training=True,
):
    """Call low-level ``fused_attn`` with THD layout."""
    head_dim = q.shape[-1]
    scaling_factor = 1.0 / sqrt(head_dim)

    return fused_attn(
        qkv=(q, k, v),
        bias=None,
        sequence_descriptor=seq_desc,
        seed=None,
        attn_bias_type=AttnBiasType.NO_BIAS,
        attn_mask_type=attn_mask_type,
        qkv_layout=QKVLayout.THD_THD_THD,
        softmax_type=AttnSoftmaxType.VANILLA_SOFTMAX,
        scaling_factor=scaling_factor,
        dropout_probability=0.0,
        is_training=is_training,
        max_segments_per_seq=max_segments_per_seq,
        window_size=window_size,
    )


# ---------------------------------------------------------------------------
# High-level Flax API
# ---------------------------------------------------------------------------


def run_flax_dpa_thd(
    q,
    k,
    v,
    num_heads_q,
    num_heads_kv,
    seq_desc,
    max_segments_per_seq=4,
    window_size=None,
):
    """Call high-level ``DotProductAttention`` with THD layout."""
    head_dim = q.shape[-1]
    dpa = te_flax.DotProductAttention(
        head_dim=head_dim,
        num_attention_heads=num_heads_q,
        num_gqa_groups=num_heads_kv,
        attn_mask_type="padding_causal",
        qkv_layout="thd_thd_thd",
        max_segments_per_seq=max_segments_per_seq,
        window_size=window_size,
        transpose_batch_sequence=False,
    )
    variables = dpa.init(
        jax.random.PRNGKey(0), q, k, v, sequence_descriptor=seq_desc, deterministic=True
    )
    return dpa.apply(variables, q, k, v, sequence_descriptor=seq_desc, deterministic=True)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def demo_cross_attn_thd(args):
    """Run cross-attention with THD packing and print results."""
    dtype = jnp.bfloat16
    batch = args.batch_size
    max_seqlen_q = args.max_seqlen_q
    max_seqlen_kv = args.max_seqlen_kv
    num_heads = args.num_heads_q
    head_dim = args.head_dim
    num_segments = 3

    rng = jax.random.PRNGKey(args.seed)
    k1, k2, k3 = jax.random.split(rng, 3)
    q = jax.random.normal(k1, (batch, max_seqlen_q, num_heads, head_dim), dtype=dtype)
    k = jax.random.normal(k2, (batch, max_seqlen_kv, num_heads, head_dim), dtype=dtype)
    v = jax.random.normal(k3, (batch, max_seqlen_kv, num_heads, head_dim), dtype=dtype)

    seg_ids_q, seg_pos_q, seqlens_q, offsets_q = build_thd_segments(
        batch, max_seqlen_q, num_segments, seed=42
    )
    _, _, seqlens_kv, offsets_kv = build_thd_segments(
        batch, max_seqlen_kv, num_segments, seed=99
    )

    print("=== THD Cross-Attention Example ===")
    print(f"Q shape: {q.shape}  K shape: {k.shape}  V shape: {v.shape}")
    print(f"Segment IDs (Q, batch 0): {seg_ids_q[0]}")
    print(f"Segment Pos (Q, batch 0): {seg_pos_q[0]}")
    print(f"Seqlens     (Q, batch 0): {seqlens_q[0]}")
    print(f"Offsets     (Q, batch 0): {offsets_q[0]}")

    seq_desc = SequenceDescriptor.from_seqlens_and_offsets(
        seqlens=(seqlens_q, seqlens_kv),
        seq_offsets=(offsets_q, offsets_kv),
    )

    # -- No sliding window --
    out = run_fused_attn_thd(q, k, v, seq_desc, max_segments_per_seq=num_segments)
    print(f"\n[Low-level fused_attn THD] output shape: {out.shape}")

    out_flax = run_flax_dpa_thd(q, k, v, num_heads, num_heads, seq_desc, num_segments)
    print(f"[Flax DPA THD]            output shape: {out_flax.shape}")

    # -- With sliding window --
    window_size = (max_seqlen_kv // 4, 0)
    out_swa = run_fused_attn_thd(
        q, k, v, seq_desc, max_segments_per_seq=num_segments, window_size=window_size
    )
    print(f"\n[Fused THD + SWA (left={window_size[0]})] output shape: {out_swa.shape}")

    # -- Backward --
    def fwd_loss(q_, k_, v_):
        return jnp.mean(run_fused_attn_thd(q_, k_, v_, seq_desc, max_segments_per_seq=num_segments))

    loss, grads = jax.value_and_grad(fwd_loss, argnums=(0, 1, 2))(q, k, v)
    dq, dk, dv = grads
    print(f"\nBackward — loss: {loss:.6f}")
    print(f"  dQ: {dq.shape}  dK: {dk.shape}  dV: {dv.shape}")
    print("PASSED")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

BATCH = 2
MAX_SEQLEN_Q = 128
MAX_SEQLEN_KV = 256
NUM_HEADS = 8
HEAD_DIM = 64
NUM_SEGMENTS = 3
DTYPE = jnp.bfloat16


class TestCrossAttnTHD(unittest.TestCase):
    """Cross-attention with THD packing examples as tests."""

    @classmethod
    def setUpClass(cls):
        """Build shared THD segment metadata and QKV tensors."""
        cls.seg_ids_q, cls.seg_pos_q, cls.seqlens_q, cls.offsets_q = build_thd_segments(
            BATCH, MAX_SEQLEN_Q, NUM_SEGMENTS, seed=42
        )
        cls.seg_ids_kv, cls.seg_pos_kv, cls.seqlens_kv, cls.offsets_kv = build_thd_segments(
            BATCH, MAX_SEQLEN_KV, NUM_SEGMENTS, seed=99
        )
        cls.seq_desc = SequenceDescriptor.from_seqlens_and_offsets(
            seqlens=(cls.seqlens_q, cls.seqlens_kv),
            seq_offsets=(cls.offsets_q, cls.offsets_kv),
        )
        rng = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(rng, 3)
        cls.q = jax.random.normal(k1, (BATCH, MAX_SEQLEN_Q, NUM_HEADS, HEAD_DIM), dtype=DTYPE)
        cls.k = jax.random.normal(k2, (BATCH, MAX_SEQLEN_KV, NUM_HEADS, HEAD_DIM), dtype=DTYPE)
        cls.v = jax.random.normal(k3, (BATCH, MAX_SEQLEN_KV, NUM_HEADS, HEAD_DIM), dtype=DTYPE)

    def _skip_if_unavailable(self, window_size=None):
        if not check_fused_attn_kernel_available(
            DTYPE,
            QKVLayout.THD_THD_THD,
            AttnBiasType.NO_BIAS,
            AttnMaskType.PADDING_CAUSAL_MASK,
            NUM_HEADS,
            NUM_HEADS,
            MAX_SEQLEN_Q,
            MAX_SEQLEN_KV,
            HEAD_DIM,
            window_size=window_size,
        ):
            self.skipTest("Fused attention kernel not available for this THD config")

    @unittest.skipIf(not is_bf16_supported(), "Device compute capability 8.0+ is required for BF16")
    def test_cross_attn_thd_bf16(self):
        """Cross-attention with THD packing, padding+causal mask (BF16)."""
        self._skip_if_unavailable()

        out_fused = run_fused_attn_thd(
            self.q, self.k, self.v, self.seq_desc, max_segments_per_seq=NUM_SEGMENTS
        )
        self.assertEqual(out_fused.shape, self.q.shape)
        self.assertTrue(jnp.all(jnp.isfinite(out_fused)))

        mask = thd_reference_mask(
            self.seg_ids_q, self.seg_pos_q, self.seg_ids_kv, self.seg_pos_kv, is_causal=True
        )
        out_ref = reference_attention(self.q, self.k, self.v, mask=mask)
        # Padded positions may differ; compare only valid Q positions.
        valid_q = self.seg_ids_q != 0
        out_fused_valid = jnp.where(valid_q[..., None, None], out_fused, 0)
        out_ref_valid = jnp.where(valid_q[..., None, None], out_ref, 0)
        assert_allclose(out_fused_valid, out_ref_valid, DTYPE)

    @unittest.skipIf(not is_bf16_supported(), "Device compute capability 8.0+ is required for BF16")
    def test_cross_attn_thd_flax(self):
        """Cross-attention with THD via Flax DotProductAttention (BF16)."""
        self._skip_if_unavailable()

        out_flax = run_flax_dpa_thd(
            self.q, self.k, self.v, NUM_HEADS, NUM_HEADS, self.seq_desc, NUM_SEGMENTS
        )
        self.assertEqual(out_flax.shape, self.q.shape)
        self.assertTrue(jnp.all(jnp.isfinite(out_flax)))

    @unittest.skipIf(not is_bf16_supported(), "Device compute capability 8.0+ is required for BF16")
    def test_cross_attn_thd_sliding_window(self):
        """Cross-attention with THD + sliding window (BF16)."""
        window_size = (MAX_SEQLEN_KV // 4, 0)
        self._skip_if_unavailable(window_size=window_size)

        out_swa = run_fused_attn_thd(
            self.q,
            self.k,
            self.v,
            self.seq_desc,
            max_segments_per_seq=NUM_SEGMENTS,
            window_size=window_size,
        )
        self.assertEqual(out_swa.shape, self.q.shape)
        self.assertTrue(jnp.all(jnp.isfinite(out_swa)))

        # Also via Flax
        out_flax_swa = run_flax_dpa_thd(
            self.q,
            self.k,
            self.v,
            NUM_HEADS,
            NUM_HEADS,
            self.seq_desc,
            NUM_SEGMENTS,
            window_size=window_size,
        )
        self.assertEqual(out_flax_swa.shape, self.q.shape)

    @unittest.skipIf(not is_bf16_supported(), "Device compute capability 8.0+ is required for BF16")
    def test_cross_attn_thd_backward(self):
        """Backward pass through THD cross-attention."""
        self._skip_if_unavailable()

        seq_desc = self.seq_desc

        def fwd(q_, k_, v_):
            return jnp.mean(
                run_fused_attn_thd(q_, k_, v_, seq_desc, max_segments_per_seq=NUM_SEGMENTS)
            )

        loss, grads = jax.value_and_grad(fwd, argnums=(0, 1, 2))(self.q, self.k, self.v)
        dq, dk, dv = grads
        self.assertEqual(dq.shape, self.q.shape)
        self.assertEqual(dk.shape, self.k.shape)
        self.assertEqual(dv.shape, self.v.shape)
        self.assertTrue(jnp.isfinite(loss))


if __name__ == "__main__":
    demo_cross_attn_thd(
        attn_parser(["--max-seqlen-q", "128", "--max-seqlen-kv", "256", "--num-heads-q", "8"])
    )
