# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Context parallelism for fused attention across multiple GPUs.

Context parallelism (CP) shards long sequences across devices so that each
device only materializes a fraction of the sequence in Q/K/V.  The fused
attention kernel communicates (all-gather or ring) to compute the full
attention output without replicating the full sequence on any single device.

This example demonstrates:

1. **ALL_GATHER strategy** — ``CPStrategy.ALL_GATHER``: each device gathers
   the full K/V from peers and computes its local Q chunk.

2. **RING strategy** — ``CPStrategy.RING``: ring-attention style communication
   where K/V chunks are forwarded around a ring.

3. **Causal load balancing** — ``reorder_causal_load_balancing`` with
   ``DualChunkSwap`` reorders the sequence so that each device gets a roughly
   equal mix of early and late tokens, avoiding the load imbalance inherent in
   causal masking.

Both the low-level ``fused_attn()`` and the high-level Flax
``DotProductAttention`` are shown.

Requires multiple GPUs. Run via the launcher script::

    bash run_test_context_parallel.sh

Or manually for a 2-GPU test::

    pytest -xvs test_context_parallel.py::TestContextParallel::test_cp_all_gather \\
        --num-process=2 --process-id=0

    # (in another terminal, same machine)
    pytest -xvs test_context_parallel.py::TestContextParallel::test_cp_all_gather \\
        --num-process=2 --process-id=1
"""
import argparse
import os
import unittest
from functools import partial
from math import sqrt

import pytest
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from common import (
    is_bf16_supported,
    generate_qkv,
    check_fused_attn_kernel_available,
)
from transformer_engine.jax import autocast
from transformer_engine.jax.sharding import MeshResource
from transformer_engine.jax.attention import (
    AttnBiasType,
    AttnMaskType,
    AttnSoftmaxType,
    QKVLayout,
    CPStrategy,
    ReorderStrategy,
    SequenceDescriptor,
    fused_attn,
    reorder_causal_load_balancing,
    inverse_reorder_causal_load_balancing,
)
import transformer_engine.jax.flax as te_flax

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

CP_AXIS = "cp"


# ---------------------------------------------------------------------------
# Low-level API
# ---------------------------------------------------------------------------


def run_fused_attn_cp(
    q,
    k,
    v,
    cp_strategy,
    load_balanced=False,
    is_training=True,
):
    """Call low-level ``fused_attn`` with context parallelism."""
    batch = q.shape[0]
    max_seqlen = q.shape[1]
    head_dim = q.shape[-1]
    scaling_factor = 1.0 / sqrt(head_dim)

    q_seqlens = jnp.full((batch,), max_seqlen, dtype=jnp.int32)
    kv_seqlens = jnp.full((batch,), max_seqlen, dtype=jnp.int32)
    seq_desc = SequenceDescriptor.from_seqlens((q_seqlens, kv_seqlens))

    return fused_attn(
        qkv=(q, k, v),
        bias=None,
        sequence_descriptor=seq_desc,
        seed=None,
        attn_bias_type=AttnBiasType.NO_BIAS,
        attn_mask_type=AttnMaskType.CAUSAL_MASK,
        qkv_layout=QKVLayout.BSHD_BSHD_BSHD,
        softmax_type=AttnSoftmaxType.VANILLA_SOFTMAX,
        scaling_factor=scaling_factor,
        dropout_probability=0.0,
        is_training=is_training,
        context_parallel_strategy=cp_strategy,
        context_parallel_causal_load_balanced=load_balanced,
        context_parallel_axis=CP_AXIS,
    )


# ---------------------------------------------------------------------------
# High-level Flax API
# ---------------------------------------------------------------------------


def run_flax_dpa_cp(
    q,
    k,
    v,
    num_heads,
    cp_strategy_str,
    load_balanced=False,
):
    """Call high-level ``DotProductAttention`` with context parallelism."""
    head_dim = q.shape[-1]
    dpa = te_flax.DotProductAttention(
        head_dim=head_dim,
        num_attention_heads=num_heads,
        attn_mask_type="causal",
        transpose_batch_sequence=False,
        context_parallel_axis=CP_AXIS,
        context_parallel_strategy=cp_strategy_str,
        context_parallel_causal_load_balanced=load_balanced,
    )
    variables = dpa.init(jax.random.PRNGKey(0), q, k, v, deterministic=True)
    return dpa.apply(variables, q, k, v, deterministic=True)


# ---------------------------------------------------------------------------
# Multi-process initialization and execution
# ---------------------------------------------------------------------------


def run_cp_example(args, cp_strategy, load_balanced=False):
    """Run a context-parallel attention forward pass.

    This function is designed to be called from each process (one GPU each)
    after ``jax.distributed.initialize()``.
    """
    dtype = jnp.bfloat16
    batch = 2
    # Total sequence length must be divisible by cp_size
    total_seqlen = 512
    num_heads = 8
    head_dim = 64

    cp_size = args.num_process

    if not check_fused_attn_kernel_available(
        dtype,
        QKVLayout.BSHD_BSHD_BSHD,
        AttnBiasType.NO_BIAS,
        AttnMaskType.CAUSAL_MASK,
        num_heads,
        num_heads,
        total_seqlen,
        total_seqlen,
        head_dim,
    ):
        print("SKIPPED — Fused attention kernel not available")
        return

    device_mesh = mesh_utils.create_device_mesh((cp_size,))
    mesh = Mesh(device_mesh, axis_names=(CP_AXIS,))
    mesh_resource = MeshResource(cp_resource=CP_AXIS)

    rng = jax.random.PRNGKey(42)
    q, k, v = generate_qkv(batch, total_seqlen, total_seqlen, num_heads, num_heads, head_dim, dtype, rng)

    # Apply causal load balancing reorder if requested
    if load_balanced:
        reorder_fn = partial(
            reorder_causal_load_balancing,
            strategy=ReorderStrategy.DualChunkSwap,
            cp_size=cp_size,
            seq_dim=1,
        )
        q = reorder_fn(q)
        k = reorder_fn(k)
        v = reorder_fn(v)

    # Shard QKV along the sequence dimension across CP devices
    qkv_pspec = PartitionSpec(None, CP_AXIS, None, None)
    qkv_sharding = NamedSharding(mesh, qkv_pspec)
    q = jax.device_put(q, qkv_sharding)
    k = jax.device_put(k, qkv_sharding)
    v = jax.device_put(v, qkv_sharding)

    # -- Low-level fused_attn with CP --
    with mesh, autocast(mesh_resource=mesh_resource):
        fwd_fn = jax.jit(
            partial(run_fused_attn_cp, cp_strategy=cp_strategy, load_balanced=load_balanced),
            in_shardings=[qkv_sharding, qkv_sharding, qkv_sharding],
        )
        out = fwd_fn(q, k, v)

    if load_balanced:
        inverse_reorder_fn = partial(
            inverse_reorder_causal_load_balancing,
            strategy=ReorderStrategy.DualChunkSwap,
            cp_size=cp_size,
            seq_dim=1,
        )
        out = inverse_reorder_fn(out)

    if args.process_id == 0:
        strategy_name = cp_strategy.name
        lb_str = "+load_balanced" if load_balanced else ""
        print(f"[Low-level fused_attn CP={strategy_name}{lb_str}]  output shape: {out.shape}")

    # -- High-level Flax DPA with CP --
    cp_strategy_str = "ALL_GATHER" if cp_strategy == CPStrategy.ALL_GATHER else "RING"
    with mesh, autocast(mesh_resource=mesh_resource):
        flax_fn = jax.jit(
            partial(
                run_flax_dpa_cp,
                num_heads=num_heads,
                cp_strategy_str=cp_strategy_str,
                load_balanced=load_balanced,
            ),
            in_shardings=[qkv_sharding, qkv_sharding, qkv_sharding],
        )
        out_flax = flax_fn(q, k, v)

    if args.process_id == 0:
        print(f"[Flax DPA CP={cp_strategy_str}{lb_str}]  output shape: {out_flax.shape}")
        assert jnp.all(jnp.isfinite(out)), "Output contains non-finite values"
        assert jnp.all(jnp.isfinite(out_flax)), "Flax output contains non-finite values"
        print("PASSED")


# ---------------------------------------------------------------------------
# Test cases (multi-process pytest)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("multiprocessing_parses")
class TestContextParallel(unittest.TestCase):
    """Context-parallel fused attention examples as tests."""

    def _init_distributed(self):
        """Initialize JAX distributed runtime (one GPU per process)."""
        jax.distributed.initialize(
            coordinator_address="127.0.0.1:1234",
            num_processes=self.num_process,
            process_id=self.process_id,
            local_device_ids=self.process_id,
        )
        assert jax.local_device_count() == 1, "Expected 1 GPU per process"

    def _shutdown_distributed(self):
        jax.distributed.shutdown()

    def _make_args(self):
        args = argparse.Namespace(
            num_process=self.num_process,
            process_id=self.process_id,
        )
        return args

    @unittest.skipIf(not is_bf16_supported(), "Device compute capability 8.0+ is required for BF16")
    def test_cp_all_gather(self):
        """Context parallelism with ALL_GATHER strategy."""
        self._init_distributed()
        try:
            run_cp_example(self._make_args(), CPStrategy.ALL_GATHER)
        finally:
            self._shutdown_distributed()

    @unittest.skipIf(not is_bf16_supported(), "Device compute capability 8.0+ is required for BF16")
    def test_cp_ring(self):
        """Context parallelism with RING strategy."""
        self._init_distributed()
        try:
            run_cp_example(self._make_args(), CPStrategy.RING)
        finally:
            self._shutdown_distributed()

    @unittest.skipIf(not is_bf16_supported(), "Device compute capability 8.0+ is required for BF16")
    def test_cp_load_balanced(self):
        """Context parallelism with ALL_GATHER + causal load balancing."""
        self._init_distributed()
        try:
            run_cp_example(self._make_args(), CPStrategy.ALL_GATHER, load_balanced=True)
        finally:
            self._shutdown_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Context Parallel Attention Example")
    parser.add_argument(
        "--coordinator-address",
        type=str,
        default="127.0.0.1:1234",
        help="coordinator address (default: 127.0.0.1:1234)",
    )
    parser.add_argument("--num-process", type=int, default=2, help="number of processes")
    parser.add_argument("--process-id", type=int, default=0, help="process ID")
    parser.add_argument(
        "--strategy",
        type=str,
        default="all_gather",
        choices=["all_gather", "ring"],
        help="CP strategy",
    )
    parser.add_argument(
        "--load-balanced",
        action="store_true",
        default=False,
        help="enable causal load balancing",
    )
    args = parser.parse_args()

    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_process,
        process_id=args.process_id,
        local_device_ids=args.process_id,
    )

    strategy = CPStrategy.ALL_GATHER if args.strategy == "all_gather" else CPStrategy.RING
    run_cp_example(args, strategy, load_balanced=args.load_balanced)

    jax.distributed.shutdown()
