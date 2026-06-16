#!/usr/bin/env python3
"""Benchmark TE JAX attention with TE CP variants and JAX-native attention."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad
from jax.sharding import Mesh, NamedSharding, PartitionSpec, SingleDeviceSharding

import transformer_engine
try:
    import flax.linen as flax_linen
except ImportError:  # pragma: no cover - optional benchmark backend
    flax_linen = None
from transformer_engine.jax import autocast
from transformer_engine.jax.attention import (
    AttnBiasType,
    AttnMaskType,
    AttnSoftmaxType,
    CPStrategy,
    QKVLayout,
    ReorderStrategy,
    SequenceDescriptor,
    fused_attn,
    is_fused_attn_kernel_available,
    reorder_causal_load_balancing,
)
from transformer_engine.jax.sharding import MeshResource
from transformer_engine_jax import get_cudnn_version, get_device_compute_capability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("te_fused", "jax_native", "flax_linen"), default="te_fused")
    parser.add_argument(
        "--jax-attn-implementation",
        choices=("xla", "cudnn"),
        default="xla",
        help="Implementation passed to jax.nn.dot_product_attention for --backend=jax_native.",
    )
    parser.add_argument("--layout", choices=("bshd", "thd"), required=True)
    parser.add_argument("--cp", choices=("none", "ring", "all_gather"), default="none")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=32768)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128, help="Default head dim for both QK and V.")
    parser.add_argument("--qk-dim", type=int, help="Head dim for Q and K. Defaults to --dim.")
    parser.add_argument("--v-dim", type=int, help="Head dim for V and output. Defaults to --dim.")
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--cp-size", type=int)
    parser.add_argument("--stripe-size", type=int)
    parser.add_argument(
        "--sharded-input-callback",
        action="store_true",
        help="Create NamedSharding inputs with jax.make_array_from_callback instead of a jitted full constant.",
    )
    parser.add_argument(
        "--skip-qkv-reorder",
        action="store_true",
        help="Skip CP q/k/v/dout reorder. Only valid for this constant-input timing harness.",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--mode", choices=("fwd", "fwd_bwd"), default="fwd_bwd")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--cuda-profiler-api", action="store_true")
    parser.add_argument("--check-backend", action="store_true", default=True)
    parser.add_argument("--no-check-backend", dest="check_backend", action="store_false")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument(
        "--coordinator-address",
        default=os.environ.get("JAX_COORDINATOR_ADDRESS", "127.0.0.1:12345"),
    )
    parser.add_argument(
        "--num-processes", type=int, default=int(os.environ.get("JAX_NUM_PROCESSES", "1"))
    )
    parser.add_argument("--process-id", type=int, default=int(os.environ.get("JAX_PROCESS_ID", "0")))
    parser.add_argument("--local-device-ids", default=os.environ.get("JAX_LOCAL_DEVICE_IDS"))
    return parser.parse_args()


def block_until_ready(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - idx) + ordered[hi] * (idx - lo)


def get_dtype(name: str) -> jnp.dtype:
    if name == "bf16":
        return jnp.bfloat16
    if name == "fp16":
        return jnp.float16
    raise ValueError(f"Unsupported dtype {name}")


def make_segment_ids_and_pos(batch: int, seqlen: int, segments: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    if seqlen % segments != 0:
        raise ValueError(f"{seqlen=} must be divisible by {segments=}")
    segment_len = seqlen // segments
    ids_1d = np.repeat(np.arange(1, segments + 1, dtype=np.int32), segment_len)
    pos_1d = np.tile(np.arange(segment_len, dtype=np.int32), segments)
    ids = np.broadcast_to(ids_1d, (batch, seqlen)).copy()
    pos = np.broadcast_to(pos_1d, (batch, seqlen)).copy()
    return jnp.asarray(ids), jnp.asarray(pos)


def make_valid_bshd_segment_ids_and_pos(batch: int, seqlen: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    ids = np.ones((batch, seqlen), dtype=np.int32)
    pos = np.broadcast_to(np.arange(seqlen, dtype=np.int32), (batch, seqlen)).copy()
    return jnp.asarray(ids), jnp.asarray(pos)


def index_shape(global_shape: tuple[int, ...], index: Any | None) -> tuple[int, ...]:
    if index is None:
        return global_shape
    local_shape = []
    for dim, item in zip(global_shape, index):
        if isinstance(item, slice):
            start, stop, step = item.indices(dim)
            if step != 1:
                raise ValueError(f"Unsupported non-contiguous shard index {index}")
            local_shape.append(stop - start)
        else:
            local_shape.append(1)
    return tuple(local_shape)


def make_constant_array(
    shape: tuple[int, int, int, int],
    value: float,
    dtype: jnp.dtype,
    sharding: Any,
) -> jax.Array:
    if isinstance(sharding, NamedSharding):
        host_dtype = np.dtype(dtype)

        def callback(index):
            return np.full(index_shape(shape, index), value, dtype=host_dtype)

        arr = jax.make_array_from_callback(shape, sharding, callback, dtype=dtype)
        arr.block_until_ready()
        return arr
    return jnp.full(shape, value, dtype=dtype)


def make_data_arrays(
    q_shape: tuple[int, int, int, int],
    k_shape: tuple[int, int, int, int],
    v_shape: tuple[int, int, int, int],
    dout_shape: tuple[int, int, int, int],
    dtype: jnp.dtype,
    sharding: Any,
    use_sharded_callback: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if use_sharded_callback and isinstance(sharding, NamedSharding):
        arrays = (
            make_constant_array(q_shape, 0.01171875, dtype, sharding),
            make_constant_array(k_shape, 0.017578125, dtype, sharding),
            make_constant_array(v_shape, 0.0234375, dtype, sharding),
            make_constant_array(dout_shape, 0.03125, dtype, sharding),
        )
        block_until_ready(arrays)
        return arrays

    def init():
        q = jnp.full(q_shape, 0.01171875, dtype=dtype)
        k = jnp.full(k_shape, 0.017578125, dtype=dtype)
        v = jnp.full(v_shape, 0.0234375, dtype=dtype)
        dout = jnp.full(dout_shape, 0.03125, dtype=dtype)
        return q, k, v, dout

    init_jit = jax.jit(init, out_shardings=(sharding, sharding, sharding, sharding))
    arrays = init_jit()
    block_until_ready(arrays)
    return arrays


def make_sequence_descriptor(
    *,
    layout: str,
    batch: int,
    seqlen: int,
    segments: int,
    cp_mode: str,
    cp_size: int,
    seq_sharding: Any | None,
    stripe_size: int | None,
) -> tuple[SequenceDescriptor, dict[str, Any]]:
    if layout == "thd":
        segment_ids, segment_pos = make_segment_ids_and_pos(batch, seqlen, segments)
        if cp_mode in ("ring", "all_gather"):
            effective_stripe_size = 1 if cp_mode == "ring" else stripe_size
            segment_ids = reorder_causal_load_balancing(
                segment_ids, ReorderStrategy.Striped, cp_size, 1, effective_stripe_size
            )
            segment_pos = reorder_causal_load_balancing(
                segment_pos, ReorderStrategy.Striped, cp_size, 1, effective_stripe_size
            )
        if seq_sharding is not None:
            segment_ids = jax.device_put(segment_ids, seq_sharding)
            segment_pos = jax.device_put(segment_pos, seq_sharding)
        desc = SequenceDescriptor.from_segment_ids_and_pos(
            (segment_ids, segment_ids), (segment_pos, segment_pos)
        )
        metadata = {
            "segments": segments,
            "segment_len": seqlen // segments,
            "padding_tokens": 0,
            "stripe_size": (1 if cp_mode == "ring" else stripe_size),
        }
        return desc, metadata

    segment_ids, segment_pos = make_valid_bshd_segment_ids_and_pos(batch, seqlen)
    if seq_sharding is not None:
        segment_ids = jax.device_put(segment_ids, seq_sharding)
        segment_pos = jax.device_put(segment_pos, seq_sharding)
    desc = SequenceDescriptor.from_segment_ids_and_pos(
        (segment_ids, segment_ids), (segment_pos, segment_pos)
    )
    return desc, {"segments": 1, "segment_len": seqlen, "padding_tokens": 0, "stripe_size": None}


def make_seq_desc_sharding(desc: SequenceDescriptor, mesh: Mesh, cp_axis: str) -> SequenceDescriptor:
    def shard_for(x):
        if x.ndim == 2:
            return NamedSharding(mesh, PartitionSpec(None, cp_axis))
        return NamedSharding(mesh, PartitionSpec(None))

    return jax.tree.map(shard_for, desc)


def jax_native_attention_loss(
    q_in,
    k_in,
    v_in,
    dout_in,
    *,
    layout: str,
    batch: int,
    seqlen: int,
    segments: int,
    dim: int,
    implementation: str,
):
    scale = 1.0 / math.sqrt(dim)
    if layout == "thd":
        segment_len = seqlen // segments
        q_view = jnp.reshape(q_in, (batch * segments, segment_len, q_in.shape[2], q_in.shape[3]))
        k_view = jnp.reshape(k_in, (batch * segments, segment_len, k_in.shape[2], k_in.shape[3]))
        v_view = jnp.reshape(v_in, (batch * segments, segment_len, v_in.shape[2], v_in.shape[3]))
        out = jax.nn.dot_product_attention(
            q_view,
            k_view,
            v_view,
            scale=scale,
            is_causal=True,
            implementation=implementation,
        )
        out = jnp.reshape(out, (batch, seqlen, q_in.shape[2], q_in.shape[3]))
    else:
        out = jax.nn.dot_product_attention(
            q_in,
            k_in,
            v_in,
            scale=scale,
            is_causal=True,
            implementation=implementation,
        )
    return jnp.sum(out.astype(jnp.float32) * dout_in.astype(jnp.float32))


def flax_linen_attention_loss(
    q_in,
    k_in,
    v_in,
    dout_in,
    *,
    layout: str,
    batch: int,
    seqlen: int,
    segments: int,
):
    if flax_linen is None:
        raise RuntimeError("flax is not installed")
    if layout == "thd":
        segment_len = seqlen // segments
        q_view = jnp.reshape(q_in, (batch * segments, segment_len, q_in.shape[2], q_in.shape[3]))
        k_view = jnp.reshape(k_in, (batch * segments, segment_len, k_in.shape[2], k_in.shape[3]))
        v_view = jnp.reshape(v_in, (batch * segments, segment_len, v_in.shape[2], v_in.shape[3]))
        dout_view = jnp.reshape(
            dout_in, (batch * segments, segment_len, dout_in.shape[2], dout_in.shape[3])
        )
        mask = jnp.tril(jnp.ones((1, 1, segment_len, segment_len), dtype=jnp.bool_))
        out = flax_linen.dot_product_attention(
            q_view,
            k_view,
            v_view,
            mask=mask,
            dropout_rate=0.0,
            deterministic=True,
            dtype=q_in.dtype,
        )
        return jnp.sum(out.astype(jnp.float32) * dout_view.astype(jnp.float32))

    mask = jnp.tril(jnp.ones((1, 1, seqlen, seqlen), dtype=jnp.bool_))
    out = flax_linen.dot_product_attention(
        q_in,
        k_in,
        v_in,
        mask=mask,
        dropout_rate=0.0,
        deterministic=True,
        dtype=q_in.dtype,
    )
    return jnp.sum(out.astype(jnp.float32) * dout_in.astype(jnp.float32))


def cuda_profiler_start_stop(enabled: bool):
    if not enabled:
        return None
    cudart = ctypes.CDLL("libcudart.so")
    return cudart


def parse_local_device_ids(value: str | None, default: int) -> int | list[int]:
    if value is None or value == "":
        return default
    values = [int(item) for item in value.split(",") if item]
    if not values:
        return default
    if len(values) == 1:
        return values[0]
    return values


def maybe_cuda_start(cudart) -> None:
    if cudart is not None:
        cudart.cudaProfilerStart()


def maybe_cuda_stop(cudart) -> None:
    if cudart is not None:
        cudart.cudaProfilerStop()


def main() -> None:
    args = parse_args()
    if args.backend in ("jax_native", "flax_linen") and args.cp != "none":
        raise ValueError(f"--backend={args.backend} only supports --cp=none")
    if args.distributed:
        if args.num_processes <= 1:
            raise ValueError("--distributed requires --num-processes > 1")
        jax.distributed.initialize(
            coordinator_address=args.coordinator_address,
            num_processes=args.num_processes,
            process_id=args.process_id,
            local_device_ids=parse_local_device_ids(args.local_device_ids, args.process_id),
        )
        if jax.local_device_count() != 1:
            raise RuntimeError(
                f"NVSHMEM requires one local device per process; got {jax.local_device_count()}"
            )

    dtype = get_dtype(args.dtype)
    qk_dim = args.qk_dim or args.dim
    v_dim = args.v_dim or args.dim
    visible_devices = len(jax.devices())
    cp_size = 1 if args.cp == "none" else (args.cp_size or visible_devices)
    if cp_size > visible_devices:
        raise RuntimeError(f"Requested {cp_size=} but only {visible_devices} JAX devices are visible")
    if args.cp != "none" and cp_size < 2:
        raise RuntimeError(f"{args.cp} CP requires at least 2 visible JAX devices")
    if args.cp != "none":
        if args.seqlen % (2 * cp_size) != 0:
            raise ValueError(f"Causal load balancing requires S divisible by {2 * cp_size}")
        if args.cp == "ring":
            os.environ.setdefault("NVTE_FUSED_RING_ATTENTION_USE_SCAN", "0")

    q_shape = (args.batch, args.seqlen, args.heads, qk_dim)
    k_shape = (args.batch, args.seqlen, args.heads, qk_dim)
    v_shape = (args.batch, args.seqlen, args.heads, v_dim)
    dout_shape = (args.batch, args.seqlen, args.heads, v_dim)
    data_shape = q_shape
    is_thd = args.layout == "thd"
    qkv_layout = QKVLayout.THD_THD_THD if is_thd else QKVLayout.BSHD_BSHD_BSHD
    attn_mask_type = AttnMaskType.PADDING_CAUSAL_MASK if is_thd else AttnMaskType.CAUSAL_MASK
    max_segments_per_seq = args.segments if is_thd else 1
    cp_strategy = {
        "none": CPStrategy.DEFAULT,
        "ring": CPStrategy.RING,
        "all_gather": CPStrategy.ALL_GATHER,
    }[args.cp]
    cp_axis = "cp" if args.cp != "none" else ""
    if args.cp == "ring" and is_thd:
        stripe_size = 1
    elif args.cp == "all_gather" and is_thd:
        stripe_size = args.stripe_size if args.stripe_size is not None else 128
    else:
        if args.stripe_size is not None and not is_thd:
            raise ValueError("--stripe-size is only supported for THD all_gather CP")
        stripe_size = None

    if args.check_backend and args.backend == "te_fused" and not is_fused_attn_kernel_available(
        True,
        dtype,
        dtype,
        qkv_layout,
        AttnBiasType.NO_BIAS,
        attn_mask_type,
        AttnSoftmaxType.VANILLA_SOFTMAX,
        0.0,
        args.heads,
        args.heads,
        args.seqlen,
        args.seqlen,
        qk_dim,
        v_dim,
        None,
    ):
        raise RuntimeError("Requested fused attention backend is not available")

    mesh = None
    mesh_resource = None
    if args.cp != "none":
        mesh = Mesh(np.asarray(jax.devices()[:cp_size]), ("cp",))
        mesh_resource = MeshResource(cp_resource="cp")
        data_sharding = NamedSharding(mesh, PartitionSpec(None, "cp", None, None))
        seq_array_sharding = NamedSharding(mesh, PartitionSpec(None, "cp"))
    else:
        data_sharding = SingleDeviceSharding(jax.devices()[0])
        seq_array_sharding = None

    context = jax.set_mesh(mesh) if mesh is not None else nullcontext()
    autocast_context = (
        autocast(mesh_resource=mesh_resource) if mesh_resource is not None else nullcontext()
    )

    with context, autocast_context:
        q, k, v, dout = make_data_arrays(
            q_shape, k_shape, v_shape, dout_shape, dtype, data_sharding, args.sharded_input_callback
        )
        if args.cp != "none" and not args.skip_qkv_reorder:
            reorder_strategy = ReorderStrategy.Striped if is_thd else ReorderStrategy.DualChunkSwap
            q = reorder_causal_load_balancing(q, reorder_strategy, cp_size, 1, stripe_size)
            k = reorder_causal_load_balancing(k, reorder_strategy, cp_size, 1, stripe_size)
            v = reorder_causal_load_balancing(v, reorder_strategy, cp_size, 1, stripe_size)
            dout = reorder_causal_load_balancing(dout, reorder_strategy, cp_size, 1, stripe_size)
            q, k, v, dout = (
                jax.device_put(q, data_sharding),
                jax.device_put(k, data_sharding),
                jax.device_put(v, data_sharding),
                jax.device_put(dout, data_sharding),
            )
            block_until_ready((q, k, v, dout))

        seq_desc, seq_metadata = make_sequence_descriptor(
            layout=args.layout,
            batch=args.batch,
            seqlen=args.seqlen,
            segments=args.segments,
            cp_mode=args.cp,
            cp_size=cp_size,
            seq_sharding=seq_array_sharding,
            stripe_size=stripe_size,
        )
        seq_desc_sharding = make_seq_desc_sharding(seq_desc, mesh, "cp") if mesh is not None else None

        def loss_fn(q_in, k_in, v_in, dout_in, seq_desc_in):
            if args.backend == "jax_native":
                return jax_native_attention_loss(
                    q_in,
                    k_in,
                    v_in,
                    dout_in,
                    layout=args.layout,
                    batch=args.batch,
                    seqlen=args.seqlen,
                    segments=args.segments,
                    dim=qk_dim,
                    implementation=args.jax_attn_implementation,
                )
            if args.backend == "flax_linen":
                return flax_linen_attention_loss(
                    q_in,
                    k_in,
                    v_in,
                    dout_in,
                    layout=args.layout,
                    batch=args.batch,
                    seqlen=args.seqlen,
                    segments=args.segments,
                )
            out = fused_attn(
                (q_in, k_in, v_in),
                None,
                seq_desc_in,
                None,
                AttnBiasType.NO_BIAS,
                attn_mask_type,
                qkv_layout,
                AttnSoftmaxType.VANILLA_SOFTMAX,
                1.0 / math.sqrt(qk_dim),
                0.0,
                True,
                max_segments_per_seq=max_segments_per_seq,
                window_size=None,
                context_parallel_strategy=cp_strategy,
                context_parallel_causal_load_balanced=args.cp != "none",
                context_parallel_axis=cp_axis,
                stripe_size=stripe_size,
            )
            return jnp.sum(out.astype(jnp.float32) * dout_in.astype(jnp.float32))

        if args.mode == "fwd_bwd":
            step = value_and_grad(loss_fn, argnums=(0, 1, 2))
            if args.cp == "ring":
                step = jax.jit(
                    step,
                    in_shardings=(data_sharding, data_sharding, data_sharding, data_sharding, seq_desc_sharding),
                    out_shardings=(None, (data_sharding, data_sharding, data_sharding)),
                )
            else:
                step = jax.jit(step)
        else:
            if args.cp == "ring":
                step = jax.jit(
                    loss_fn,
                    in_shardings=(data_sharding, data_sharding, data_sharding, data_sharding, seq_desc_sharding),
                    out_shardings=None,
                )
            else:
                step = jax.jit(loss_fn)

        compile_start = time.perf_counter()
        result = step(q, k, v, dout, seq_desc)
        block_until_ready(result)
        compile_and_first_run_s = time.perf_counter() - compile_start

        for _ in range(args.warmup):
            result = step(q, k, v, dout, seq_desc)
            block_until_ready(result)

        cudart = cuda_profiler_start_stop(args.cuda_profiler_api)
        maybe_cuda_start(cudart)
        times = []
        for _ in range(args.iters):
            start = time.perf_counter()
            result = step(q, k, v, dout, seq_desc)
            block_until_ready(result)
            times.append(time.perf_counter() - start)
        maybe_cuda_stop(cudart)


    mean_s = statistics.fmean(times)
    stdev_s = statistics.stdev(times) if len(times) > 1 else 0.0
    summary = {
        "layout": args.layout,
        "backend": args.backend,
        "jax_attn_implementation": (
            args.jax_attn_implementation if args.backend == "jax_native" else None
        ),
        "sharded_input_callback": args.sharded_input_callback,
        "skip_qkv_reorder": args.skip_qkv_reorder,
        "cp": args.cp,
        "cp_size": cp_size,
        "global_shape": data_shape,
        "q_shape": q_shape,
        "k_shape": k_shape,
        "v_shape": v_shape,
        "dout_shape": dout_shape,
        "head_dim_qk": qk_dim,
        "head_dim_v": v_dim,
        "dtype": args.dtype,
        "qkv_layout": str(qkv_layout),
        "attn_mask_type": str(attn_mask_type),
        "mode": args.mode,
        "warmup": args.warmup,
        "iters": args.iters,
        "compile_and_first_run_s": compile_and_first_run_s,
        "times_s": times,
        "mean_s": mean_s,
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "stdev_s": stdev_s,
        "p20_s": percentile(times, 0.20),
        "p80_s": percentile(times, 0.80),
        "te_version": getattr(transformer_engine, "__version__", "unknown"),
        "jax_version": jax.__version__,
        "jax_devices": [str(d) for d in jax.devices()],
        "distributed": args.distributed,
        "process_id": args.process_id,
        "num_processes": args.num_processes if args.distributed else 1,
        "local_device_count": jax.local_device_count(),
        "compute_capability_0": get_device_compute_capability(0),
        "cudnn_version": get_cudnn_version(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "nvte_fused_ring_attention_use_scan": os.environ.get(
            "NVTE_FUSED_RING_ATTENTION_USE_SCAN", ""
        ),
        **seq_metadata,
    }

    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_output is not None and args.process_id == 0:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
