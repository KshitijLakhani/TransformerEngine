#!/usr/bin/env python3
"""Validate TE JAX attention benchmark cases against JAX native attention."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import value_and_grad
from jax.sharding import Mesh, NamedSharding, PartitionSpec, SingleDeviceSharding

from benchmark_te_jax_attention import (
    block_until_ready,
    get_dtype,
    make_sequence_descriptor,
    make_seq_desc_sharding,
)
from transformer_engine.jax import autocast
from transformer_engine.jax.attention import (
    AttnBiasType,
    AttnMaskType,
    AttnSoftmaxType,
    CPStrategy,
    QKVLayout,
    ReorderStrategy,
    fused_attn,
    inverse_reorder_causal_load_balancing,
    reorder_causal_load_balancing,
)
from transformer_engine.jax.sharding import MeshResource


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=("bshd",), default="bshd")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seqlen", type=int, action="append", required=True)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128, help="Default head dim for both QK and V.")
    parser.add_argument("--qk-dim", type=int, help="Head dim for Q and K. Defaults to --dim.")
    parser.add_argument("--v-dim", type=int, help="Head dim for V and output. Defaults to --dim.")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--reference-implementation", choices=("cudnn", "xla", "te_fused"), default="cudnn")
    parser.add_argument("--dout-mode", choices=("pattern", "constant"), default="pattern")
    parser.add_argument("--cp-size", type=int, default=4)
    parser.add_argument(
        "--cp",
        choices=("none", "ring", "all_gather"),
        action="append",
        default=None,
    )
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def make_patterned_arrays(
    q_shape: tuple[int, int, int, int],
    k_shape: tuple[int, int, int, int],
    v_shape: tuple[int, int, int, int],
    dout_shape: tuple[int, int, int, int],
    dtype: jnp.dtype,
    sharding: Any,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Create deterministic non-constant tensors without host materialization."""

    def make_one(shape, coeff, mod, offset, scale):
        batch, seqlen, heads, dim = shape
        b = jnp.arange(batch, dtype=jnp.float32)[:, None, None, None]
        s = jnp.arange(seqlen, dtype=jnp.float32)[None, :, None, None]
        h = jnp.arange(heads, dtype=jnp.float32)[None, None, :, None]
        d = jnp.arange(dim, dtype=jnp.float32)[None, None, None, :]
        base = jnp.mod(3.0 * b + 5.0 * s + 7.0 * h + coeff * d, mod)
        return ((base - offset) / scale).astype(dtype)

    def init():
        q = make_one(q_shape, 11.0, 257.0, 128.0, 4096.0)
        k = make_one(k_shape, 13.0, 263.0, 131.0, 4096.0)
        v = make_one(v_shape, 19.0, 269.0, 134.0, 2048.0)
        dout = make_one(dout_shape, 29.0, 271.0, 135.0, 2048.0)
        return q, k, v, dout

    init_jit = jax.jit(init, out_shardings=(sharding, sharding, sharding, sharding))
    arrays = init_jit()
    block_until_ready(arrays)
    return arrays


def reference_forward(q, k, v, *, dim: int, implementation: str, seq_desc=None):
    if implementation == "te_fused":
        return fused_attn(
            (q, k, v),
            None,
            seq_desc,
            None,
            AttnBiasType.NO_BIAS,
            AttnMaskType.CAUSAL_MASK,
            QKVLayout.BSHD_BSHD_BSHD,
            AttnSoftmaxType.VANILLA_SOFTMAX,
            1.0 / math.sqrt(dim),
            0.0,
            True,
            max_segments_per_seq=1,
            window_size=None,
            context_parallel_strategy=CPStrategy.DEFAULT,
            context_parallel_causal_load_balanced=False,
            context_parallel_axis="",
            stripe_size=None,
        )
    return jax.nn.dot_product_attention(
        q,
        k,
        v,
        scale=1.0 / math.sqrt(dim),
        is_causal=True,
        implementation=implementation,
    )


def compare_arrays(actual, expected, *, rtol: float, atol: float) -> dict[str, Any]:
    if getattr(actual, "sharding", None) != getattr(expected, "sharding", None):
        expected = jax.device_put(expected, actual.sharding)
    actual_f32 = actual.astype(jnp.float32)
    expected_f32 = expected.astype(jnp.float32)
    diff = jnp.abs(actual_f32 - expected_f32)
    rel = diff / jnp.maximum(jnp.abs(expected_f32), jnp.asarray(1e-8, dtype=jnp.float32))
    close = jnp.isclose(actual_f32, expected_f32, rtol=rtol, atol=atol)
    result = {
        "max_abs": float(jnp.max(diff)),
        "mean_abs": float(jnp.mean(diff)),
        "max_rel": float(jnp.max(rel)),
        "fail_fraction": float(1.0 - jnp.mean(close.astype(jnp.float32))),
        "allclose": bool(jnp.all(close)),
    }
    return result


def validate_case(args: argparse.Namespace, seqlen: int, cp_mode: str) -> dict[str, Any]:
    dtype = get_dtype(args.dtype)
    qk_dim = args.qk_dim or args.dim
    v_dim = args.v_dim or args.dim
    q_shape = (args.batch, seqlen, args.heads, qk_dim)
    k_shape = (args.batch, seqlen, args.heads, qk_dim)
    v_shape = (args.batch, seqlen, args.heads, v_dim)
    dout_shape = (args.batch, seqlen, args.heads, v_dim)
    ref_sharding = SingleDeviceSharding(jax.devices()[0])
    q_ref, k_ref, v_ref, dout_ref = make_patterned_arrays(
        q_shape, k_shape, v_shape, dout_shape, dtype, ref_sharding
    )
    if args.dout_mode == "constant":
        make_dout = jax.jit(
            lambda: jnp.full(dout_shape, 0.03125, dtype=dtype), out_shardings=ref_sharding
        )
        dout_ref = make_dout()
        dout_ref.block_until_ready()

    ref_seq_desc, _ = make_sequence_descriptor(
        layout="bshd",
        batch=args.batch,
        seqlen=seqlen,
        segments=1,
        cp_mode="none",
        cp_size=1,
        seq_sharding=None,
        stripe_size=None,
    )

    ref_forward_jit = jax.jit(
        lambda q, k, v: reference_forward(
            q, k, v, dim=qk_dim, implementation=args.reference_implementation, seq_desc=ref_seq_desc
        )
    )

    def ref_loss(q, k, v):
        out = reference_forward(
            q, k, v, dim=qk_dim, implementation=args.reference_implementation, seq_desc=ref_seq_desc
        )
        return jnp.sum(out.astype(jnp.float32) * dout_ref.astype(jnp.float32))

    ref_grad_jit = jax.jit(value_and_grad(ref_loss, argnums=(0, 1, 2)))
    ref_out = ref_forward_jit(q_ref, k_ref, v_ref)
    ref_loss_value, ref_grads = ref_grad_jit(q_ref, k_ref, v_ref)
    block_until_ready((ref_out, ref_loss_value, ref_grads))

    cp_size = 1 if cp_mode == "none" else args.cp_size
    if cp_mode != "none" and seqlen % (2 * cp_size) != 0:
        raise ValueError(f"{seqlen=} must be divisible by {2 * cp_size=} for CP load balancing")

    mesh = None
    mesh_resource = None
    if cp_mode == "none":
        data_sharding = ref_sharding
        seq_array_sharding = None
        cp_axis = ""
        cp_strategy = CPStrategy.DEFAULT
        q_te, k_te, v_te, dout_te = q_ref, k_ref, v_ref, dout_ref
    else:
        mesh = Mesh(np.asarray(jax.devices()[:cp_size]), ("cp",))
        mesh_resource = MeshResource(cp_resource="cp")
        data_sharding = NamedSharding(mesh, PartitionSpec(None, "cp", None, None))
        seq_array_sharding = NamedSharding(mesh, PartitionSpec(None, "cp"))
        cp_axis = "cp"
        cp_strategy = {
            "ring": CPStrategy.RING,
            "all_gather": CPStrategy.ALL_GATHER,
        }[cp_mode]
        q_te = reorder_causal_load_balancing(
            q_ref, ReorderStrategy.DualChunkSwap, cp_size, 1, None
        )
        k_te = reorder_causal_load_balancing(
            k_ref, ReorderStrategy.DualChunkSwap, cp_size, 1, None
        )
        v_te = reorder_causal_load_balancing(
            v_ref, ReorderStrategy.DualChunkSwap, cp_size, 1, None
        )
        q_te, k_te, v_te, dout_te = (
            jax.device_put(q_te, data_sharding),
            jax.device_put(k_te, data_sharding),
            jax.device_put(v_te, data_sharding),
            jax.device_put(dout_ref, data_sharding),
        )
        block_until_ready((q_te, k_te, v_te, dout_te))

    seq_desc, _ = make_sequence_descriptor(
        layout="bshd",
        batch=args.batch,
        seqlen=seqlen,
        segments=1,
        cp_mode=cp_mode,
        cp_size=cp_size,
        seq_sharding=seq_array_sharding,
        stripe_size=None,
    )
    seq_desc_sharding = make_seq_desc_sharding(seq_desc, mesh, "cp") if mesh is not None else None

    def maybe_inverse(x):
        if cp_mode == "none":
            return x
        return inverse_reorder_causal_load_balancing(
            x, ReorderStrategy.DualChunkSwap, cp_size, 1, None
        )

    def te_forward(q, k, v, seq_desc_in):
        return fused_attn(
            (q, k, v),
            None,
            seq_desc_in,
            None,
            AttnBiasType.NO_BIAS,
            AttnMaskType.CAUSAL_MASK,
            QKVLayout.BSHD_BSHD_BSHD,
            AttnSoftmaxType.VANILLA_SOFTMAX,
            1.0 / math.sqrt(qk_dim),
            0.0,
            True,
            max_segments_per_seq=1,
            window_size=None,
            context_parallel_strategy=cp_strategy,
            context_parallel_causal_load_balanced=cp_mode != "none",
            context_parallel_axis=cp_axis,
            stripe_size=None,
        )

    def te_loss(q, k, v, dout, seq_desc_in):
        out = maybe_inverse(te_forward(q, k, v, seq_desc_in))
        return jnp.sum(out.astype(jnp.float32) * dout.astype(jnp.float32))

    context = jax.set_mesh(mesh) if mesh is not None else nullcontext()
    autocast_context = (
        autocast(mesh_resource=mesh_resource) if mesh_resource is not None else nullcontext()
    )
    in_shardings = (data_sharding, data_sharding, data_sharding, seq_desc_sharding)
    loss_in_shardings = (data_sharding, data_sharding, data_sharding, data_sharding, seq_desc_sharding)
    out_shardings = data_sharding
    grad_shardings = (data_sharding, data_sharding, data_sharding)
    with context, autocast_context:
        te_forward_jit = jax.jit(te_forward, in_shardings=in_shardings, out_shardings=out_shardings)
        te_grad_jit = jax.jit(
            value_and_grad(te_loss, argnums=(0, 1, 2)),
            in_shardings=loss_in_shardings,
            out_shardings=(None, grad_shardings),
        )
        te_out = maybe_inverse(te_forward_jit(q_te, k_te, v_te, seq_desc))
        te_loss_value, te_grads = te_grad_jit(q_te, k_te, v_te, dout_te, seq_desc)
        te_grads = tuple(maybe_inverse(g) for g in te_grads)
        block_until_ready((te_out, te_loss_value, te_grads))

    metrics = {
        "layout": "bshd",
        "seqlen": seqlen,
        "cp": cp_mode,
        "cp_size": cp_size,
        "reference": args.reference_implementation,
        "dtype": args.dtype,
        "dout_mode": args.dout_mode,
        "q_shape": q_shape,
        "k_shape": k_shape,
        "v_shape": v_shape,
        "dout_shape": dout_shape,
        "head_dim_qk": qk_dim,
        "head_dim_v": v_dim,
        "rtol": args.rtol,
        "atol": args.atol,
        "loss_ref": float(ref_loss_value),
        "loss_te": float(te_loss_value),
    }
    metrics["loss_abs"] = abs(metrics["loss_te"] - metrics["loss_ref"])
    metrics["loss_rel"] = metrics["loss_abs"] / max(abs(metrics["loss_ref"]), 1e-8)
    metrics["loss_allclose"] = bool(
        math.isclose(metrics["loss_te"], metrics["loss_ref"], rel_tol=args.rtol, abs_tol=args.atol)
    )
    metrics["out"] = compare_arrays(te_out, ref_out, rtol=args.rtol, atol=args.atol)
    for name, actual, expected in zip(("dq", "dk", "dv"), te_grads, ref_grads):
        metrics[name] = compare_arrays(actual, expected, rtol=args.rtol, atol=args.atol)
    metrics["passed"] = (
        metrics["loss_allclose"]
        and metrics["out"]["allclose"]
        and metrics["dq"]["allclose"]
        and metrics["dk"]["allclose"]
        and metrics["dv"]["allclose"]
    )
    return metrics


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    flat = {
        "layout": row["layout"],
        "seqlen": row["seqlen"],
        "head_dim_qk": row["head_dim_qk"],
        "head_dim_v": row["head_dim_v"],
        "cp": row["cp"],
        "cp_size": row["cp_size"],
        "reference": row["reference"],
        "dtype": row["dtype"],
        "dout_mode": row["dout_mode"],
        "passed": row["passed"],
        "loss_abs": row["loss_abs"],
        "loss_rel": row["loss_rel"],
    }
    for field in ("out", "dq", "dk", "dv"):
        for metric, value in row[field].items():
            flat[f"{field}_{metric}"] = value
    return flat


def main() -> None:
    args = parse_args()
    if args.cp is None:
        args.cp = ["none", "ring", "all_gather"]
    rows = []
    for seqlen in args.seqlen:
        for cp_mode in args.cp:
            row = validate_case(args, seqlen, cp_mode)
            rows.append(row)
            print(json.dumps(row, indent=2, sort_keys=True))

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    if args.csv_output is not None:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        flat_rows = [flatten_row(row) for row in rows]
        with args.csv_output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)

    if not all(row["passed"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
