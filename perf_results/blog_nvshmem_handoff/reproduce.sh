#!/usr/bin/env bash
set -euo pipefail

# Reproducer for TE JAX Ring attention under XLA NVSHMEM.
# Must be run from the TE repo root.
# This intentionally uses one JAX process per GPU. Single-process / multi-GPU
# NVSHMEM is a known invalid topology for this local XLA build and aborts before
# the backend-selection behavior under investigation can be measured.
# If TE JAX is not already built in this environment, run first:
#   scripts/te_cluster_setup.sh --framework jax --arch 120

export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/nvshmem/13:${LD_LIBRARY_PATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export NVTE_FUSED_ATTN=1
export NVTE_FUSED_RING_ATTENTION_USE_SCAN=0

mkdir -p perf_results/te_jax_attention

# Baseline: one-process-per-GPU, NCCL/default backend.
XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true" \
NUM_PROCESSES=4 COORDINATOR_ADDRESS=127.0.0.1:23601 \
  scripts/run_te_jax_attention_mp.sh \
  --layout bshd --cp ring --batch 2 --seqlen 32768 --heads 128 --dim 128 \
  --mode fwd_bwd --warmup 3 --iters 10 \
  --json-output perf_results/te_jax_attention/repro_bshd_ring_cp4_mp_no_nvshmem.json

# NVSHMEM flag: should not abort in one-process-per-GPU topology.
XLA_FLAGS="--xla_gpu_experimental_enable_nvshmem=true --xla_gpu_enable_latency_hiding_scheduler=true" \
NUM_PROCESSES=4 COORDINATOR_ADDRESS=127.0.0.1:23602 \
  scripts/run_te_jax_attention_mp.sh \
  --layout bshd --cp ring --batch 2 --seqlen 32768 --heads 128 --dim 128 \
  --mode fwd_bwd --warmup 3 --iters 10 \
  --json-output perf_results/te_jax_attention/repro_bshd_ring_cp4_mp_nvshmem.json

# NVSHMEM + symmetric memory mode.
XLA_FLAGS="--xla_gpu_experimental_enable_nvshmem=true --xla_gpu_collective_permute_mode=symmetric --xla_gpu_enable_latency_hiding_scheduler=true" \
NUM_PROCESSES=4 COORDINATOR_ADDRESS=127.0.0.1:23603 \
  scripts/run_te_jax_attention_mp.sh \
  --layout bshd --cp ring --batch 2 --seqlen 32768 --heads 128 --dim 128 \
  --mode fwd_bwd --warmup 3 --iters 10 \
  --json-output perf_results/te_jax_attention/repro_bshd_ring_cp4_mp_nvshmem_symmetric.json
