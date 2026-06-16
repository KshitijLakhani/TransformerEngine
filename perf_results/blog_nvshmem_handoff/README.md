# XLA NVSHMEM / TE JAX Ring Attention Handoff

## Goal
Validate whether the XLA NVSHMEM path from NVIDIA's blog improves Transformer Engine JAX Ring attention context parallelism on this local GB200 node.

Blog: https://developer.nvidia.com/blog/accelerating-long-context-model-training-in-jax-and-xla/

## Local Environment
- Node: 4x NVIDIA GB200
- TE: `2.17.0.dev0+9b06f268`
- JAX: `0.10.2.dev20260610+39b296a07`
- cuDNN reported by TE JAX: `92300`
- NVSHMEM library path needed at runtime: `/usr/lib/aarch64-linux-gnu/nvshmem/13`
- NVSHMEM reports: `NVSHMEM v3.6.5`

## Repro Prerequisites
This handoff is a TE-repo repro, not a standalone pure-JAX script. The authors need:
- a Transformer Engine checkout at this branch or equivalent TE source
- TE JAX built/installed for the target JAX/XLA environment
- a JAX/XLA build with the NVSHMEM flags available
- NVSHMEM runtime libraries available at `/usr/lib/aarch64-linux-gnu/nvshmem/13` or equivalent
- a one-JAX-process-per-GPU launch for NVSHMEM, because this local XLA build aborts if NVSHMEM initializes with more than one local device per process

A smaller pure-JAX `collective-permute` reproducer may be possible for XLA backend debugging, but the evidence here uses TE Ring attention to match the blog-relevant workload.

## Build TE For The Repro
From the TE repo root, inside the target JAX/XLA container or environment:

```bash
scripts/te_cluster_setup.sh --framework jax --arch 120
```

What this setup script does:
- installs the JAX-image dependencies needed by TE
- syncs/updates TE git submodules
- builds an editable TE install with `python3 -m pip install --no-build-isolation -e .`

Useful variants:
```bash
# If the environment already has dependencies installed:
scripts/te_cluster_setup.sh --framework jax --arch 120 --skip-deps

# If submodules are already synced:
scripts/te_cluster_setup.sh --framework jax --arch 120 --skip-submodules

# If using a non-default Python:
scripts/te_cluster_setup.sh --framework jax --arch 120 --python /path/to/python3
```

For GB200, `--arch 120` is the intended CUDA arch setting. If the authors use a different GPU, they should pass the matching architecture or omit `--arch` and let the script infer it from `nvidia-smi`.

Quick sanity check after build:
```bash
python3 - <<'PY'
import jax
import transformer_engine as te
from transformer_engine.jax import get_cudnn_version
print("jax", jax.__version__)
print("te", te.__version__)
print("devices", jax.devices())
print("cudnn", get_cudnn_version())
PY
```

## Main Findings
1. Single-process JAX over 4 GPUs aborts with `--xla_gpu_experimental_enable_nvshmem=true` for TE Ring attention.
2. Root cause is local XLA's hard requirement that NVSHMEM has one local device per process.
   - Source: `/opt/xla/xla/stream_executor/cuda/nvshmem.cc`
   - Failing check: `if (env->device_count_per_process != 1) LOG(FATAL) << "NVSHMEM API is only supported with one device per process";`
3. Launching one JAX process per GPU fixes the SIGABRT.
4. Even after fixing topology, TE Ring attention does not speed up here.
5. `nsys-jax` shows communication is still NCCL `ncclDevKernel_SendRecv`, not NVSHMEM P2P kernels.
6. HLO shows Ring `collective-permute-start` backend remains `backend:"DEFAULT"` with plain NVSHMEM flag.
7. Adding `--xla_gpu_collective_permute_mode=symmetric` changes HLO memory space to `COLLECTIVES_SYMMETRIC_MEMORY` and initializes NVSHMEM, but backend still remains `DEFAULT`, and Nsight still shows NCCL SendRecv.

## Ownership Triage
| Observation | Classification | Current read |
|---|---|---|
| Single-process / 4-GPU SIGABRT | Test launch topology vs XLA runtime requirement | Not a TE bug. This launch violates local XLA's one-device-per-process NVSHMEM requirement. Fixed by one JAX process per GPU. |
| No speedup after one-process-per-GPU launch | Lower-level JAX/XLA backend selection/runtime behavior | This is the handoff issue. HLO remains `backend:"DEFAULT"` and profiles still show NCCL. |
| TE Ring attention implementation/build | No evidence of local TE-side issue from these tests | TE Ring lowers to `collective-permute-start`, which is the op family the blog discusses. The missing piece appears to be XLA selecting/executing NVSHMEM for those ops. |
| Container/runtime libraries | NVSHMEM is present but not sufficient | `LD_LIBRARY_PATH` and NVSHMEM initialization are not enough to make this XLA build select the NVSHMEM collective backend. |

## Why This Differs From The Blog Expectation
The blog describes automatic backend selection for `CollectivePermute`, selecting NVSHMEM and generating NVSHMEM host API calls for ring-style P2P. In this local XLA source/build, `CollectiveBackendAssigner` only assigns `collectives_mode`; it does not set `CollectiveBackendConfig::NVSHMEM` for these TE Ring collective-permute ops.

Relevant local source:
- `/opt/xla/xla/backends/gpu/transforms/collectives/collective_backend_assigner.cc`
- `/opt/xla/xla/service/gpu/gpu_memory_space_assignment.cc`
- `/opt/xla/xla/backends/gpu/collectives/nvshmem_collectives.h`

Notable local source comment:
`NvshmemCollectives currently does not implement GpuCollectives, so it cannot be used as drop-in replacement of GPU collectives.`

## Wall-Clock Results
All results are fwd+bwd, B=2, S=32768, H=128, D=128, CP=4 unless noted.

| Case | Mean ms | Notes |
|---|---:|---|
| BSHD Ring CP4 single-process | 60.840 | NCCL baseline |
| BSHD Ring CP4 multiprocess, no NVSHMEM | 60.504 | one process per GPU |
| BSHD Ring CP4 multiprocess, NVSHMEM flag | 60.514 | no speedup |
| BSHD Ring CP4 multiprocess, NVSHMEM + symmetric permute | 60.497 | no speedup; compile slower |
| THD16 Ring CP4 multiprocess, no NVSHMEM | 34.870 | 16 segments, 2048 tokens/segment |
| THD16 Ring CP4 multiprocess, NVSHMEM flag | 34.888 | no speedup |
| THD16 Ring CP4 multiprocess, NVSHMEM + symmetric modes | 34.939 | no speedup |

Full table: `../te_jax_attention/summary.md`

## Profile Evidence
Rank-0 `nsys-jax` profile with NVSHMEM flag:
- Archive: `../te_jax_attention/nsys/bshd_ring_cp4_mp_nvshmem_rank0.zip`
- Kernel stats: `../te_jax_attention/nsys/stats/bshd_ring_cp4_mp_nvshmem_rank0_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`
- Top communication kernel: `ncclDevKernel_SendRecv`, ~30.5% of GPU kernel time.

Rank-0 `nsys-jax` profile with NVSHMEM + symmetric permute mode:
- Archive: `../te_jax_attention/nsys/bshd_ring_cp4_mp_nvshmem_symmetric_rank0.zip`
- Kernel stats: `../te_jax_attention/nsys/stats/bshd_ring_cp4_mp_nvshmem_symmetric_rank0_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`
- Top communication kernel still: `ncclDevKernel_SendRecv`, ~30.5% of GPU kernel time.

## HLO Evidence
Plain NVSHMEM flag HLO for Ring `collective-permute-start`:
```text
backend_config={"collective_backend_config":{"backend":"DEFAULT","collectives_mode":"COLLECTIVES_MODE_INVALID"}}
```

NVSHMEM + symmetric permute mode HLO:
```text
bf16[...] { ... :S(1) } collective-permute-start(...),
backend_config={"collective_backend_config":{"backend":"DEFAULT","collectives_mode":"COLLECTIVES_SYMMETRIC_MEMORY"}}
```

So symmetric memory is applied, but backend is still `DEFAULT`.

## Reproducer Summary
Use one process per GPU. Single-process 4-GPU JAX is expected to abort in this environment when NVSHMEM actually initializes.

```bash
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvshmem/13:${LD_LIBRARY_PATH:-}
export XLA_FLAGS="--xla_gpu_experimental_enable_nvshmem=true --xla_gpu_enable_latency_hiding_scheduler=true"
NUM_PROCESSES=4 COORDINATOR_ADDRESS=127.0.0.1:23460 \
  scripts/run_te_jax_attention_mp.sh \
  --layout bshd --cp ring --batch 2 --seqlen 32768 --heads 128 --dim 128 \
  --mode fwd_bwd --warmup 3 --iters 10 \
  --json-output perf_results/te_jax_attention/bshd_ring_cp4_mp_nvshmem.json
```

To force symmetric collective-permute memory:
```bash
export XLA_FLAGS="--xla_gpu_experimental_enable_nvshmem=true --xla_gpu_collective_permute_mode=symmetric --xla_gpu_enable_latency_hiding_scheduler=true"
```

To profile rank 0:
```bash
JAX_PROCESS_ID=0 nsys-jax --nsys-jax-condition='$JAX_PROCESS_ID == 0' -f \
  -o perf_results/te_jax_attention/nsys/bshd_ring_cp4_mp_nvshmem_rank0 \
  --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop -- \
  python3 -u scripts/benchmark_te_jax_attention.py \
  --distributed --coordinator-address 127.0.0.1:23464 --num-processes 4 --process-id 0 \
  --layout bshd --cp ring --batch 2 --seqlen 32768 --heads 128 --dim 128 \
  --mode fwd_bwd --warmup 2 --iters 3 --cuda-profiler-api
```

## Ask For Blog/XLA Authors
Can you confirm whether this XLA build is expected to set `backend:"NVSHMEM"` for TE/JAX Ring attention's `collective-permute-start` ops? If yes, what flag/build/runtime condition is missing here? If not, which XLA commit or JAX Toolbox build contains the backend assignment behavior described in the blog?
