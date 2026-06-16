# Issue: XLA NVSHMEM Not Selected For TE/JAX Ring Attention

## Goal
Verify the NVIDIA blog NVSHMEM path for JAX/XLA `CollectivePermute` improves Transformer Engine JAX Ring attention context parallelism.

Blog: https://developer.nvidia.com/blog/accelerating-long-context-model-training-in-jax-and-xla/

## Environment
- Node: 4x NVIDIA GB200
- TE: 2.17.0.dev0+9b06f268
- JAX: 0.10.2.dev20260610+39b296a07
- NVSHMEM: v3.6.5, library path `/usr/lib/aarch64-linux-gnu/nvshmem/13`
- cuDNN reported by TE JAX: 92300

## Expected
With `--xla_gpu_experimental_enable_nvshmem=true`, Ring attention `collective-permute-start` operations should use the NVSHMEM backend or otherwise show NVSHMEM P2P kernels/host API activity in Nsight Systems.

## Actual
- Single-process JAX over 4 local GPUs aborts when NVSHMEM initializes because local XLA requires `device_count_per_process == 1`.
- One process per GPU fixes the abort, but performance is unchanged.
- HLO still reports `backend:"DEFAULT"` for Ring `collective-permute-start`.
- `--xla_gpu_collective_permute_mode=symmetric` changes HLO memory space/collectives mode to symmetric memory, but backend remains `DEFAULT`.
- `nsys-jax` still shows NCCL `ncclDevKernel_SendRecv`; no NVSHMEM communication kernel replaces it.

## Triage
The initial abort is understood and should not be treated as the handoff issue: single-process / multi-GPU JAX violates this XLA build's one-local-device-per-process NVSHMEM requirement. The reproducer uses one JAX process per GPU to avoid that topology issue.

The remaining issue is that, even with the topology corrected, TE Ring attention's `collective-permute-start` ops are still assigned/executed through the default/NCCL path rather than the NVSHMEM path described in the blog. We do not currently have evidence that this is caused by the TE build or TE attention implementation.

## Evidence
See `README.md` in this directory for wall-clock and HLO snippets.
Key profile artifacts:
- `../te_jax_attention/nsys/bshd_ring_cp4_mp_nvshmem_rank0.zip`
- `../te_jax_attention/nsys/bshd_ring_cp4_mp_nvshmem_symmetric_rank0.zip`
- `../te_jax_attention/nsys/stats/bshd_ring_cp4_mp_nvshmem_rank0_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`
- `../te_jax_attention/nsys/stats/bshd_ring_cp4_mp_nvshmem_symmetric_rank0_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`

## Reproducer
Run `./reproduce.sh` from the TE repo root, or copy the commands from `README.md`.

## Questions
1. Is this XLA/JAX build expected to assign `CollectiveBackendConfig::NVSHMEM` for these TE Ring attention collective-permute ops?
2. If yes, what flag/build/runtime condition is missing?
3. If no, which public XLA commit or JAX Toolbox build includes the backend assignment behavior described in the blog?
