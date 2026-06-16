# Artifact Manifest

Main benchmark harness:
- ../../scripts/benchmark_te_jax_attention.py
- ../../scripts/run_te_jax_attention_mp.sh
- ../../scripts/te_cluster_setup.sh

Optional validation helper:
- ../../scripts/validate_te_jax_attention.py

Summary and triage docs:
- README.md
- ISSUE.md
- environment_and_source.txt
- ../te_jax_attention/summary.md
- ../te_jax_attention/summary.csv

Profiles:
- ../te_jax_attention/nsys/bshd_ring_cp4_mp_nvshmem_rank0.zip
- ../te_jax_attention/nsys/bshd_ring_cp4_mp_nvshmem_symmetric_rank0.zip

Profile CSV summaries:
- ../te_jax_attention/nsys/stats/bshd_ring_cp4_mp_nvshmem_rank0_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv
- ../te_jax_attention/nsys/stats/bshd_ring_cp4_mp_nvshmem_symmetric_rank0_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv

Representative JSON results:
- ../te_jax_attention/bshd_ring_cp4_mp_no_nvshmem.json
- ../te_jax_attention/bshd_ring_cp4_mp_nvshmem.json
- ../te_jax_attention/bshd_ring_cp4_mp_nvshmem_symmetric.json
- ../te_jax_attention/thd16_ring_cp4_mp_no_nvshmem.json
- ../te_jax_attention/thd16_ring_cp4_mp_nvshmem.json
- ../te_jax_attention/thd16_ring_cp4_mp_nvshmem_symmetric.json

HLO/debug JSON summaries:
- ../te_jax_attention/debug_bshd_ring_mp_nvshmem_hlo.json
- ../te_jax_attention/debug_bshd_ring_mp_nvshmem_sym_hlo.json

Issue template:
- ISSUE.md
