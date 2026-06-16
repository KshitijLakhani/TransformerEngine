# TE JAX Attention Benchmark Summary

| file | layout | cp | S | segments | seg len | distributed | mean ms | median ms | compile+first s |
|---|---|---|---:|---:|---:|---|---:|---:|---:|
| bshd_no_cp.json | bshd | none | 32768 | 1 | 32768 | False | 142.319 | 142.312 | 1.673 |
| bshd_ring_cp4.json | bshd | ring | 32768 | 1 | 32768 | False | 60.840 | 60.842 | 9.420 |
| bshd_ring_cp4_mp_no_nvshmem.json | bshd | ring | 32768 | 1 | 32768 | True | 60.504 | 60.499 | 6.186 |
| bshd_ring_cp4_mp_nvshmem.json | bshd | ring | 32768 | 1 | 32768 | True | 60.514 | 60.515 | 6.258 |
| bshd_ring_cp4_mp_nvshmem_cmd_default.json | bshd | ring | 32768 | 1 | 32768 | True | 60.498 | 60.487 | 6.179 |
| bshd_ring_cp4_mp_nvshmem_symmetric.json | bshd | ring | 32768 | 1 | 32768 | True | 60.497 | 60.489 | 10.478 |
| thd16_no_cp.json | thd | none | 32768 | 16 | 2048 | False | 20.071 | 20.066 | 2.322 |
| thd16_ring_cp4.json | thd | ring | 32768 | 16 | 2048 | False | 35.149 | 35.183 | 8.050 |
| thd8_no_cp.json | thd | none | 32768 | 8 | 4096 | False | 31.719 | 31.715 | 2.991 |
| thd8_ring_cp4.json | thd | ring | 32768 | 8 | 4096 | False | 36.892 | 36.883 | 8.941 |
| thd4_no_cp.json | thd | none | 32768 | 4 | 8192 | False | 55.002 | 54.998 | 2.230 |
| thd4_ring_cp4.json | thd | ring | 32768 | 4 | 8192 | False | 41.888 | 41.901 | 6.635 |
| thd2_no_cp.json | thd | none | 32768 | 2 | 16384 | False | 101.694 | 101.685 | 2.267 |
| thd2_ring_cp4.json | thd | ring | 32768 | 2 | 16384 | False | 53.665 | 53.679 | 6.664 |
| thd1_no_cp.json | thd | none | 32768 | 1 | 32768 | False | 195.143 | 195.137 | 2.347 |
| thd1_ring_cp4.json | thd | ring | 32768 | 1 | 32768 | False | 76.745 | 76.733 | 6.593 |
| thd16_s65536_no_cp.json | thd | none | 65536 | 16 | 4096 | False | 62.850 | 62.851 | 2.361 |
| thd16_s65536_ring_cp4.json | thd | ring | 65536 | 16 | 4096 | False | 71.360 | 71.354 | 8.230 |
| thd8_s65536_no_cp.json | thd | none | 65536 | 8 | 8192 | False | 109.484 | 109.452 | 2.988 |
| thd8_s65536_ring_cp4.json | thd | ring | 65536 | 8 | 8192 | False | 82.351 | 82.343 | 8.899 |
| thd4_s65536_no_cp.json | thd | none | 65536 | 4 | 16384 | False | 202.629 | 202.648 | 2.310 |
| thd4_s65536_ring_cp4.json | thd | ring | 65536 | 4 | 16384 | False | 105.463 | 105.476 | 6.865 |
| thd2_s65536_no_cp.json | thd | none | 65536 | 2 | 32768 | False | 389.163 | 389.166 | 2.506 |
| thd2_s65536_ring_cp4.json | thd | ring | 65536 | 2 | 32768 | False | 152.055 | 152.076 | 6.713 |
| thd16_ring_cp4_mp_no_nvshmem.json | thd | ring | 32768 | 16 | 2048 | True | 34.870 | 34.898 | 7.540 |
| thd16_ring_cp4_mp_nvshmem.json | thd | ring | 32768 | 16 | 2048 | True | 34.888 | 34.912 | 7.590 |
| thd16_ring_cp4_mp_nvshmem_symmetric.json | thd | ring | 32768 | 16 | 2048 | True | 34.939 | 34.936 | 7.523 |
