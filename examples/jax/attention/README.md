# Fused Attention Examples #

These examples demonstrate how to use Transformer Engine's [cuDNN fused attention](https://github.com/NVIDIA/cudnn-frontend/blob/main/docs/operations/Attention.md) in JAX. Fused attention combines the batched matrix multiplications, masking, softmax, and optional dropout of the attention mechanism into a single GPU kernel, delivering higher performance and lower memory usage compared to an unfused implementation.

Each example shows **two API levels**:

1. **Low-level** — `transformer_engine.jax.attention.fused_attn`: full control over every parameter (QKV layout, mask type, bias type, scaling factor, sequence descriptor, etc.).
2. **High-level** — `transformer_engine.jax.flax.DotProductAttention`: a Flax `nn.Module` that wraps the fused kernel and automatically falls back to an unfused backend when the kernel is unavailable.

Before running the examples, please study the [MNIST example](/examples/jax/mnist) and the [encoder example](/examples/jax/encoder) to understand the basics of Transformer Engine + JAX.

---

## Example 1 — Self-Attention with GQA and Bias ##

**File:** `test_self_attn.py`

This is the simplest entry point. It covers:

* **BSHD layout** (`QKVLayout.BSHD_BSHD_BSHD`) — query, key, and value are separate tensors with shape `[batch, seqlen, heads, dim]`.
* **Causal masking** (`AttnMaskType.CAUSAL_MASK`) — upper-triangular mask preventing tokens from attending to future positions.
* **Post-scale additive bias** (`AttnBiasType.POST_SCALE_BIAS`) — bias added after the Q*K scaling: `softmax(scale * QK + bias)`.
* **Grouped-Query Attention (GQA)** — `num_heads_q > num_heads_kv`, so multiple query heads share fewer key/value heads.
* **Forward and backward** pass via `jax.value_and_grad`.

### Key API concepts ###

1. Build a `SequenceDescriptor` from sequence lengths:

```python
from transformer_engine.jax.attention import SequenceDescriptor

q_seqlens = jnp.full((batch,), max_seqlen, dtype=jnp.int32)
kv_seqlens = jnp.full((batch,), max_seqlen, dtype=jnp.int32)
seq_desc = SequenceDescriptor.from_seqlens((q_seqlens, kv_seqlens))
```

2. Call the low-level API:

```python
from transformer_engine.jax.attention import fused_attn, AttnBiasType, AttnMaskType, QKVLayout

out = fused_attn(
    qkv=(q, k, v),
    bias=bias,
    sequence_descriptor=seq_desc,
    seed=None,
    attn_bias_type=AttnBiasType.POST_SCALE_BIAS,
    attn_mask_type=AttnMaskType.CAUSAL_MASK,
    qkv_layout=QKVLayout.BSHD_BSHD_BSHD,
    softmax_type=AttnSoftmaxType.VANILLA_SOFTMAX,
    scaling_factor=1.0 / sqrt(head_dim),
    dropout_probability=0.0,
    is_training=True,
)
```

3. Or use the Flax module:

```python
import transformer_engine.jax.flax as te_flax

dpa = te_flax.DotProductAttention(
    head_dim=64,
    num_attention_heads=12,
    num_gqa_groups=4,        # GQA with 4 KV groups
    attn_mask_type="causal",
    attn_bias_type="post_scale_bias",
    transpose_batch_sequence=False,
)
variables = dpa.init(rng, q, k, v, bias=bias, deterministic=True)
out = dpa.apply(variables, q, k, v, bias=bias, deterministic=True)
```

### Run ###

```bash
python test_self_attn.py
pytest -xvs test_self_attn.py
```

---

## Example 2 — Cross-Attention with THD Packing and Sliding Window ##

**File:** `test_cross_attn_thd.py`

This example demonstrates the more advanced THD (packed sequence) format and sliding window attention.

* **THD layout** (`QKVLayout.THD_THD_THD`) — allows multiple variable-length segments to be packed into a single `[batch, max_seqlen, heads, dim]` tensor, avoiding wasted padding.
* **Cross-attention** — query and key/value come from different sources with different sequence lengths (`max_seqlen_q != max_seqlen_kv`).
* **Padding+causal mask** (`AttnMaskType.PADDING_CAUSAL_MASK`) — required for THD format; combines padding and causal masking.
* **Sliding window** (`window_size=(left, 0)`) — restricts each query to attend only to a local window of keys.
* **SequenceDescriptor.from_seqlens_and_offsets** — for THD, you must provide per-segment lengths and byte offsets.

### Key API concepts ###

1. Build THD segment metadata (segment IDs, positions, lengths, offsets) from your variable-length data.

2. Construct a `SequenceDescriptor` for THD:

```python
seq_desc = SequenceDescriptor.from_seqlens_and_offsets(
    seqlens=(q_seqlens, kv_seqlens),      # [batch, max_segments], -1 = unused
    seq_offsets=(q_offsets, kv_offsets),   # [batch, max_segments + 1], -1 = unused
)
```

3. Pass `max_segments_per_seq` to tell the kernel the maximum number of segments packed per sequence (trades compile time for flexibility):

```python
out = fused_attn(
    qkv=(q, k, v),
    ...,
    qkv_layout=QKVLayout.THD_THD_THD,
    max_segments_per_seq=3,
    window_size=(64, 0),  # attend to 64 tokens to the left only
)
```

4. Or via Flax:

```python
dpa = te_flax.DotProductAttention(
    ...,
    qkv_layout="thd_thd_thd",
    attn_mask_type="padding_causal",
    max_segments_per_seq=3,
    window_size=(64, 0),
)
```

### Run ###

```bash
python test_cross_attn_thd.py
pytest -xvs test_cross_attn_thd.py
```

---

## Example 3 — Context Parallelism (Multi-GPU) ##

**File:** `test_context_parallel.py`

Context parallelism (CP) shards long sequences across multiple GPUs. Each device holds a chunk of the sequence and the fused attention kernel communicates across devices to compute the correct output. This is essential for very long context lengths that exceed single-GPU memory.

* **ALL_GATHER strategy** (`CPStrategy.ALL_GATHER`) — each device gathers the full K/V from all peers and computes its local Q chunk against them.
* **RING strategy** (`CPStrategy.RING`) — ring-attention where K/V chunks are pipelined around a ring of devices.
* **Causal load balancing** — with causal masking, early tokens attend to fewer keys than late tokens, creating load imbalance. `reorder_causal_load_balancing` redistributes tokens so that each device processes a mix of early and late tokens.

### Key API concepts ###

1. Create a JAX mesh with a context-parallel axis:

```python
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec

device_mesh = mesh_utils.create_device_mesh((cp_size,))
mesh = Mesh(device_mesh, axis_names=("cp",))
```

2. Shard Q/K/V along the sequence dimension:

```python
qkv_sharding = NamedSharding(mesh, PartitionSpec(None, "cp", None, None))
q = jax.device_put(q, qkv_sharding)
```

3. Optionally reorder for causal load balancing:

```python
from transformer_engine.jax.attention import reorder_causal_load_balancing, ReorderStrategy

q = reorder_causal_load_balancing(q, ReorderStrategy.DualChunkSwap, cp_size, seq_dim=1)
# ... run attention ...
out = inverse_reorder_causal_load_balancing(out, ReorderStrategy.DualChunkSwap, cp_size, seq_dim=1)
```

4. Call fused attention with CP parameters:

```python
with mesh, autocast(mesh_resource=MeshResource(cp_resource="cp")):
    out = fused_attn(
        ...,
        context_parallel_strategy=CPStrategy.ALL_GATHER,
        context_parallel_causal_load_balanced=True,
        context_parallel_axis="cp",
    )
```

5. Or via Flax:

```python
dpa = te_flax.DotProductAttention(
    ...,
    context_parallel_axis="cp",
    context_parallel_strategy="ALL_GATHER",
    context_parallel_causal_load_balanced=True,
)
```

### Run ###

Context parallelism requires multiple GPUs. Use the launcher script:

```bash
bash run_test_context_parallel.sh
```

Or manually (2 GPUs):

```bash
# Terminal 1:
python test_context_parallel.py --num-process 2 --process-id 0

# Terminal 2:
python test_context_parallel.py --num-process 2 --process-id 1
```

---

## API Summary ##

| Concept | Low-level (`fused_attn`) | Flax (`DotProductAttention`) |
|---|---|---|
| QKV layout | `QKVLayout.BSHD_BSHD_BSHD`, `.THD_THD_THD`, etc. | `qkv_layout="bshd_bshd_bshd"` |
| Mask type | `AttnMaskType.CAUSAL_MASK`, `.PADDING_CAUSAL_MASK` | `attn_mask_type="causal"`, `"padding_causal"` |
| Bias | `AttnBiasType.POST_SCALE_BIAS` + `bias` tensor | `attn_bias_type="post_scale_bias"` + `bias` kwarg |
| GQA | Different `num_heads` in Q vs K/V shapes | `num_gqa_groups=N` |
| Sequence metadata | `SequenceDescriptor.from_seqlens()` / `.from_seqlens_and_offsets()` | `sequence_descriptor=` kwarg |
| Sliding window | `window_size=(left, right)` | `window_size=(left, right)` |
| Context parallel | `context_parallel_axis`, `context_parallel_strategy`, `context_parallel_causal_load_balanced` | Same parameter names as strings |
| Kernel availability | `is_fused_attn_kernel_available(...)` | Automatic fallback with `NVTE_FUSED_ATTN=1` (default) |
