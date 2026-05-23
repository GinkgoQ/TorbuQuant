# Algorithms

The algorithms in this repository are written in layers. The lowest layer
quantizes vectors. The middle layer stores K/V cache state. The attention layer
decides whether compressed bytes are reconstructed for diagnostics or consumed
inside scoring and accumulation.

## MSE Vector Quantization

Implementation: `torbuquant.core.mse.TorbuquantMSE`.

1. Compute vector norms.
2. Normalize vectors.
3. Apply QR or RHT transform.
4. Search Lloyd-Max decision boundaries.
5. Pack centroid indices with `pack_unsigned`.
6. Store norms.
7. Reconstruct with centroid lookup, inverse transform, and norm scaling.

### Tensor Contract

Input vectors are quantized along the last dimension:

```text
[..., dim]
```

Output `MSEData` contains:

```text
indices: [..., packed_dim] uint8
norms:   [...] floating point tensor
bits:    int
dim:     int
```

`packed_dim = ceil(dim * bits / 8)`.

### Search And Reconstruction

`torch.searchsorted` maps rotated coordinates to Lloyd-Max buckets. During
dequantization, packed indices are unpacked, used to gather centroids, then
rotated back. Optional norm correction renormalizes the reconstructed direction
before multiplying by stored norms.

## QJL Product Quantization

Implementation: `torbuquant.core.qjl.TorbuquantQJL` and
`torbuquant.core.polar.TorbuquantProd`.

The product path combines MSE reconstruction with a sign sketch of the residual.
It is used for inner-product experiments and recipe vector packing.

### Residual Flow

```text
x_unit -> MSE reconstruction -> residual
residual -> random projection -> sign bits
```

The sign bits are packed by `pack_sign_bits`. The estimator is useful for raw
inner products, but the attention path must still pass quality checks because
softmax changes the error model.

## Value Quantization

Implementation: `torbuquant.kv.values`.

Values use group-wise integer quantization with per-group scale and zero
metadata. V4 packs two values per byte; V2 packs four values per byte; V3 uses
3-bit bitstream packing.

### Group Formula

For dimension `dim` and group size `g`:

```text
groups = ceil(dim / g)
payload bytes = outer_size * ceil(dim * bits / 8)
scale bytes = outer_size * groups * sizeof(scale)
zero bytes = outer_size * groups * sizeof(zero)
```

The memory report includes payload, scale, and zero bytes separately.

## Direct-QK Diagnostic Attention

Implementation: `torbuquant.attention.reference.direct_qk_attention`.

This path computes scores from compressed keys and can accumulate compressed
values through `weighted_packed_v_accumulation`. It is a reference path used to
validate layout and math.

### GQA Mapping

For query heads `Hq` and KV heads `Hkv`, query head `h` maps to:

```text
kv_head = h // (Hq / Hkv)
```

The helper `gqa_kv_heads` creates this mapping as a tensor.

## TQ Packed Page Update

Implementation: `torbuquant.triton.tq4_update`.

The current writer:

1. normalizes raw K/V rows,
2. rotates them,
3. assigns nearest centroids,
4. packs unsigned indices,
5. appends fp32 norm bytes,
6. writes rows by flat slot mapping into `[blocks, block_size, heads, row_bytes]`.

Negative slots are ignored. Out-of-range slots raise `IndexError`.

### Row Layout

For bit width `b` and head dimension `d`:

```text
index_bytes = ceil(d * b / 8)
row_bytes = index_bytes + 4
row = [packed_indices][fp32_norm_bytes]
```

The writer accepts raw rows shaped:

```text
[tokens, kv_heads, head_dim]
```

and pages shaped:

```text
[blocks, block_size, kv_heads, row_bytes]
```

## TQ Paged Decode Reference

Implementation: `torbuquant.triton.tq4_decode`.

The current paged decode reference:

1. reads physical pages through a block table,
2. unpacks key rows,
3. computes scaled scores per GQA group,
4. applies softmax,
5. unpacks value rows,
6. accumulates output.

The file name is under `torbuquant.triton`, but this specific path is a PyTorch
contract implementation.

### Block Table Semantics

For each logical token position:

```text
logical_block = token // block_size
offset = token % block_size
physical_block = block_table[sequence, logical_block]
```

The decode reference uses this mapping to read K/V rows from page tensors. It
supports GQA and sliding-window restriction.

## Recipe Vector Packing

Implementation: `torbuquant.integration.vllm.vector`.

Recipe packing supports `turboquant25` and `turboquant35`:

- gather high-precision and low-precision channel groups,
- normalize each group,
- apply block-Hadamard transforms,
- pack MSE centroid indices,
- compute residuals,
- pack QJL signs,
- store vector and residual norm bytes,
- concatenate group rows into one `uint8` row.

### Recipe Constants

| Recipe | High group ratio | Group bits | MSE bits | Packed bytes at head dim 128 |
| --- | ---: | --- | --- | ---: |
| `turboquant25` | `0.25` | `(3, 2)` | `(2, 1)` | `44` |
| `turboquant35` | `0.50` | `(4, 3)` | `(3, 2)` | `64` |

Norm storage:

```text
vector norm bytes = 2
residual norm bytes = 2
```

The recipe row stores K and V independently when used in a cache.
