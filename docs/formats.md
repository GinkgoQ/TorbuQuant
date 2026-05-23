# Input And Output Formats

Formats in TorbuQuant describe both tensor meaning and byte layout. The same
mathematical quantizer can appear in several storage layouts, so the docs use
format names carefully.

## MSE Payload

`MSEData` stores:

| Field | Type | Meaning |
| --- | --- | --- |
| `indices` | `torch.uint8` tensor | Packed centroid indices. |
| `norms` | tensor | Vector norms. |
| `bits` | `int` | Bits per coordinate. |
| `dim` | `int` | Original vector dimension. |

Packed index width:

```text
ceil(dim * bits / 8)
```

The last dimension of `indices` is packed; all leading dimensions mirror the
input vector batch shape.

## Value Payload

`ValueData` stores:

| Field | Meaning |
| --- | --- |
| `data` | Packed value indices. |
| `scales` | Per-group scale values. |
| `zeros` | Per-group zero points. |
| `bits` | Value bits. |
| `dim` | Original dimension. |
| `group_size` | Value quantization group size. |

For a value tensor shaped:

```text
[tokens, heads, dim]
```

the scale and zero tensors have one value per group:

```text
[tokens, heads, ceil(dim / group_size)]
```

## TQ Row Layout

For one token and one KV head:

```text
[packed_indices][fp32_norm_bytes]
```

The row byte count is:

```text
ceil(head_dim * bits / 8) + 4
```

K and V pages may be stored as separate tensors or as regions of one combined
page tensor, depending on the caller.

### Example

For `head_dim=128` and `bits=4`:

```text
packed_indices = 64 bytes
norm = 4 bytes
row = 68 bytes
```

For K4/V4, one token and one KV head uses:

```text
68 + 68 = 136 bytes
```

## Paged Cache Shape

`PackedPageCache` stores:

```text
[num_blocks, block_size, num_kv_heads, row_bytes]
```

Slot mapping uses flat token slots:

```text
block = slot // block_size
offset = slot % block_size
```

Negative slots are ignored.

Out-of-range non-negative slots raise `IndexError`. This is intentional because
a write outside allocated pages indicates allocator/runtime mismatch.

## Recipe Metadata

Recipe metadata stores selected channel indices for keys and values:

```text
key_high_precision_indices
value_high_precision_indices
```

Each KV head must have the recipe-defined number of selected indices.

For `turboquant25` at `head_dim=128`, each KV head needs 32 high-group indices.
For `turboquant35`, each KV head needs 64.

## Recipe Row Layout

For each group:

```text
[MSE index bitstream][QJL sign bitstream][vector norm bytes][residual norm bytes]
```

Group fields:

| Field | Meaning |
| --- | --- |
| MSE index bitstream | Lloyd-Max centroid ids for transformed unit-group coordinates. |
| QJL sign bitstream | One sign bit per residual coordinate. |
| vector norm bytes | fp16 byte representation of the group norm. |
| residual norm bytes | fp16 byte representation of the residual norm. |

The two groups are concatenated:

```text
row = high_group_row || low_group_row
```

At `head_dim=128`:

| Recipe | Row bytes |
| --- | ---: |
| `turboquant25` | 44 |
| `turboquant35` | 64 |

## Qwen A/B Artifacts

`scripts/qwen_ab_eval.py` writes:

- scenario manifest,
- raw JSONL outputs,
- metrics JSON,
- comparison report,
- separate baseline and TurboQuant directories.

The A/B script also records artifact manifests so stale or shared files are
visible during comparison.
