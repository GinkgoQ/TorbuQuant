# Mathematical Notes

This page states the mathematical claims used by the code. It does not convert
those claims into LLM quality or serving claims.

## Symbols

| Symbol | Meaning |
| --- | --- |
| `d` | Vector dimension. For Qwen/Qwen2.5-3B attention heads this is typically `128`. |
| `b` | Bits per coordinate. |
| `x` | Original vector. |
| `u` | Unit vector `x / ||x||_2`. |
| `Pi` | Rotation or structured transform. |
| `q(.)` | Scalar Lloyd-Max quantizer applied coordinate-wise. |
| `S` | QJL random projection matrix. |
| `r` | Residual after MSE reconstruction. |

## Unit-Vector Setup

TurboQuant quantizes a vector `x in R^d` by separating norm and direction:

```text
r = ||x||_2
u = x / max(r, eps)
```

The quantizer stores `r` and quantizes `u`.

The norm is part of the representation. It is counted in memory reports.

## Random Rotation

For a unit vector `u`, a random rotation `Pi` spreads mass across coordinates:

```text
y = Pi u
```

For Haar rotation, each coordinate follows the Beta-sphere density:

```text
f(t) = Gamma(d / 2) / (sqrt(pi) Gamma((d - 1) / 2))
       (1 - t^2)^((d - 3) / 2),  -1 <= t <= 1
```

The implementation supports:

- QR rotation for dense diagnostic transforms,
- RHT for structured transforms,
- block Hadamard transforms in recipe vector packing.

RHT is not identical to Haar rotation, so model-level claims require measured
evidence.

### Why Rotation Matters

Without rotation, a few coordinates can dominate the error. Rotation makes the
coordinate distribution more predictable, which allows a single scalar
codebook to be used across coordinates. The paper's density model applies to
Haar rotation. The code supports QR rotation to approximate that diagnostic
setting and RHT/block-Hadamard transforms for practical structured paths.

## Lloyd-Max Scalar Quantization

For each coordinate of the rotated unit vector, Lloyd-Max computes centroids
and boundaries for the chosen coordinate density. The project implements exact
Beta-sphere codebooks and a Gaussian approximation option.

For centroids `c_i` and boundaries `b_i`, quantization maps:

```text
q(y_j) = c_i  when b_i <= y_j < b_{i+1}
```

Reconstruction:

```text
x_hat = ||x||_2 Pi^T q(Pi x / ||x||_2)
```

The code path is:

| Math step | Code |
| --- | --- |
| density | `beta_pdf` |
| centroid solve | `compute_lloyd_max` |
| cached codebook | `get_codebook`, `get_codebook_tensors` |
| boundary search | `TorbuquantMSE.quantize` |
| reconstruction | `TorbuquantMSE.dequantize` |

## Vector Distortion Bound

Under the paper assumptions, the MSE quantizer has vector distortion scaling:

```text
E[||x - x_hat||_2^2] <= (sqrt(3) pi / 2) 4^(-b)
```

This is a vector reconstruction statement. It is not a statement about logits,
perplexity, generated text, or serving throughput.

For non-unit vectors, the implementation stores `||x||_2` and applies the
directional reconstruction to the normalized vector.

## QJL Residual

The QJL path stores signs of a sketch:

```text
z = sign(S r)
```

where `r` is the residual after MSE reconstruction. The estimator scales with:

```text
sqrt(pi / 2)
```

The paper gives an unbiased raw inner-product estimator under its setup. In
attention, scores pass through softmax, so QJL must be tested on model outputs
before it is used as a default attention path.

## Attention Error Boundary

Raw score error is not the same as output error. Attention computes:

```text
scores = Q K^T / sqrt(d)
weights = softmax(scores)
output = weights V
```

Small key errors can change score ranking. Value errors are weighted by the
softmax distribution. Autoregressive generation then feeds output changes into
future layers and tokens. This is why the repository requires model-level
checks in addition to random tensor tests.

## Recipe Group Math

For recipe mode, a vector is partitioned into high and low channel groups:

```text
x = scatter(x_high, x_low)
```

Each group is normalized and encoded separately:

```text
g_norm = ||g||_2
g_unit = g / max(g_norm, eps)
mse = q(H g_unit)
residual = g_unit - H^-1 mse
qjl = sign(H_res residual)
```

The stored group representation contains MSE payload, QJL sign payload, vector
norm, and residual norm. The output row concatenates both groups.

## Compression Ratio Formula

A useful byte-level ratio for one K or V component is:

```text
dense_bytes = tokens * heads * dim * dense_element_bytes
packed_bytes = tokens * heads * (ceil(dim * bits / 8) + norm_bytes)
ratio = dense_bytes / packed_bytes
```

This formula is only a storage formula. Runtime memory reports must also count
metadata and workspace.

## Memory Accounting

Compressed memory is not only index bytes. A valid byte report counts:

- packed key bytes,
- packed value bytes,
- scales and zeros,
- vector norms,
- residual norms,
- centroids,
- transform metadata,
- recent dense window,
- boundary-token storage,
- sparse/outlier storage,
- temporary workspace,
- CUDA peak allocation when measured.
