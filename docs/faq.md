# FAQ

## Is HuggingFace DynamicCache compression a serving-speed path?

No. The wrapper returns dense tensors to HuggingFace attention. Use it for
diagnostics, byte reports, and quality checks.

## Does smaller storage imply lower latency?

No. Packed storage can reduce memory pressure, but latency depends on the decode
kernel and whether dense historical K/V is materialized.

## Which model is the primary target?

`Qwen/Qwen2.5-3B`.

## Which formats should I test first?

Use dense baseline and q8/FP8 baselines when available. Within this repository,
K16V4 and K8V4 have more quality margin than K4V4 in the reported Qwen
activation checks.

## Can QJL be enabled by default for attention?

Not yet. QJL has raw inner-product guarantees under the paper setup, but
attention softmax and autoregressive generation require model measurements.

## Does this repo import reference implementations?

No. Code under `other_implemnetations/` is used for comparison and audit only.

## Are recipe modes ready for live vLLM serving?

No. Recipe layout, metadata, vector packing, and calibration pieces exist.
Recipe update/decode kernels and live backend wiring remain open work.

