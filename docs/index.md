<section class="tq-hero">
  <div class="tq-hero-copy">
    <div class="tq-eyebrow">KV cache compression · vector quantization · serving diagnostics</div>
    <h1>TorbuQuant</h1>
    <p>
      TorbuQuant is a Python package for LLM KV-cache compression, vector
      quantization, cache accounting, attention diagnostics, and runtime
      integration work for HuggingFace and vLLM.
    </p>
    <p>
      The repository is built around a measured goal: reduce stored KV bytes,
      preserve model behavior under explicit gates, and improve serving capacity
      when KV memory, context length, or batch size is the limiting factor.
    </p>
    <div class="tq-actions">
      <a class="tq-button primary" href="quickstart/">Quickstart</a>
      <a class="tq-button secondary" href="design/">Read the design</a>
      <a class="tq-button secondary" href="api/">API reference</a>
    </div>
  </div>
  <div class="tq-hero-visual" role="img" aria-label="TQ4 packed page byte layout showing page tensor shape, slot row, key region, value region, and decode flow.">
    <div class="tq-diagram-head">
      <span>TQ4 page tensor</span>
      <code>[num_blocks, block_size, num_kv_heads, slot_bytes]</code>
    </div>
    <div class="tq-page-card">
      <div class="tq-page-meta">
        <span>slot 17</span>
        <span>block = slot // block_size</span>
        <span>row = slot % block_size</span>
      </div>
      <div class="tq-slot-address">
        <span>slot row</span>
        <code>page[block, row, kv_head, :]</code>
      </div>
      <div class="tq-region-row">
        <span>K region bytes</span>
        <span>V region bytes</span>
      </div>
      <div class="tq-component-row tq-key-row">
        <b>K</b>
        <span class="tq-payload">packed K indices</span>
        <span>fp32 norm</span>
      </div>
      <div class="tq-component-row tq-value-row">
        <b>V</b>
        <span class="tq-payload">packed V indices</span>
        <span>fp32 norm</span>
      </div>
      <div class="tq-byte-formula">
        <span><b>K bytes</b><code>ceil(D * k_bits / 8) + 4B</code></span>
        <span><b>V bytes</b><code>ceil(D * v_bits / 8) + 4B</code></span>
        <span><b>slot_bytes</b><code>K + V, optional power-of-two padding</code></span>
      </div>
    </div>
    <div class="tq-route-line">
      <span>slot</span>
      <i></i>
      <span>slot row</span>
      <i></i>
      <span>decode</span>
    </div>
    <p>This TQ4 page path uses the same norm-coded row packer for K and V; other V formats in the library may use scale and zero metadata.</p>
  </div>
</section>

<div class="tq-metrics">
  <div class="tq-metric">
    <strong>Diagnostic and serving routes</strong>
    <span>Separated by design</span>
  </div>
  <div class="tq-metric">
    <strong>Packed K/V accounting</strong>
    <span>Metadata included</span>
  </div>
  <div class="tq-metric">
    <strong>Qwen A/B evaluation</strong>
    <span>Measured comparison</span>
  </div>
</div>

## What Makes This Repository Different

TorbuQuant is documented as a research implementation and a serving-system
prototype at the same time. The documentation therefore records:

- which formulas come from the paper,
- which paths are diagnostic,
- which paths own packed cache storage,
- which byte formulas are layout estimates,
- which measurements have been run,
- which vLLM pieces are still contracts rather than live hooks.

That separation is intentional. It prevents a storage result from being
misread as a serving result.

<div class="tq-flow">
  <div>
    <b>1. Quantize</b>
    <span>Rotate, normalize, codebook-encode, and pack tensor rows.</span>
  </div>
  <div>
    <b>2. Store</b>
    <span>Keep historical K/V in compact uint8 page or row layouts.</span>
  </div>
  <div>
    <b>3. Attend</b>
    <span>Use diagnostic or packed-reference attention paths by intent.</span>
  </div>
  <div>
    <b>4. Measure</b>
    <span>Report memory, quality, latency, throughput, and raw outputs.</span>
  </div>
</div>

## Current Status

| Area | Status |
| --- | --- |
| Lloyd-Max codebooks | Implemented for exact Beta-sphere and Gaussian approximation. |
| Rotations | QR and RHT are implemented. |
| MSE vector quantization | Implemented with packed index storage. |
| QJL residual | Implemented for vector and inner-product experiments. |
| KV formats | K16, K8, K4 and V8, V4, V3, V2 helpers exist. |
| Diagnostic attention | Dense, dequantized, direct-QK, and packed-V reference paths exist. |
| Triton decode | Format-specific kernels exist for selected non-paged paths. |
| TQ packed pages | Page storage, PyTorch row packing, and paged decode reference exist. |
| HuggingFace | Diagnostic DynamicCache wrappers and Qwen capture helpers exist. |
| vLLM | Metadata, page geometry, registry, runtime selection, and verification helpers exist. |
| vLLM live backend | Not yet wired as a live vLLM attention backend in this repository. |
| Qwen A/B | Scripted A/B evaluation exists for HF diagnostic runs. |

## What This Documentation Covers

The site documents the code currently present in this repository. Where a
serving feature is not wired end to end, the page says so explicitly. It does
not import or depend on code under `other_implemnetations/`.

<div class="tq-card-grid">
  <div class="tq-card">
    <h3><a href="install/">Install</a></h3>
    <p>Environment setup, optional packages, and validation commands.</p>
  </div>
  <div class="tq-card">
    <h3><a href="quickstart/">Quickstart</a></h3>
    <p>Small commands that exercise quantization, cache accounting, and evaluation paths.</p>
  </div>
  <div class="tq-card">
    <h3><a href="design/">Design</a></h3>
    <p>Route ownership, data contracts, diagnostic boundaries, and serving constraints.</p>
  </div>
  <div class="tq-card">
    <h3><a href="api/">API reference</a></h3>
    <p>Generated module reference from the repository's Python source.</p>
  </div>
</div>

## Reader Paths

| Reader | Suggested path |
| --- | --- |
| Researcher | [Math](math.md), [Algorithms](algorithms.md), [Limits](limits.md). |
| Runtime engineer | [Design](design.md), [Architecture](architecture.md), [vLLM Usage](usage/vllm.md). |
| Model evaluator | [Qwen A/B](usage/qwen-ab.md), [Performance Notes](performance.md), [Troubleshooting](troubleshooting.md). |
| Contributor | [Developer Guide](developer.md), [Contributing](contributing.md), [API Reference](api/index.md). |
