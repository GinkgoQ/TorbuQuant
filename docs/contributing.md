# Contributing

## Patch Expectations

Contributions should be small enough to review and should include tests for the
behavior they change.

Required for relevant changes:

- unit tests for math and packing,
- integration tests for cache/runtime paths,
- byte accounting tests,
- quality checks for model-facing behavior,
- documentation updates for public APIs.

## Reports And Claims

A benchmark contribution must include:

- command,
- model id and revision,
- hardware,
- software versions,
- baseline,
- repeat count,
- memory report,
- raw outputs for generation comparisons.

## Style

Follow existing module boundaries:

- math in `core`,
- storage and policy in `kv`,
- attention reference paths in `attention`,
- runtime adapters in `integration`,
- kernels and kernel contracts in `triton`.

## Reference Code

Do not import from `other_implemnetations/`. Use it only as reading material.

