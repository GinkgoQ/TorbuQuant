from pathlib import Path


def test_vllm_reference_audit_tracks_required_rows():
    audit = Path("docs/vllm_reference_audit.md").read_text(encoding="utf-8")

    required = (
        "turboquant-vllm/src/turboquant_vllm/kv_cache.py",
        "vllm-turboquant/vllm/v1/attention/ops/turboquant_kv_cache.py",
        "vllm-turboquant/vllm/v1/attention/ops/triton_turboquant_kv_update.py",
        "vllm-turboquant/benchmarks/generate_turboquant_metadata.py",
        "activation calibration",
        "packed paged decode kernel",
    )
    for item in required:
        assert item in audit


def test_vllm_reference_audit_has_phase_rule():
    audit = Path("docs/vllm_reference_audit.md").read_text(encoding="utf-8")

    assert "Each later vLLM phase must update the affected row" in audit
