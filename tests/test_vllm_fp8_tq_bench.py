from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("vllm_fp8_tq_bench", ROOT / "scripts" / "vllm_fp8_tq_bench.py")
assert SPEC is not None
assert SPEC.loader is not None
bench = module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)

BenchPoint = bench.BenchPoint
Variant = bench.Variant
VariantReport = bench.VariantReport
VLLM_COMMAND = bench.VLLM_COMMAND
build_variants = bench.build_variants
check_vllm_cli_support = bench.check_vllm_cli_support
compare_variant = bench.compare_variant
extract_bench_metrics = bench.extract_bench_metrics
add_point_resource_metrics = bench.add_point_resource_metrics
parse_bench_stdout = bench.parse_bench_stdout
parse_prometheus = bench.parse_prometheus
parse_server_log_kv = bench.parse_server_log_kv
resolve_input_len = bench.resolve_input_len
validate_bench_result = bench.validate_bench_result


def test_vllm_command_uses_current_python() -> None:
    assert VLLM_COMMAND[0] == sys.executable


def test_build_variants_maps_upstream_dtype_names() -> None:
    variants = build_variants("fp8,tq4,tq_k8v4,turboquant_3bit_nc")

    assert [item.label for item in variants] == [
        "fp8",
        "tq4",
        "tq_k8v4",
        "turboquant_3bit_nc",
    ]
    assert [item.kv_cache_dtype for item in variants] == [
        "fp8",
        "turboquant_4bit_nc",
        "turboquant_k8v4",
        "turboquant_3bit_nc",
    ]


def test_resolve_input_len_leaves_output_budget() -> None:
    assert resolve_input_len(8192, 256, None) == 7936
    assert resolve_input_len(8192, 256, 4096) == 4096


def test_resolve_input_len_rejects_over_budget() -> None:
    try:
        resolve_input_len(8192, 256, 8192)
    except ValueError as exc:
        assert "--input-len + --output-len" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_validate_bench_result_rejects_all_failed_requests(tmp_path: Path) -> None:
    server_log = tmp_path / "server.log"
    server_log.write_text("VLLMValidationError: prompt and output exceed context\n", encoding="utf-8")
    try:
        validate_bench_result(
            {"completed": 0, "failed": 1, "errors": ["Bad Request"]},
            variant=Variant(label="fp8", kv_cache_dtype="fp8"),
            request_rate="1",
            stdout="",
            stderr="",
            server_log_path=server_log,
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "zero successful requests" in message
        assert "Bad Request" in message
        assert "VLLMValidationError" in message
    else:
        raise AssertionError("expected RuntimeError")


def test_check_vllm_cli_support_reads_installed_fp8_dtype() -> None:
    support = check_vllm_cli_support([Variant(label="fp8", kv_cache_dtype="fp8")])
    if support["returncode"] != 0:
        pytest.skip("vLLM CLI is not usable in this test environment")

    assert support["returncode"] == 0
    assert support["kv_cache_dtype_found"]["fp8"] is True


def test_parse_server_log_kv_extracts_capacity_and_oom() -> None:
    log = """
    INFO GPU KV cache size: 131,072 tokens
    INFO Maximum concurrency for 8,192 tokens per request: 16.00x
    INFO available KV cache memory: 21.5 GiB
    ERROR CUDA out of memory
    """

    parsed = parse_server_log_kv(log)

    assert parsed["gpu_kv_cache_tokens"] == 131072
    assert parsed["max_concurrency_tokens_per_request"] == 8192
    assert parsed["max_concurrency_from_log"] == 16.0
    assert parsed["kv_cache_memory_bytes_from_log"] == int(21.5 * 1024**3)
    assert parsed["oom_mentions"] == 1


def test_extract_bench_metrics_from_saved_vllm_shape() -> None:
    result = {
        "request_throughput": 7.5,
        "output_throughput": 1024.0,
        "percentiles": {
            "percentile_ttft_50_ms": 91.0,
            "percentile_ttft_90_ms": 140.0,
            "percentile_ttft_99_ms": 300.0,
            "percentile_tpot_50_ms": 12.5,
            "percentile_tpot_90_ms": 19.0,
            "percentile_tpot_99_ms": 40.0,
            "percentile_itl_50_ms": 12.0,
            "percentile_itl_90_ms": 18.0,
            "percentile_itl_99_ms": 39.0,
            "percentile_e2el_50_ms": 2000.0,
            "percentile_e2el_90_ms": 2600.0,
            "percentile_e2el_99_ms": 3100.0,
        },
        "request_outputs": [
            {"success": True},
            {"success": False, "error": "server error"},
        ],
    }

    metrics = extract_bench_metrics(result)

    assert metrics["request_throughput"] == 7.5
    assert metrics["output_throughput"] == 1024.0
    assert metrics["median_ttft_ms"] == 91.0
    assert metrics["p90_ttft_ms"] == 140.0
    assert metrics["p99_ttft_ms"] == 300.0
    assert metrics["median_tpot_ms"] == 12.5
    assert metrics["p90_tpot_ms"] == 19.0
    assert metrics["p99_tpot_ms"] == 40.0
    assert metrics["median_itl_ms"] == 12.0
    assert metrics["p90_itl_ms"] == 18.0
    assert metrics["p99_itl_ms"] == 39.0
    assert metrics["median_e2el_ms"] == 2000.0
    assert metrics["p90_e2el_ms"] == 2600.0
    assert metrics["p99_e2el_ms"] == 3100.0
    assert metrics["failed_requests"] == 1


def test_add_point_resource_metrics_adds_peak_and_per_gb_value() -> None:
    metrics = add_point_resource_metrics(
        {"output_throughput": 2048.0},
        {
            "gpu_mem_used_bytes_max": 8 * 1024**3,
            "process_rss_bytes_max": 3,
            "system_ram_used_bytes_max": 5,
        },
    )

    assert metrics["peak_gpu_memory_bytes"] == 8 * 1024**3
    assert metrics["process_rss_bytes_max"] == 3
    assert metrics["system_ram_used_bytes_max"] == 5
    assert metrics["output_tokens_per_s_per_peak_gpu_gb"] == 256.0


def test_parse_bench_stdout_fallback() -> None:
    stdout = """
    Request throughput (req/s): 2.50
    Output token throughput (tok/s): 512.00
    Median TTFT (ms): 110.00
    P99 TTFT (ms): 220.00
    Median TPOT (ms): 9.50
    P99 TPOT (ms): 31.00
    """

    parsed = parse_bench_stdout(stdout)

    assert parsed["request_throughput"] == 2.5
    assert parsed["output_throughput"] == 512.0
    assert parsed["median_ttft_ms"] == 110.0
    assert parsed["p99_ttft_ms"] == 220.0
    assert parsed["median_tpot_ms"] == 9.5
    assert parsed["p99_tpot_ms"] == 31.0


def test_parse_prometheus_preserves_numeric_cache_metrics() -> None:
    text = """
    # HELP vllm:num_requests_running Running requests.
    vllm:num_requests_running{model_name="m"} 3
    vllm:gpu_cache_usage_perc{model_name="m"} 0.75
    python_info{implementation="CPython"} 1.0
    """

    parsed = parse_prometheus(text)

    assert parsed["vllm:num_requests_running"] == 3.0
    assert parsed["vllm:gpu_cache_usage_perc"] == 0.75


def test_compare_variant_uses_separate_ratio_directions() -> None:
    fp8 = VariantReport(
        variant=Variant(label="fp8", kv_cache_dtype="fp8"),
        out_dir="fp8",
        server_command=[],
        server_log="",
        startup_seconds=1.0,
        server_log_kv={"gpu_kv_cache_tokens": 100},
        prometheus_before={},
        prometheus_after={},
        resource_summary={},
        bench_points=[
            BenchPoint(
                variant="fp8",
                request_rate="1",
                result_path=None,
                raw_result={},
                metrics={
                    "median_ttft_ms": 100.0,
                    "median_tpot_ms": 10.0,
                    "output_throughput": 1000.0,
                    "request_throughput": 4.0,
                    "peak_gpu_memory_bytes": 10,
                    "output_tokens_per_s_per_peak_gpu_gb": 100.0,
                },
                resource_summary={},
            )
        ],
    )
    tq = VariantReport(
        variant=Variant(label="tq4", kv_cache_dtype="turboquant_4bit_nc"),
        out_dir="tq4",
        server_command=[],
        server_log="",
        startup_seconds=1.0,
        server_log_kv={"gpu_kv_cache_tokens": 250},
        prometheus_before={},
        prometheus_after={},
        resource_summary={},
        bench_points=[
            BenchPoint(
                variant="tq4",
                request_rate="1",
                result_path=None,
                raw_result={},
                metrics={
                    "median_ttft_ms": 80.0,
                    "median_tpot_ms": 20.0,
                    "output_throughput": 700.0,
                    "request_throughput": 3.0,
                    "peak_gpu_memory_bytes": 5,
                    "output_tokens_per_s_per_peak_gpu_gb": 140.0,
                },
                resource_summary={},
            )
        ],
    )

    comparison = compare_variant(fp8, tq)
    row = comparison["by_request_rate"][0]

    assert row["ttft_ratio_ref_over_candidate"] == 1.25
    assert row["tpot_ratio_ref_over_candidate"] == 0.5
    assert row["throughput_ratio_candidate_over_ref"] == 0.7
    assert row["request_ratio_candidate_over_ref"] == 0.75
    assert row["peak_gpu_memory_ratio_ref_over_candidate"] == 2.0
    assert row["tokens_per_peak_gpu_ratio_candidate_over_ref"] == 1.4
    assert comparison["cache_token_capacity_ratio_candidate_over_ref"] == 2.5
