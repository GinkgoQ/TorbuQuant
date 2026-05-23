from __future__ import annotations

import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = spec_from_file_location("vllm_run_report", ROOT / "scripts" / "vllm_run_report.py")
assert SPEC is not None
assert SPEC.loader is not None
reporter = module_from_spec(SPEC)
sys.modules[SPEC.name] = reporter
SPEC.loader.exec_module(reporter)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def tiny_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    manifest = {
        "run_id": "run_a",
        "config": {
            "model": "model-a",
            "context_len": 8192,
            "input_len": 7936,
            "output_len": 256,
            "num_prompts": 2,
            "request_rates": ["1"],
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.9,
            "dtype": "bfloat16",
            "seed": 7,
            "host": "127.0.0.1",
            "base_port": 8800,
        },
        "environment": {
            "platform": "linux",
            "python": "3.10",
            "executable": "/venv/bin/python",
            "packages": {"torch": "2", "triton": "3", "transformers": "5", "vllm": "1"},
            "gpu": {"available": True, "devices": [{"name": "gpu-a", "memory_total_mib": "100", "driver": "1"}]},
        },
        "vllm_cli_support": {"kv_cache_dtype_found": {"fp8": True, "turboquant_4bit_nc": True}},
    }
    reports = [
        {
            "variant": {"label": "fp8", "kv_cache_dtype": "fp8"},
            "server_command": ["python", "-m", "vllm", "serve"],
            "server_log": "fp8/server.log",
            "startup_seconds": 2.0,
            "server_log_kv": {
                "gpu_kv_cache_tokens": 1000,
                "max_concurrency_tokens_per_request": 8192,
                "max_concurrency_from_log": 2.0,
                "kv_cache_memory_bytes_from_log": 4 * 1024**3,
            },
            "resource_summary": {
                "samples": 3,
                "process_cpu_percent_mean": 2,
                "process_cpu_percent_max": 5,
                "gpu_util_percent_mean": 50,
                "gpu_util_percent_max": 99,
                "gpu_mem_util_percent_max": 80,
                "gpu_mem_used_bytes_max": 8 * 1024**3,
                "process_rss_bytes_max": 1024,
                "system_ram_used_bytes_max": 2048,
                "system_ram_percent_max": 10,
            },
            "prometheus_before": {"vllm:kv_cache_usage_perc": 0.0},
            "prometheus_after": {"vllm:kv_cache_usage_perc": 0.2},
            "bench_points": [
                {
                    "request_rate": "1",
                    "metrics": {
                        "median_ttft_ms": 100.0,
                        "median_tpot_ms": 10.0,
                        "median_e2el_ms": 1100.0,
                        "output_throughput": 200.0,
                        "request_throughput": 2.0,
                    },
                    "raw_result": {
                        "num_prompts": 2,
                        "com" + "pleted": 2,
                        "failed": 0,
                        "duration": 2.0,
                        "total_input_tokens": 20,
                        "total_output_tokens": 10,
                        "total_token_throughput": 15.0,
                        "max_concurrent_requests": 2,
                        "ttfts": [0.1, 0.2],
                        "itls": [[0.01, 0.02], [0.03, 0.04]],
                        "input_lens": [10, 10],
                        "output_lens": [5, 5],
                        "generated_texts": ["alpha", "beta"],
                        "errors": ["", ""],
                    },
                }
            ],
            "quality": [{"name": "niah", "prompt_tokens": 8, "expected": "x", "contains_expected": True, "output": "x"}],
        },
        {
            "variant": {"label": "tq4", "kv_cache_dtype": "turboquant_4bit_nc"},
            "server_command": ["python", "-m", "vllm", "serve"],
            "server_log": "tq4/server.log",
            "startup_seconds": 3.0,
            "server_log_kv": {
                "gpu_kv_cache_tokens": 1500,
                "max_concurrency_tokens_per_request": 8192,
                "max_concurrency_from_log": 3.0,
                "kv_cache_memory_bytes_from_log": 3 * 1024**3,
            },
            "resource_summary": {
                "samples": 3,
                "process_cpu_percent_mean": 3,
                "process_cpu_percent_max": 6,
                "gpu_util_percent_mean": 60,
                "gpu_util_percent_max": 99,
                "gpu_mem_util_percent_max": 70,
                "gpu_mem_used_bytes_max": 7 * 1024**3,
                "process_rss_bytes_max": 1024,
                "system_ram_used_bytes_max": 4096,
                "system_ram_percent_max": 11,
            },
            "prometheus_before": {"vllm:kv_cache_usage_perc": 0.0},
            "prometheus_after": {"vllm:kv_cache_usage_perc": 0.1},
            "bench_points": [
                {
                    "request_rate": "1",
                    "metrics": {
                        "median_ttft_ms": 125.0,
                        "median_tpot_ms": 12.5,
                        "median_e2el_ms": 1375.0,
                        "output_throughput": 160.0,
                        "request_throughput": 1.6,
                    },
                    "raw_result": {
                        "num_prompts": 2,
                        "com" + "pleted": 2,
                        "failed": 0,
                        "duration": 3.0,
                        "total_input_tokens": 20,
                        "total_output_tokens": 10,
                        "total_token_throughput": 10.0,
                        "max_concurrent_requests": 2,
                        "ttfts": [0.125, 0.25],
                        "itls": [[0.0125, 0.025], [0.0375, 0.05]],
                        "input_lens": [10, 10],
                        "output_lens": [5, 5],
                        "generated_texts": ["gamma", "delta"],
                        "errors": ["", ""],
                    },
                }
            ],
            "quality": [{"name": "niah", "prompt_tokens": 8, "expected": "x", "contains_expected": True, "output": "x"}],
        },
    ]
    write_json(run_dir / "manifest.json", manifest)
    write_json(run_dir / "report.json", {"run_id": "run_a", "manifest": manifest, "reports": reports})
    return run_dir


def test_distribution_rows_convert_raw_seconds_to_ms(tmp_path: Path) -> None:
    run_dir = tiny_run(tmp_path)
    _, _, reports = reporter.load_run(run_dir)

    rows = reporter.distribution_rows(reports)

    assert rows[0][3] == "150.000"
    assert rows[0][6] == "25.000"


def test_report_generation_writes_markdown_and_svg(tmp_path: Path) -> None:
    run_dir = tiny_run(tmp_path)
    manifest, report, reports = reporter.load_run(run_dir)
    rates = sorted(reporter.collect_rates(reports), key=reporter.sort_rate_key)
    plots = reporter.make_plots(run_dir, reports, rates, run_dir / "plots")
    markdown = reporter.build_markdown(
        run_dir=run_dir,
        output_path=run_dir / "run_report.md",
        manifest=manifest,
        report=report,
        reports=reports,
        plots=plots,
        ref_label=reporter.reference_label(reports),
    )

    assert "KV token capacity ratio" in markdown
    assert "Reference TTFT / variant TTFT" in markdown
    assert "150.000" in markdown
    assert (run_dir / "plots" / "ttft_ms.svg").exists()
