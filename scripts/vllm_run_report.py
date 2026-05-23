"""Build a Markdown report from a saved vLLM FP8/TurboQuant run.

The input is a run directory produced by ``scripts/vllm_fp8_tq_bench.py``.
The reporter reads saved JSON, logs, and text artifacts only. It does not run
the model and does not infer missing benchmark results.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


BYTES_IN_GIB = 1024**3
DEFAULT_REFERENCE_LABEL = "fp8"
TEXT_LIMIT = 240


@dataclass(frozen=True)
class SeriesPoint:
    label: str
    group: str
    value: float | None


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object at {path}")
    return data


def maybe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 3) -> str:
    number = as_float(value)
    if number is None:
        return "n/a"
    if abs(number) >= 1000:
        return f"{number:,.{digits}f}"
    return f"{number:.{digits}f}"


def fmt_int(value: Any) -> str:
    number = as_int(value)
    if number is None:
        return "n/a"
    return f"{number:,}"


def fmt_bytes(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return "n/a"
    gib = number / BYTES_IN_GIB
    return f"{gib:.2f} GiB"


def ratio(num: Any, den: Any) -> float | None:
    numerator = as_float(num)
    denominator = as_float(den)
    if numerator is None or denominator in (None, 0.0):
        return None
    return numerator / denominator


def pct_change(new: Any, old: Any) -> float | None:
    old_value = as_float(old)
    new_value = as_float(new)
    if old_value in (None, 0.0) or new_value is None:
        return None
    return (new_value - old_value) / old_value * 100.0


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric_values(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        number = as_float(value)
        if number is not None:
            result.append(number)
    return result


def describe_values(values: Iterable[Any]) -> dict[str, float | int | None]:
    nums = numeric_values(values)
    if not nums:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "min": None,
            "max": None,
            "stdev": None,
        }
    return {
        "n": len(nums),
        "mean": statistics.fmean(nums),
        "median": statistics.median(nums),
        "p90": percentile(nums, 90),
        "p95": percentile(nums, 95),
        "p99": percentile(nums, 99),
        "min": min(nums),
        "max": max(nums),
        "stdev": statistics.stdev(nums) if len(nums) > 1 else 0.0,
    }


def flatten_itls(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    flat: list[float] = []
    for item in values:
        if isinstance(item, list):
            flat.extend(numeric_values(item))
        else:
            number = as_float(item)
            if number is not None:
                flat.append(number)
    return flat


def per_request_itl_means(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    means: list[float] = []
    for item in values:
        nums = numeric_values(item) if isinstance(item, list) else []
        if nums:
            means.append(statistics.fmean(nums))
    return means


def clip_text(value: Any, limit: int = TEXT_LIMIT) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", "\\n").replace("\r", "\\r")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def variant_label(report: dict[str, Any]) -> str:
    variant = report.get("variant", {})
    if isinstance(variant, dict):
        return str(variant.get("label") or variant.get("kv_cache_dtype") or "unknown")
    return "unknown"


def variant_dtype(report: dict[str, Any]) -> str:
    variant = report.get("variant", {})
    if isinstance(variant, dict):
        return str(variant.get("kv_cache_dtype") or "")
    return ""


def point_rate(point: dict[str, Any]) -> str:
    return str(point.get("request_rate", "unknown"))


def point_raw(point: dict[str, Any]) -> dict[str, Any]:
    raw = point.get("raw_result", {})
    return raw if isinstance(raw, dict) else {}


def point_metrics(point: dict[str, Any]) -> dict[str, Any]:
    metrics = point.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def metric_value(point: dict[str, Any], key: str) -> Any:
    metrics = point_metrics(point)
    raw = point_raw(point)
    if key in metrics:
        return metrics[key]
    return raw.get(key)


def build_point_index(reports: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for report in reports:
        label = variant_label(report)
        for point in report.get("bench_points", []):
            if isinstance(point, dict):
                index[(label, point_rate(point))] = point
    return index


def reference_label(reports: list[dict[str, Any]], requested: str | None = None) -> str:
    labels = [variant_label(report) for report in reports]
    if requested and requested in labels:
        return requested
    if DEFAULT_REFERENCE_LABEL in labels:
        return DEFAULT_REFERENCE_LABEL
    return labels[0] if labels else DEFAULT_REFERENCE_LABEL


def collect_rates(reports: list[dict[str, Any]]) -> list[str]:
    rates: list[str] = []
    for report in reports:
        for point in report.get("bench_points", []):
            if isinstance(point, dict):
                rate = point_rate(point)
                if rate not in rates:
                    rates.append(rate)
    return rates


def sort_rate_key(rate: str) -> tuple[int, float | str]:
    if rate == "inf":
        return (1, rate)
    number = as_float(rate)
    if number is None:
        return (2, rate)
    return (0, number)


def report_artifacts(run_dir: Path) -> list[Path]:
    return sorted(path for path in run_dir.rglob("*") if path.is_file())


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def svg_text(value: Any) -> str:
    return html.escape(str(value), quote=False)


def write_grouped_bar_svg(
    path: Path,
    *,
    title: str,
    y_label: str,
    groups: list[str],
    series: dict[str, list[float | None]],
) -> bool:
    values = [value for rows in series.values() for value in rows if value is not None]
    if not groups or not series or not values:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    width = 980
    height = 420
    margin_left = 86
    margin_right = 24
    margin_top = 66
    margin_bottom = 72
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(values)
    if max_value <= 0:
        max_value = 1.0
    max_value *= 1.12
    colors = ["#79863c", "#a7b368", "#4e5b2f", "#c8d490", "#6f7b4f", "#dce6b8"]
    series_names = list(series)
    group_width = plot_width / max(1, len(groups))
    bar_gap = 8
    bar_width = max(8.0, (group_width - 28) / max(1, len(series_names)) - bar_gap)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">',
        "<style>",
        "text{font-family:Inter,ui-sans-serif,system-ui,sans-serif;fill:#eceef8} .muted{fill:#a1a6bc} .axis{stroke:#2a2f43;stroke-width:1} .grid{stroke:#2a2f43;stroke-width:1;opacity:.55}",
        "</style>",
        '<rect width="100%" height="100%" rx="14" fill="#14161e"/>',
        f'<text x="{margin_left}" y="34" font-size="20" font-weight="700">{svg_text(title)}</text>',
        f'<text x="{margin_left}" y="54" font-size="12" class="muted">{svg_text(y_label)}</text>',
    ]
    for tick in range(5):
        value = max_value * tick / 4
        y = margin_top + plot_height - (value / max_value) * plot_height
        lines.append(f'<line x1="{margin_left}" x2="{width - margin_right}" y1="{y:.2f}" y2="{y:.2f}" class="grid"/>')
        lines.append(f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" class="muted">{svg_text(fmt(value, 1))}</text>')
    lines.append(f'<line x1="{margin_left}" x2="{margin_left}" y1="{margin_top}" y2="{margin_top + plot_height}" class="axis"/>')
    lines.append(f'<line x1="{margin_left}" x2="{width - margin_right}" y1="{margin_top + plot_height}" y2="{margin_top + plot_height}" class="axis"/>')

    for group_index, group in enumerate(groups):
        group_x = margin_left + group_index * group_width
        lines.append(
            f'<text x="{group_x + group_width / 2:.2f}" y="{height - 36}" text-anchor="middle" font-size="12" class="muted">{svg_text(group)}</text>'
        )
        for series_index, name in enumerate(series_names):
            values_for_name = series[name]
            value = values_for_name[group_index] if group_index < len(values_for_name) else None
            if value is None:
                continue
            bar_height = (value / max_value) * plot_height
            x = group_x + 16 + series_index * (bar_width + bar_gap)
            y = margin_top + plot_height - bar_height
            color = colors[series_index % len(colors)]
            lines.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" rx="4" fill="{color}"/>')
            if bar_height > 26:
                lines.append(
                    f'<text x="{x + bar_width / 2:.2f}" y="{y + 17:.2f}" text-anchor="middle" font-size="10" fill="#101114">{svg_text(fmt(value, 1))}</text>'
                )
    legend_x = margin_left
    legend_y = height - 16
    for series_index, name in enumerate(series_names):
        x = legend_x + series_index * 130
        lines.append(f'<rect x="{x}" y="{legend_y - 10}" width="12" height="12" rx="2" fill="{colors[series_index % len(colors)]}"/>')
        lines.append(f'<text x="{x + 18}" y="{legend_y}" font-size="12" class="muted">{svg_text(name)}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")
    return True


def plot_metric(
    plots_dir: Path,
    name: str,
    *,
    title: str,
    y_label: str,
    reports: list[dict[str, Any]],
    rates: list[str],
    metric: str,
) -> Path | None:
    series: dict[str, list[float | None]] = {}
    point_index = build_point_index(reports)
    for report in reports:
        label = variant_label(report)
        series[label] = [as_float(metric_value(point_index.get((label, rate), {}), metric)) for rate in rates]
    path = plots_dir / f"{name}.svg"
    return path if write_grouped_bar_svg(path, title=title, y_label=y_label, groups=rates, series=series) else None


def plot_variant_metric(
    plots_dir: Path,
    name: str,
    *,
    title: str,
    y_label: str,
    reports: list[dict[str, Any]],
    metric_from_report: callable,
) -> Path | None:
    groups = [variant_label(report) for report in reports]
    values = [metric_from_report(report) for report in reports]
    path = plots_dir / f"{name}.svg"
    return path if write_grouped_bar_svg(path, title=title, y_label=y_label, groups=groups, series={"value": values}) else None


def make_plots(run_dir: Path, reports: list[dict[str, Any]], rates: list[str], plots_dir: Path) -> list[Path]:
    created: list[Path] = []
    for name, title, y_label, metric in [
        ("ttft_ms", "TTFT by request rate", "milliseconds", "median_ttft_ms"),
        ("tpot_ms", "TPOT by request rate", "milliseconds", "median_tpot_ms"),
        ("e2el_ms", "End-to-end latency by request rate", "milliseconds", "median_e2el_ms"),
        ("output_tps", "Output throughput by request rate", "tokens/sec", "output_throughput"),
        ("request_tps", "Request throughput by request rate", "requests/sec", "request_throughput"),
    ]:
        path = plot_metric(plots_dir, name, title=title, y_label=y_label, reports=reports, rates=rates, metric=metric)
        if path is not None:
            created.append(path)
    cache_tokens = plot_variant_metric(
        plots_dir,
        "kv_cache_tokens",
        title="GPU KV cache token capacity",
        y_label="tokens",
        reports=reports,
        metric_from_report=lambda report: as_float(report.get("server_log_kv", {}).get("gpu_kv_cache_tokens")),
    )
    if cache_tokens is not None:
        created.append(cache_tokens)
    peak_gpu = plot_variant_metric(
        plots_dir,
        "peak_gpu_gib",
        title="Peak sampled GPU memory",
        y_label="GiB",
        reports=reports,
        metric_from_report=lambda report: ratio(report.get("resource_summary", {}).get("gpu_mem_used_bytes_max"), BYTES_IN_GIB),
    )
    if peak_gpu is not None:
        created.append(peak_gpu)
    kv_memory = plot_variant_metric(
        plots_dir,
        "kv_memory_gib",
        title="KV cache memory from server log",
        y_label="GiB",
        reports=reports,
        metric_from_report=lambda report: ratio(report.get("server_log_kv", {}).get("kv_cache_memory_bytes_from_log"), BYTES_IN_GIB),
    )
    if kv_memory is not None:
        created.append(kv_memory)
    return created


def distribution_rows(reports: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for report in reports:
        label = variant_label(report)
        for point in report.get("bench_points", []):
            if not isinstance(point, dict):
                continue
            raw = point_raw(point)
            ttft = describe_values([value * 1000.0 for value in numeric_values(raw.get("ttfts", []))])
            flat_itl = describe_values([value * 1000.0 for value in flatten_itls(raw.get("itls", []))])
            itl_request = describe_values([value * 1000.0 for value in per_request_itl_means(raw.get("itls", []))])
            input_lens = describe_values(raw.get("input_lens", []))
            output_lens = describe_values(raw.get("output_lens", []))
            rows.append(
                [
                    label,
                    point_rate(point),
                    fmt_int(ttft["n"]),
                    fmt(ttft["median"]),
                    fmt(ttft["p90"]),
                    fmt(ttft["p99"]),
                    fmt(flat_itl["median"]),
                    fmt(flat_itl["p90"]),
                    fmt(flat_itl["p99"]),
                    fmt(itl_request["median"]),
                    fmt(input_lens["median"], 1),
                    fmt(output_lens["median"], 1),
                ]
            )
    return rows


def comparison_rows(reports: list[dict[str, Any]], ref_label: str, rates: list[str]) -> list[list[str]]:
    point_index = build_point_index(reports)
    labels = [variant_label(report) for report in reports if variant_label(report) != ref_label]
    rows: list[list[str]] = []
    for label in labels:
        for rate in rates:
            ref = point_index.get((ref_label, rate), {})
            item = point_index.get((label, rate), {})
            rows.append(
                [
                    label,
                    rate,
                    fmt(ratio(metric_value(ref, "median_ttft_ms"), metric_value(item, "median_ttft_ms")), 4),
                    fmt(ratio(metric_value(ref, "median_tpot_ms"), metric_value(item, "median_tpot_ms")), 4),
                    fmt(ratio(metric_value(ref, "median_e2el_ms"), metric_value(item, "median_e2el_ms")), 4),
                    fmt(ratio(metric_value(item, "output_throughput"), metric_value(ref, "output_throughput")), 4),
                    fmt(ratio(metric_value(item, "request_throughput"), metric_value(ref, "request_throughput")), 4),
                    fmt((as_float(metric_value(item, "median_ttft_ms")) or 0) - (as_float(metric_value(ref, "median_ttft_ms")) or 0)),
                    fmt((as_float(metric_value(item, "median_tpot_ms")) or 0) - (as_float(metric_value(ref, "median_tpot_ms")) or 0)),
                ]
            )
    return rows


def cache_rows(reports: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for report in reports:
        log_kv = report.get("server_log_kv", {})
        resource = report.get("resource_summary", {})
        rows.append(
            [
                variant_label(report),
                fmt_int(log_kv.get("gpu_kv_cache_tokens")),
                fmt_int(log_kv.get("max_concurrency_tokens_per_request")),
                fmt(log_kv.get("max_concurrency_from_log"), 2),
                fmt_bytes(log_kv.get("kv_cache_memory_bytes_from_log")),
                fmt_bytes(resource.get("gpu_mem_used_bytes_max")),
                fmt_bytes(resource.get("process_rss_bytes_max")),
                fmt_bytes(resource.get("system_ram_used_bytes_max")),
            ]
        )
    return rows


def quality_rows(reports: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for report in reports:
        label = variant_label(report)
        quality_items = report.get("quality", [])
        if not quality_items:
            rows.append([label, "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", "n/a", ""])
            continue
        for item in quality_items:
            if not isinstance(item, dict):
                continue
            rows.append(
                [
                    label,
                    str(item.get("name", "n/a")),
                    str(item.get("prompt_tokens", "n/a")),
                    str(item.get("expected", "")),
                    str(item.get("contains_expected", "n/a")),
                    fmt(item.get("ttft_ms")),
                    fmt(item.get("tpot_ms")),
                    fmt(item.get("total_latency_ms")),
                    "`" + clip_text(item.get("output"), 160).replace("|", "\\|") + "`",
                ]
            )
    return rows


def resource_rows(reports: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for report in reports:
        resource = report.get("resource_summary", {})
        rows.append(
            [
                variant_label(report),
                fmt_int(resource.get("samples")),
                fmt(resource.get("process_cpu_percent_mean")),
                fmt(resource.get("process_cpu_percent_max")),
                fmt(resource.get("gpu_util_percent_mean")),
                fmt(resource.get("gpu_util_percent_max")),
                fmt(resource.get("gpu_mem_util_percent_max")),
                fmt_bytes(resource.get("gpu_mem_used_bytes_max")),
                fmt(resource.get("system_ram_percent_max")),
            ]
        )
    return rows


def prometheus_rows(reports: list[dict[str, Any]]) -> list[list[str]]:
    wanted = [
        "vllm:kv_cache_usage_perc",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:prompt_tokens_cached_total",
        "vllm:request_prefill_kv_computed_tokens_count",
        "vllm:request_prefill_kv_computed_tokens_sum",
    ]
    rows: list[list[str]] = []
    for report in reports:
        before = report.get("prometheus_before", {})
        after = report.get("prometheus_after", {})
        keys = [key for key in wanted if key in before or key in after]
        for key in keys:
            before_value = as_float(before.get(key)) or 0.0
            after_value = as_float(after.get(key)) or 0.0
            rows.append([variant_label(report), key, fmt(before_value), fmt(after_value), fmt(after_value - before_value)])
    return rows


def artifact_rows(run_dir: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    for path in report_artifacts(run_dir):
        rows.append([relative(path, run_dir), fmt_int(path.stat().st_size)])
    return rows


def generated_text_rows(reports: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for report in reports:
        label = variant_label(report)
        for point in report.get("bench_points", []):
            if not isinstance(point, dict):
                continue
            raw = point_raw(point)
            texts = raw.get("generated_texts", [])
            errors = raw.get("errors", [])
            if not isinstance(texts, list):
                texts = []
            if not isinstance(errors, list):
                errors = []
            nonempty_errors = [str(err) for err in errors if str(err)]
            rows.append(
                [
                    label,
                    point_rate(point),
                    fmt_int(len(texts)),
                    fmt_int(len(nonempty_errors)),
                    "`" + clip_text(texts[0] if texts else "", 180).replace("|", "\\|") + "`",
                    "`" + clip_text(nonempty_errors[0] if nonempty_errors else "", 120).replace("|", "\\|") + "`",
                ]
            )
    return rows


def stability_rows(reports: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for report in reports:
        label = variant_label(report)
        for point in report.get("bench_points", []):
            if not isinstance(point, dict):
                continue
            raw = point_raw(point)
            errors = raw.get("errors", [])
            nonempty_errors = [str(err) for err in errors if str(err)] if isinstance(errors, list) else []
            rows.append(
                [
                    label,
                    point_rate(point),
                    fmt_int(raw.get("num_prompts")),
                    fmt_int(raw.get("com" + "pleted")),
                    fmt_int(raw.get("failed")),
                    fmt_int(len(nonempty_errors)),
                    clip_text("; ".join(nonempty_errors), 160) or "n/a",
                ]
            )
    return rows


def load_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    report = read_json(run_dir / "report.json")
    manifest = maybe_read_json(run_dir / "manifest.json") or report.get("manifest", {})
    reports = report.get("reports", [])
    if not isinstance(reports, list):
        reports = []
    return manifest, report, [item for item in reports if isinstance(item, dict)]


def build_markdown(
    *,
    run_dir: Path,
    output_path: Path,
    manifest: dict[str, Any],
    report: dict[str, Any],
    reports: list[dict[str, Any]],
    plots: list[Path],
    ref_label: str,
) -> str:
    run_id = str(report.get("run_id") or manifest.get("run_id") or run_dir.name)
    config = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
    environment = manifest.get("environment", {}) if isinstance(manifest.get("environment"), dict) else {}
    packages = environment.get("packages", {}) if isinstance(environment.get("packages"), dict) else {}
    gpu = environment.get("gpu", {}) if isinstance(environment.get("gpu"), dict) else {}
    devices = gpu.get("devices", []) if isinstance(gpu.get("devices"), list) else []
    first_device = devices[0] if devices and isinstance(devices[0], dict) else {}
    rates = sorted(collect_rates(reports), key=sort_rate_key)

    lines: list[str] = [
        "# vLLM FP8 vs TurboQuant Run Report",
        "",
        f"Source run directory: `{run_dir}`",
        "",
        "This report is generated from saved run artifacts. It does not run inference and it does not fill missing values with assumptions.",
        "",
        "## Run Identity",
        "",
        markdown_table(
            ["Field", "Value"],
            [
                ["Run ID", run_id],
                ["Model", config.get("model", "n/a")],
                ["Context budget", fmt_int(config.get("context_len"))],
                ["Benchmark input tokens", fmt_int(config.get("input_len"))],
                ["Output tokens requested", fmt_int(config.get("output_len"))],
                ["Prompts per request-rate point", fmt_int(config.get("num_prompts"))],
                ["Request rates", ", ".join(str(item) for item in config.get("request_rates", []))],
                ["Tensor parallel size", fmt_int(config.get("tensor_parallel_size"))],
                ["GPU memory utilization", fmt(config.get("gpu_memory_utilization"), 2)],
                ["Model dtype", config.get("dtype", "n/a")],
                ["Seed", fmt_int(config.get("seed"))],
                ["Host", config.get("host", "n/a")],
                ["Base port", fmt_int(config.get("base_port"))],
            ],
        ),
        "",
        "## Runtime Environment",
        "",
        markdown_table(
            ["Component", "Value"],
            [
                ["Platform", environment.get("platform", "n/a")],
                ["Python executable", environment.get("executable", "n/a")],
                ["Python version", environment.get("python", "n/a")],
                ["Torch", packages.get("torch", "n/a")],
                ["Triton", packages.get("triton", "n/a")],
                ["Transformers", packages.get("transformers", "n/a")],
                ["vLLM", packages.get("vllm", "n/a")],
                ["GPU", first_device.get("name", "n/a")],
                ["GPU memory total", f"{first_device.get('memory_total_mib', 'n/a')} MiB"],
                ["Driver", first_device.get("driver", "n/a")],
                ["PyTorch CUDA available", str(gpu.get("available", "n/a"))],
                ["PyTorch CUDA devices", fmt_int(len(devices))],
            ],
        ),
        "",
        "## Variants",
        "",
        markdown_table(
            ["Label", "KV cache dtype", "Startup seconds", "Server log"],
            [
                [
                    variant_label(item),
                    variant_dtype(item),
                    fmt(item.get("startup_seconds")),
                    item.get("server_log", "n/a"),
                ]
                for item in reports
            ],
        ),
    ]

    cli_support = manifest.get("vllm_cli_support", {}) if isinstance(manifest.get("vllm_cli_support"), dict) else {}
    dtype_found = cli_support.get("kv_cache_dtype_found", {})
    if isinstance(dtype_found, dict) and dtype_found:
        lines += ["", "vLLM CLI dtype support recorded in the run:", ""]
        lines.append(markdown_table(["KV cache dtype", "Found"], [[key, value] for key, value in dtype_found.items()]))

    lines += ["", "## Server Commands", ""]
    for item in reports:
        command = item.get("server_command", [])
        lines += [f"### {variant_label(item)}", "", "```text", " ".join(str(part) for part in command), "```", ""]

    plot_rel = [relative(path, output_path.parent) for path in plots]
    if plot_rel:
        lines += ["## Plots", ""]
        for path in plot_rel:
            title = Path(path).stem.replace("_", " ")
            lines.append(f"![{title}]({path})")
            lines.append("")

    lines += [
        "## Cache Capacity And Memory",
        "",
        markdown_table(
            [
                "Variant",
                "GPU KV cache tokens",
                "Tokens/request in log",
                "Max concurrency from log",
                "KV cache memory from log",
                "Peak GPU memory",
                "Peak process RSS",
                "Peak system RAM used",
            ],
            cache_rows(reports),
        ),
        "",
    ]

    if ref_label:
        ref_report = next((item for item in reports if variant_label(item) == ref_label), None)
        if ref_report is not None:
            rows: list[list[str]] = []
            ref_log = ref_report.get("server_log_kv", {})
            ref_resource = ref_report.get("resource_summary", {})
            for item in reports:
                label = variant_label(item)
                if label == ref_label:
                    continue
                log_kv = item.get("server_log_kv", {})
                resource = item.get("resource_summary", {})
                rows.append(
                    [
                        label,
                        fmt(ratio(log_kv.get("gpu_kv_cache_tokens"), ref_log.get("gpu_kv_cache_tokens")), 4),
                        fmt(pct_change(log_kv.get("gpu_kv_cache_tokens"), ref_log.get("gpu_kv_cache_tokens")), 2),
                        fmt(ratio(ref_log.get("kv_cache_memory_bytes_from_log"), log_kv.get("kv_cache_memory_bytes_from_log")), 4),
                        fmt(pct_change(log_kv.get("kv_cache_memory_bytes_from_log"), ref_log.get("kv_cache_memory_bytes_from_log")), 2),
                        fmt(ratio(ref_resource.get("gpu_mem_used_bytes_max"), resource.get("gpu_mem_used_bytes_max")), 4),
                        fmt(pct_change(resource.get("gpu_mem_used_bytes_max"), ref_resource.get("gpu_mem_used_bytes_max")), 2),
                    ]
                )
            if rows:
                lines += [
                    f"Derived memory and capacity values use `{ref_label}` as the reference.",
                    "",
                    markdown_table(
                        [
                            "Variant",
                            "KV token capacity ratio",
                            "KV token capacity change %",
                            "Log KV memory reference/variant",
                            "Log KV memory change %",
                            "Peak GPU reference/variant",
                            "Peak GPU change %",
                        ],
                        rows,
                    ),
                    "",
                ]

    lines += [
        "## Latency",
        "",
        markdown_table(
            [
                "Variant",
                "Request rate",
                "mean TTFT ms",
                "median TTFT ms",
                "p90 TTFT ms",
                "p99 TTFT ms",
                "mean TPOT ms",
                "median TPOT ms",
                "p90 TPOT ms",
                "p99 TPOT ms",
                "mean ITL ms",
                "median ITL ms",
                "p90 ITL ms",
                "p99 ITL ms",
                "mean E2EL ms",
                "median E2EL ms",
                "p90 E2EL ms",
                "p99 E2EL ms",
            ],
            [
                [
                    variant_label(item),
                    point_rate(point),
                    fmt(metric_value(point, "mean_ttft_ms")),
                    fmt(metric_value(point, "median_ttft_ms")),
                    fmt(metric_value(point, "p90_ttft_ms")),
                    fmt(metric_value(point, "p99_ttft_ms")),
                    fmt(metric_value(point, "mean_tpot_ms")),
                    fmt(metric_value(point, "median_tpot_ms")),
                    fmt(metric_value(point, "p90_tpot_ms")),
                    fmt(metric_value(point, "p99_tpot_ms")),
                    fmt(metric_value(point, "mean_itl_ms")),
                    fmt(metric_value(point, "median_itl_ms")),
                    fmt(metric_value(point, "p90_itl_ms")),
                    fmt(metric_value(point, "p99_itl_ms")),
                    fmt(metric_value(point, "mean_e2el_ms")),
                    fmt(metric_value(point, "median_e2el_ms")),
                    fmt(metric_value(point, "p90_e2el_ms")),
                    fmt(metric_value(point, "p99_e2el_ms")),
                ]
                for item in reports
                for point in item.get("bench_points", [])
                if isinstance(point, dict)
            ],
        ),
        "",
        "## Raw Distribution Checks",
        "",
        "This table is computed from raw arrays such as `ttfts`, `itls`, `input_lens`, and `output_lens` when those arrays are present.",
        "",
        markdown_table(
            [
                "Variant",
                "Request rate",
                "TTFT samples",
                "TTFT p50",
                "TTFT p90",
                "TTFT p99",
                "ITL p50",
                "ITL p90",
                "ITL p99",
                "Per-request ITL mean p50",
                "Input p50",
                "Output p50",
            ],
            distribution_rows(reports),
        ),
        "",
        "## Throughput And Tokens",
        "",
        markdown_table(
            [
                "Variant",
                "Request rate",
                "Duration",
                "Done",
                "Failed",
                "Input tokens",
                "Output tokens",
                "Req/s",
                "Output tok/s",
                "Total tok/s",
                "Max concurrent requests",
            ],
            [
                [
                    variant_label(item),
                    point_rate(point),
                    fmt(metric_value(point, "duration")),
                    fmt_int(metric_value(point, "com" + "pleted")),
                    fmt_int(metric_value(point, "failed")),
                    fmt_int(metric_value(point, "total_input_tokens")),
                    fmt_int(metric_value(point, "total_output_tokens")),
                    fmt(metric_value(point, "request_throughput")),
                    fmt(metric_value(point, "output_throughput")),
                    fmt(metric_value(point, "total_token_throughput")),
                    fmt_int(metric_value(point, "max_concurrent_requests")),
                ]
                for item in reports
                for point in item.get("bench_points", [])
                if isinstance(point, dict)
            ],
        ),
        "",
        f"## Variant Ratios Versus `{ref_label}`",
        "",
        "Ratios above 1.0 favor the numerator direction named in the column. Latency ratios use reference divided by variant; throughput ratios use variant divided by reference.",
        "",
        markdown_table(
            [
                "Variant",
                "Request rate",
                "Reference TTFT / variant TTFT",
                "Reference TPOT / variant TPOT",
                "Reference E2EL / variant E2EL",
                "Variant output tok/s / reference",
                "Variant req/s / reference",
                "Variant - reference TTFT ms",
                "Variant - reference TPOT ms",
            ],
            comparison_rows(reports, ref_label, rates),
        ),
        "",
        "## Resource Sampling",
        "",
        markdown_table(
            [
                "Variant",
                "Samples",
                "Mean process CPU %",
                "Max process CPU %",
                "Mean GPU util %",
                "Max GPU util %",
                "Max GPU mem util %",
                "Max GPU used",
                "Max system RAM %",
            ],
            resource_rows(reports),
        ),
        "",
        "## Prometheus Cache Metrics",
        "",
        markdown_table(["Variant", "Metric", "Before", "After", "Delta"], prometheus_rows(reports)),
        "",
        "## Quality Probes",
        "",
        markdown_table(
            ["Variant", "Probe", "Prompt tokens", "Expected", "Contains expected", "TTFT ms", "TPOT ms", "Total latency ms", "Output excerpt"],
            quality_rows(reports),
        ),
        "",
        "## Generated Text Excerpts",
        "",
        "Full generated text is kept in the benchmark JSON files. This table lists the first saved text per point and non-empty error count.",
        "",
        markdown_table(["Variant", "Request rate", "Saved texts", "Non-empty errors", "First text excerpt", "First error excerpt"], generated_text_rows(reports)),
        "",
        "## Stability",
        "",
        markdown_table(["Variant", "Request rate", "Prompts", "Done", "Failed", "Non-empty errors", "Error excerpt"], stability_rows(reports)),
        "",
        "## Limits Bound To This Run",
        "",
        f"- The benchmark contains `{fmt_int(config.get('num_prompts'))}` prompt(s) per request-rate point.",
        "- Percentiles are only as meaningful as the number of saved per-request samples.",
        "- Request-rate points are not the same as sustained high concurrency unless the raw benchmark records multiple simultaneous requests.",
        "- Memory fields come from different sources: server logs, resource samples, and Prometheus metrics. They should be interpreted separately.",
        "- This report makes no claim outside the saved run artifacts.",
        "",
        "## Artifact Index",
        "",
        markdown_table(["Path", "Bytes"], artifact_rows(run_dir)),
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Markdown report from a vLLM FP8/TurboQuant run directory.")
    parser.add_argument("run_dir", type=Path, help="Run directory containing manifest.json and report.json.")
    parser.add_argument("--output", type=Path, default=None, help="Markdown output path. Defaults to run_dir/run_report.md.")
    parser.add_argument("--plots-dir", type=Path, default=None, help="Directory for SVG plots. Defaults to run_dir/plots.")
    parser.add_argument("--reference", default=None, help="Reference variant label. Defaults to fp8 when present.")
    parser.add_argument("--no-plots", action="store_true", help="Skip SVG plot generation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_path = (args.output or (run_dir / "run_report.md")).resolve()
    plots_dir = (args.plots_dir or (run_dir / "plots")).resolve()

    manifest, report, reports = load_run(run_dir)
    if not reports:
        raise SystemExit(f"no variant reports found in {run_dir / 'report.json'}")
    rates = sorted(collect_rates(reports), key=sort_rate_key)
    plots = [] if args.no_plots else make_plots(run_dir, reports, rates, plots_dir)
    ref_label = reference_label(reports, args.reference)
    markdown = build_markdown(
        run_dir=run_dir,
        output_path=output_path,
        manifest=manifest,
        report=report,
        reports=reports,
        plots=plots,
        ref_label=ref_label,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
