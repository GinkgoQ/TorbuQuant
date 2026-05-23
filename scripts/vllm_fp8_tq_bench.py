"""Run vLLM FP8 versus TurboQuant KV-cache measurements.

The runner starts one vLLM server per KV-cache dtype, runs the same benchmark
workload against each server, captures server logs, Prometheus metrics, process
resource samples, benchmark JSON, and optional retrieval prompts.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import logging
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


KV_DTYPES = {
    "fp8": "fp8",
    "tq4": "turboquant_4bit_nc",
    "tq_k8v4": "turboquant_k8v4",
    "tq_k3v4": "turboquant_k3v4_nc",
    "tq3": "turboquant_3bit_nc",
}
VLLM_COMMAND = (sys.executable, "-m", "vllm.entrypoints.cli.main")
LOGGER = logging.getLogger("vllm-fp8-tq")


@dataclass(frozen=True)
class Variant:
    label: str
    kv_cache_dtype: str

    def to_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class RunConfig:
    model: str
    context_len: int
    input_len: int
    output_len: int
    num_prompts: int
    request_rates: tuple[str, ...]
    max_concurrency: int | None
    tensor_parallel_size: int
    gpu_memory_utilization: float
    dtype: str | None
    trust_remote_code: bool
    seed: int
    host: str
    base_port: int
    startup_timeout_s: float
    skip_cli_check: bool = False
    serve_extra_args: tuple[str, ...] = tuple()
    bench_extra_args: tuple[str, ...] = tuple()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class ProgressBar:
    def __init__(self, *, total: int, label: str):
        self.total = max(1, total)
        self.label = label
        self.current = 0

    def advance(self, detail: str) -> None:
        self.current += 1
        filled = int(24 * self.current / self.total)
        bar = "#" * filled + "." * (24 - filled)
        LOGGER.info("[%s] %s %d/%d - %s", bar, self.label, self.current, self.total, detail)


@dataclass
class ResourceSample:
    time_s: float
    process_cpu_percent: float | None = None
    process_rss_bytes: int | None = None
    system_ram_used_bytes: int | None = None
    system_ram_percent: float | None = None
    gpu_util_percent: int | None = None
    gpu_mem_util_percent: int | None = None
    gpu_mem_used_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class ResourceSummary:
    samples: int
    process_cpu_percent_mean: float | None
    process_cpu_percent_max: float | None
    process_rss_bytes_max: int | None
    system_ram_used_bytes_max: int | None
    system_ram_percent_max: float | None
    gpu_util_percent_mean: float | None
    gpu_util_percent_max: int | None
    gpu_mem_util_percent_max: int | None
    gpu_mem_used_bytes_max: int | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class BenchPoint:
    variant: str
    request_rate: str
    result_path: str | None
    raw_result: dict[str, Any]
    metrics: dict[str, Any]
    resource_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class VariantReport:
    variant: Variant
    out_dir: str
    server_command: list[str]
    server_log: str
    startup_seconds: float
    server_log_kv: dict[str, Any]
    prometheus_before: dict[str, float]
    prometheus_after: dict[str, float]
    resource_summary: dict[str, Any]
    bench_points: list[BenchPoint] = field(default_factory=list)
    quality: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["variant"] = self.variant.to_dict()
        data["bench_points"] = [point.to_dict() for point in self.bench_points]
        return data


class ResourceSampler:
    def __init__(self, *, pid: int | None = None, device_index: int = 0, interval_s: float = 0.1):
        self.pid = pid
        self.device_index = device_index
        self.interval_s = interval_s
        self.samples: list[ResourceSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._psutil = _load_psutil()
        self._nvml = _load_nvml()
        self._process = None
        if self._psutil is not None and pid is not None:
            try:
                self._process = self._psutil.Process(pid)
            except Exception:
                self._process = None
        self._handle = None
        if self._nvml is not None:
            try:
                self._handle = self._nvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception:
                self._handle = None

    def __enter__(self) -> "ResourceSampler":
        if self._process is not None:
            with contextlib.suppress(Exception):
                self._process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return False

    def _loop(self) -> None:
        while not self._stop.is_set():
            row = ResourceSample(time_s=time.time())
            if self._process is not None and self._psutil is not None:
                try:
                    with self._process.oneshot():
                        mem = self._process.memory_info()
                        row.process_cpu_percent = float(self._process.cpu_percent(interval=None))
                        row.process_rss_bytes = int(mem.rss)
                    vm = self._psutil.virtual_memory()
                    row.system_ram_used_bytes = int(vm.used)
                    row.system_ram_percent = float(vm.percent)
                except Exception:
                    pass
            if self._handle is not None and self._nvml is not None:
                try:
                    util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                    mem_info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                    row.gpu_util_percent = int(util.gpu)
                    row.gpu_mem_util_percent = int(util.memory)
                    row.gpu_mem_used_bytes = int(mem_info.used)
                except Exception:
                    pass
            self.samples.append(row)
            self._stop.wait(self.interval_s)

    def summary(self) -> ResourceSummary:
        rows = [sample.to_dict() for sample in self.samples]

        def values(name: str) -> list[Any]:
            return [row[name] for row in rows if row.get(name) is not None]

        def mean(name: str) -> float | None:
            vals = [float(x) for x in values(name)]
            return sum(vals) / len(vals) if vals else None

        def max_value(name: str) -> Any:
            vals = values(name)
            return max(vals) if vals else None

        return ResourceSummary(
            samples=len(rows),
            process_cpu_percent_mean=mean("process_cpu_percent"),
            process_cpu_percent_max=max_value("process_cpu_percent"),
            process_rss_bytes_max=max_value("process_rss_bytes"),
            system_ram_used_bytes_max=max_value("system_ram_used_bytes"),
            system_ram_percent_max=max_value("system_ram_percent"),
            gpu_util_percent_mean=mean("gpu_util_percent"),
            gpu_util_percent_max=max_value("gpu_util_percent"),
            gpu_mem_util_percent_max=max_value("gpu_mem_util_percent"),
            gpu_mem_used_bytes_max=max_value("gpu_mem_used_bytes"),
        )


def _load_psutil():
    try:
        import psutil
    except Exception:
        return None
    return psutil


def _load_nvml():
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None


def http_json(url: str, *, timeout_s: float = 10.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8"))


def http_text(url: str, *, timeout_s: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return response.read().decode("utf-8", errors="replace")


def tail_text(path: Path, *, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def wait_for_server(
    base_url: str,
    *,
    timeout_s: float,
    process: subprocess.Popen[str],
    log_path: Path,
) -> float:
    start = time.perf_counter()
    last_error = ""
    LOGGER.info("waiting for server readiness at %s", base_url)
    while time.perf_counter() - start < timeout_s:
        if process.poll() is not None:
            log_tail = tail_text(log_path)
            detail = f"server exited during startup with code {process.returncode}"
            if log_tail:
                detail += f"\n\nLast server log lines:\n{log_tail}"
            raise RuntimeError(detail)
        try:
            http_json(f"{base_url}/v1/models", timeout_s=5.0)
            LOGGER.info("server is ready after %.2fs", time.perf_counter() - start)
            return time.perf_counter() - start
        except Exception as exc:
            last_error = str(exc)
            time.sleep(1.0)
    raise TimeoutError(f"server did not become ready in {timeout_s:.1f}s: {last_error}")


@contextlib.contextmanager
def start_server(
    *,
    config: RunConfig,
    variant: Variant,
    port: int,
    out_dir: Path,
) -> Iterator[tuple[subprocess.Popen[str], list[str], float]]:
    log_path = out_dir / "server.log"
    command = [
        *VLLM_COMMAND,
        "serve",
        config.model,
        "--host",
        config.host,
        "--port",
        str(port),
        "--kv-cache-dtype",
        variant.kv_cache_dtype,
        "--max-model-len",
        str(config.context_len),
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
    ]
    if config.dtype:
        command.extend(["--dtype", config.dtype])
    if config.trust_remote_code:
        command.append("--trust-remote-code")
    command.extend(config.serve_extra_args)

    LOGGER.info("starting vLLM server for %s on port %d", variant.label, port)
    LOGGER.info("server log: %s", log_path)
    LOGGER.debug("server command: %s", " ".join(command))
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT),
            start_new_session=True,
        )
    try:
        startup_s = wait_for_server(
            f"http://{config.host}:{port}",
            timeout_s=config.startup_timeout_s,
            process=process,
            log_path=log_path,
        )
        yield process, command, startup_s
    finally:
        terminate_process_group(process)


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def run_bench(
    *,
    config: RunConfig,
    variant: Variant,
    port: int,
    request_rate: str,
    out_dir: Path,
) -> tuple[dict[str, Any], Path | None, str]:
    result_name = f"{variant.label}_rate_{_slug(request_rate)}.json"
    command = [
        *VLLM_COMMAND,
        "bench",
        "serve",
        "--backend",
        "openai",
        "--host",
        config.host,
        "--port",
        str(port),
        "--endpoint",
        "/v1/completions",
        "--model",
        config.model,
        "--dataset-name",
        "random",
        "--input-len",
        str(config.input_len),
        "--output-len",
        str(config.output_len),
        "--num-prompts",
        str(config.num_prompts),
        "--request-rate",
        request_rate,
        "--seed",
        str(config.seed),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(out_dir),
        "--result-filename",
        result_name,
        "--metadata",
        f"variant={variant.label}",
        f"kv_cache_dtype={variant.kv_cache_dtype}",
        f"context_len={config.context_len}",
        f"input_len={config.input_len}",
        f"output_len={config.output_len}",
        "--disable-tqdm",
        "--temperature",
        "0",
    ]
    if config.max_concurrency is not None:
        command.extend(["--max-concurrency", str(config.max_concurrency)])
    if config.trust_remote_code:
        command.append("--trust-remote-code")
    command.extend(config.bench_extra_args)

    LOGGER.info(
        "running vLLM bench for %s: request_rate=%s input_len=%d output_len=%d",
        variant.label,
        request_rate,
        config.input_len,
        config.output_len,
    )
    LOGGER.debug("bench command: %s", " ".join(command))
    proc = subprocess.run(command, cwd=str(ROOT), text=True, capture_output=True)
    (out_dir / f"bench_{variant.label}_{_slug(request_rate)}.stdout.txt").write_text(
        proc.stdout,
        encoding="utf-8",
    )
    (out_dir / f"bench_{variant.label}_{_slug(request_rate)}.stderr.txt").write_text(
        proc.stderr,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise RuntimeError(f"vllm bench serve failed for {variant.label} at rate {request_rate}: {proc.stderr[-2000:]}")
    result_path = out_dir / result_name
    result = load_bench_result(result_path) if result_path.exists() else parse_bench_stdout(proc.stdout)
    validate_bench_result(
        result,
        variant=variant,
        request_rate=request_rate,
        stdout=proc.stdout,
        stderr=proc.stderr,
        server_log_path=out_dir / "server.log",
    )
    return result, result_path if result_path.exists() else None, " ".join(command)


def validate_bench_result(
    result: dict[str, Any],
    *,
    variant: Variant,
    request_rate: str,
    stdout: str,
    stderr: str,
    server_log_path: Path,
) -> None:
    completed = result.get("completed")
    failed = result.get("failed")
    errors = result.get("errors")
    if completed == 0 and failed:
        error_text = ""
        if isinstance(errors, list) and errors:
            error_text = "; ".join(str(item) for item in errors[:5])
        if not error_text:
            error_text = (stderr or stdout)[-2000:]
        server_log_tail = tail_text(server_log_path, max_lines=40)
        if server_log_tail:
            error_text += f"\n\nRecent server log:\n{server_log_tail}"
        raise RuntimeError(
            f"vLLM bench produced zero successful requests for {variant.label} "
            f"at request_rate={request_rate}. Error detail: {error_text}"
        )


def load_bench_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if not data:
            return {}
        item = data[-1]
        return item if isinstance(item, dict) else {}
    return data if isinstance(data, dict) else {}


def parse_bench_stdout(text: str) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    patterns = {
        "request_throughput": r"Request throughput.*?:\s*([0-9.]+)",
        "output_throughput": r"Output token throughput.*?:\s*([0-9.]+)",
        "total_token_throughput": r"Total Token throughput.*?:\s*([0-9.]+)",
        "mean_ttft_ms": r"Mean TTFT.*?:\s*([0-9.]+)",
        "median_ttft_ms": r"Median TTFT.*?:\s*([0-9.]+)",
        "p99_ttft_ms": r"P99 TTFT.*?:\s*([0-9.]+)",
        "mean_tpot_ms": r"Mean TPOT.*?:\s*([0-9.]+)",
        "median_tpot_ms": r"Median TPOT.*?:\s*([0-9.]+)",
        "p99_tpot_ms": r"P99 TPOT.*?:\s*([0-9.]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            rows[key] = float(match.group(1))
    return rows


def extract_bench_metrics(result: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "request_throughput": ("request_throughput", "requests_per_second", "req_s"),
        "output_throughput": ("output_throughput", "output_tokens_per_s", "output_token_throughput"),
        "mean_ttft_ms": ("mean_ttft_ms", "mean_ttft_ms_per_req"),
        "median_ttft_ms": ("median_ttft_ms", "p50_ttft_ms", "percentile_ttft_50_ms"),
        "p90_ttft_ms": ("p90_ttft_ms", "percentile_ttft_90_ms"),
        "p99_ttft_ms": ("p99_ttft_ms", "percentile_ttft_99_ms"),
        "mean_tpot_ms": ("mean_tpot_ms", "mean_tpot_ms_per_req"),
        "median_tpot_ms": ("median_tpot_ms", "p50_tpot_ms", "percentile_tpot_50_ms"),
        "p90_tpot_ms": ("p90_tpot_ms", "percentile_tpot_90_ms"),
        "p99_tpot_ms": ("p99_tpot_ms", "percentile_tpot_99_ms"),
        "median_itl_ms": ("median_itl_ms", "p50_itl_ms", "percentile_itl_50_ms"),
        "p90_itl_ms": ("p90_itl_ms", "percentile_itl_90_ms"),
        "p99_itl_ms": ("p99_itl_ms", "percentile_itl_99_ms"),
        "median_e2el_ms": ("median_e2el_ms", "p50_e2el_ms", "percentile_e2el_50_ms"),
        "p90_e2el_ms": ("p90_e2el_ms", "percentile_e2el_90_ms"),
        "p99_e2el_ms": ("p99_e2el_ms", "percentile_e2el_99_ms"),
    }
    out: dict[str, Any] = {}
    flat = _flatten_dict(result)
    for target, names in aliases.items():
        for name in names:
            if name in flat:
                out[target] = flat[name]
                break
    details = result.get("request_outputs") or result.get("details") or result.get("per_request")
    if isinstance(details, list):
        out["failed_requests"] = count_failed_requests(details)
        out["num_request_records"] = len(details)
    return out


def add_point_resource_metrics(metrics: dict[str, Any], resource_summary: dict[str, Any]) -> dict[str, Any]:
    out = dict(metrics)
    peak_gpu = resource_summary.get("gpu_mem_used_bytes_max")
    peak_rss = resource_summary.get("process_rss_bytes_max")
    peak_ram = resource_summary.get("system_ram_used_bytes_max")
    if peak_gpu is not None:
        out["peak_gpu_memory_bytes"] = peak_gpu
    if peak_rss is not None:
        out["process_rss_bytes_max"] = peak_rss
    if peak_ram is not None:
        out["system_ram_used_bytes_max"] = peak_ram
    throughput = out.get("output_throughput")
    try:
        if peak_gpu and throughput is not None:
            out["output_tokens_per_s_per_peak_gpu_gb"] = float(throughput) / (float(peak_gpu) / 1024**3)
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return out


def count_failed_requests(rows: list[Any]) -> int:
    failures = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        error = row.get("error") or row.get("error_msg") or row.get("exception")
        success = row.get("success")
        if error or success is False:
            failures += 1
    return failures


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(_flatten_dict(value, name))
            out.update(_flatten_dict(value, ""))
        else:
            out[name] = value
            out[str(key)] = value
    return out


def parse_prometheus(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$", line)
        if not match:
            continue
        try:
            metrics[match.group(1)] = float(match.group(2))
        except ValueError:
            continue
    return metrics


def filter_cache_metrics(metrics: dict[str, float]) -> dict[str, float]:
    keep: dict[str, float] = {}
    for key, value in metrics.items():
        low = key.lower()
        if "cache" in low or "kv" in low or "gpu" in low:
            keep[key] = value
    return keep


def parse_server_log_kv(text: str) -> dict[str, Any]:
    data: dict[str, Any] = {}
    token_match = re.search(r"GPU KV cache size:\s*([0-9,]+)\s*tokens", text, flags=re.IGNORECASE)
    if token_match:
        data["gpu_kv_cache_tokens"] = int(token_match.group(1).replace(",", ""))
    concurrency_match = re.search(
        r"Maximum concurrency for\s*([0-9,]+)\s*tokens per request:\s*([0-9.]+)x",
        text,
        flags=re.IGNORECASE,
    )
    if concurrency_match:
        data["max_concurrency_tokens_per_request"] = int(concurrency_match.group(1).replace(",", ""))
        data["max_concurrency_from_log"] = float(concurrency_match.group(2))
    memory_patterns = [
        r"available[^.\n]*KV cache memory[^0-9]*([0-9.]+)\s*(GiB|GB|MiB|MB)",
        r"KV cache memory[^0-9]*([0-9.]+)\s*(GiB|GB|MiB|MB)",
    ]
    for pattern in memory_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            data["kv_cache_memory_bytes_from_log"] = _size_to_bytes(float(match.group(1)), match.group(2))
            break
    oom_count = len(re.findall(r"\b(?:out of memory|OOM)\b", text, flags=re.IGNORECASE))
    if oom_count:
        data["oom_mentions"] = oom_count
    return data


def _size_to_bytes(value: float, unit: str) -> int:
    unit = unit.lower()
    if unit in {"gib", "gb"}:
        return int(value * 1024**3)
    if unit in {"mib", "mb"}:
        return int(value * 1024**2)
    return int(value)


def run_stream_probe(
    *,
    base_url: str,
    model: str,
    prompt: str,
    expected: str,
    max_tokens: int,
    seed: int,
    out_path: Path,
) -> dict[str, Any]:
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "seed": seed,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_token_s: float | None = None
    event_times: list[float] = []
    output_parts: list[str] = []
    LOGGER.info("running streaming quality probe: prompt_chars=%d max_tokens=%d", len(prompt), max_tokens)
    try:
        response_context = urllib.request.urlopen(request, timeout=600)
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        error_result = {
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "expected": expected,
            "output": "",
            "contains_expected": False,
            "ttft_ms": None,
            "total_latency_ms": (time.perf_counter() - start) * 1000.0,
            "stream_events": 0,
            "tpot_ms": None,
            "itl_ms": [],
            "error": {
                "type": "HTTPError",
                "status": exc.code,
                "reason": exc.reason,
                "body": body_text,
            },
        }
        out_path.write_text(json.dumps(error_result, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"streaming quality probe failed with HTTP {exc.code}: {body_text}"
        ) from exc
    with response_context as response:
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                item = json.loads(payload)
            except json.JSONDecodeError:
                continue
            text = _completion_text_delta(item)
            if text:
                now = time.perf_counter()
                if first_token_s is None:
                    first_token_s = now - start
                event_times.append(now)
                output_parts.append(text)
    end = time.perf_counter()
    output = "".join(output_parts)
    inter_token = [
        (event_times[idx] - event_times[idx - 1]) * 1000.0
        for idx in range(1, len(event_times))
    ]
    result = {
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "expected": expected,
        "output": output,
        "contains_expected": expected.lower() in output.lower(),
        "ttft_ms": None if first_token_s is None else first_token_s * 1000.0,
        "total_latency_ms": (end - start) * 1000.0,
        "stream_events": len(event_times),
        "tpot_ms": sum(inter_token) / len(inter_token) if inter_token else None,
        "itl_ms": inter_token,
    }
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    LOGGER.info(
        "quality probe done: contains_expected=%s ttft_ms=%s events=%d",
        result["contains_expected"],
        result["ttft_ms"],
        result["stream_events"],
    )
    return result


def _completion_text_delta(item: dict[str, Any]) -> str:
    choices = item.get("choices")
    if not choices:
        return ""
    choice = choices[0]
    if "text" in choice and choice["text"] is not None:
        return str(choice["text"])
    delta = choice.get("delta") or {}
    if isinstance(delta, dict):
        return str(delta.get("content") or "")
    return ""


def build_niah_prompt(
    *,
    model: str,
    token_target: int,
    depth: float,
    secret: str,
    local_files_only: bool,
) -> tuple[str, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=local_files_only)
    hay = (
        "This document contains operational notes about datacenter scheduling, "
        "database maintenance, scientific computing, and storage systems. "
    )
    question = f"\n\nQuestion: What is the secret code? Answer with only the code."
    needle = f"\n\nThe secret code is {secret}.\n\n"
    needle_tokens = tokenizer.encode(needle, add_special_tokens=False)
    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    body_budget = max(1, token_target - len(needle_tokens) - len(question_tokens))
    prefix_budget = int(body_budget * depth)
    suffix_budget = body_budget - prefix_budget
    prefix = _text_for_token_budget(tokenizer, hay, prefix_budget)
    suffix = _text_for_token_budget(tokenizer, hay[::-1], suffix_budget)
    prompt = prefix + needle + suffix + question
    encoded = tokenizer(prompt, truncation=True, max_length=token_target, return_tensors=None)
    prompt = tokenizer.decode(encoded["input_ids"], skip_special_tokens=True)
    return prompt, len(encoded["input_ids"])


def _text_for_token_budget(tokenizer: Any, unit: str, budget: int) -> str:
    if budget <= 0:
        return ""
    text = unit
    while len(tokenizer.encode(text, add_special_tokens=False)) < budget:
        text += unit
    ids = tokenizer.encode(text, add_special_tokens=False)[:budget]
    return tokenizer.decode(ids, skip_special_tokens=True)


def run_quality_jsonl(
    *,
    base_url: str,
    model: str,
    path: Path,
    max_examples: int,
    max_tokens: int,
    seed: int,
    out_dir: Path,
) -> list[dict[str, Any]]:
    rows = []
    for idx, raw in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if idx >= max_examples:
            break
        if not raw.strip():
            continue
        item = json.loads(raw)
        prompt = str(item["prompt"])
        expected = str(item.get("expected") or item.get("answer") or "")
        name = _slug(str(item.get("name") or f"example_{idx}"))
        rows.append(
            run_stream_probe(
                base_url=base_url,
                model=model,
                prompt=prompt,
                expected=expected,
                max_tokens=max_tokens,
                seed=seed,
                out_path=out_dir / f"quality_{name}.json",
            )
        )
    return rows


def capture_env() -> dict[str, Any]:
    packages: dict[str, Any] = {}
    for package in ("torch", "triton", "transformers", "vllm"):
        try:
            module = __import__(package)
            packages[package] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            packages[package] = {"error": str(exc)}
    gpu = _nvidia_smi()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "packages": packages,
        "gpu": gpu,
    }


def _nvidia_smi() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return {"available": False}
    if proc.returncode != 0:
        return {"available": False, "stderr": proc.stderr}
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            rows.append({"name": parts[0], "memory_total_mib": parts[1], "driver": parts[2]})
    return {"available": True, "devices": rows}


def check_torch_cuda_runtime() -> dict[str, Any]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            device_count = int(torch.cuda.device_count())
            device_names = []
            if cuda_available:
                for idx in range(device_count):
                    device_names.append(torch.cuda.get_device_name(idx))
            result = {
                "torch": getattr(torch, "__version__", "unknown"),
                "cuda_available": cuda_available,
                "device_count": device_count,
                "device_names": device_names,
                "warnings": [str(item.message) for item in caught],
            }
        except Exception as exc:
            result = {
                "torch": None,
                "cuda_available": False,
                "device_count": 0,
                "device_names": [],
                "warnings": [str(item.message) for item in caught],
                "error": f"{type(exc).__name__}: {exc}",
            }
    if not result["cuda_available"]:
        notes = "; ".join(result.get("warnings") or [])
        error = result.get("error") or notes or "PyTorch reports CUDA unavailable."
        result["error"] = error
    return result


def require_torch_cuda_runtime() -> dict[str, Any]:
    result = check_torch_cuda_runtime()
    if not result["cuda_available"]:
        raise RuntimeError(
            "PyTorch cannot initialize CUDA in this Python environment. "
            f"torch={result.get('torch')}; device_count={result.get('device_count')}; "
            f"reason={result.get('error')}"
        )
    return result


def build_variants(names: str) -> list[Variant]:
    variants = []
    for raw in names.split(","):
        name = raw.strip()
        if not name:
            continue
        dtype = KV_DTYPES.get(name, name)
        label = name if name in KV_DTYPES else _slug(dtype)
        variants.append(Variant(label=label, kv_cache_dtype=dtype))
    if not variants:
        raise ValueError("at least one variant is required")
    return variants


def derive_metrics(reports: list[VariantReport]) -> dict[str, Any]:
    by_label = {report.variant.label: report for report in reports}
    fp8 = by_label.get("fp8")
    out: dict[str, Any] = {"comparisons": []}
    if fp8 is None:
        return out
    for report in reports:
        if report.variant.label == "fp8":
            continue
        out["comparisons"].append(compare_variant(fp8, report))
    return out


def compare_variant(reference: VariantReport, candidate: VariantReport) -> dict[str, Any]:
    rows = []
    ref_points = {point.request_rate: point for point in reference.bench_points}
    cand_points = {point.request_rate: point for point in candidate.bench_points}
    for rate in sorted(set(ref_points) & set(cand_points)):
        ref = ref_points[rate].metrics
        cand = cand_points[rate].metrics
        rows.append(
            {
                "request_rate": rate,
                "ttft_ratio_ref_over_candidate": _ratio(ref.get("median_ttft_ms"), cand.get("median_ttft_ms")),
                "tpot_ratio_ref_over_candidate": _ratio(ref.get("median_tpot_ms"), cand.get("median_tpot_ms")),
                "e2el_ratio_ref_over_candidate": _ratio(ref.get("median_e2el_ms"), cand.get("median_e2el_ms")),
                "throughput_ratio_candidate_over_ref": _ratio(
                    cand.get("output_throughput"),
                    ref.get("output_throughput"),
                ),
                "request_ratio_candidate_over_ref": _ratio(
                    cand.get("request_throughput"),
                    ref.get("request_throughput"),
                ),
                "peak_gpu_memory_ratio_ref_over_candidate": _ratio(
                    ref.get("peak_gpu_memory_bytes"),
                    cand.get("peak_gpu_memory_bytes"),
                ),
                "tokens_per_peak_gpu_ratio_candidate_over_ref": _ratio(
                    cand.get("output_tokens_per_s_per_peak_gpu_gb"),
                    ref.get("output_tokens_per_s_per_peak_gpu_gb"),
                ),
                "failed_requests_ref": ref.get("failed_requests"),
                "failed_requests_candidate": cand.get("failed_requests"),
            }
        )
    ref_log = reference.server_log_kv
    cand_log = candidate.server_log_kv
    return {
        "reference": reference.variant.label,
        "candidate": candidate.variant.label,
        "by_request_rate": rows,
        "cache_token_capacity_ratio_candidate_over_ref": _ratio(
            cand_log.get("gpu_kv_cache_tokens"),
            ref_log.get("gpu_kv_cache_tokens"),
        ),
        "kv_memory_ratio_ref_over_candidate": _ratio(
            ref_log.get("kv_cache_memory_bytes_from_log"),
            cand_log.get("kv_cache_memory_bytes_from_log"),
        ),
        "quality": compare_quality(reference, candidate),
        "kv_from_log": {
            "reference": ref_log,
            "candidate": cand_log,
        },
    }


def compare_quality(reference: VariantReport, candidate: VariantReport) -> list[dict[str, Any]]:
    ref_quality = {str(row.get("name")): row for row in reference.quality}
    cand_quality = {str(row.get("name")): row for row in candidate.quality}
    rows: list[dict[str, Any]] = []
    for name in sorted(set(ref_quality) & set(cand_quality)):
        ref = ref_quality[name]
        cand = cand_quality[name]
        rows.append(
            {
                "name": name,
                "contains_expected_ref": ref.get("contains_expected"),
                "contains_expected_candidate": cand.get("contains_expected"),
                "ttft_ratio_ref_over_candidate": _ratio(ref.get("ttft_ms"), cand.get("ttft_ms")),
                "tpot_ratio_ref_over_candidate": _ratio(ref.get("tpot_ms"), cand.get("tpot_ms")),
            }
        )
    return rows


def _ratio(numerator: Any, denominator: Any) -> float | None:
    try:
        num = float(numerator)
        den = float(denominator)
    except (TypeError, ValueError):
        return None
    if den == 0:
        return None
    return num / den


def resolve_input_len(context_len: int, output_len: int, input_len: int | None) -> int:
    if input_len is not None:
        if input_len <= 0:
            raise ValueError("--input-len must be positive")
        if input_len + output_len > context_len:
            raise ValueError(
                f"--input-len + --output-len must fit --context-len "
                f"({input_len} + {output_len} > {context_len})"
            )
        return input_len
    resolved = context_len - output_len
    if resolved <= 0:
        raise ValueError("--context-len must be larger than --output-len")
    return resolved


def run_variant(
    *,
    config: RunConfig,
    variant: Variant,
    port: int,
    run_root: Path,
    run_niah: bool,
    quality_jsonl: Path | None,
    quality_max_examples: int,
    local_files_only: bool,
) -> VariantReport:
    out_dir = run_root / variant.label
    out_dir.mkdir(parents=True, exist_ok=False)
    base_url = f"http://{config.host}:{port}"
    report = VariantReport(
        variant=variant,
        out_dir=str(out_dir),
        server_command=[],
        server_log=str(out_dir / "server.log"),
        startup_seconds=0.0,
        server_log_kv={},
        prometheus_before={},
        prometheus_after={},
        resource_summary={},
    )
    total_steps = len(config.request_rates) + int(run_niah) + int(quality_jsonl is not None)
    progress = ProgressBar(total=total_steps, label=variant.label)
    with start_server(config=config, variant=variant, port=port, out_dir=out_dir) as (process, command, startup_s):
        report.server_command = command
        report.startup_seconds = startup_s
        before_metrics = get_prometheus_metrics(base_url)
        report.prometheus_before = filter_cache_metrics(before_metrics)
        with ResourceSampler(pid=process.pid) as sampler:
            for rate in config.request_rates:
                progress.advance(f"bench request_rate={rate}")
                raw_result, result_path, bench_command = run_bench(
                    config=config,
                    variant=variant,
                    port=port,
                    request_rate=rate,
                    out_dir=out_dir,
                )
                resource_summary = sampler.summary().to_dict()
                metrics = add_point_resource_metrics(
                    extract_bench_metrics(raw_result),
                    resource_summary,
                )
                metrics["bench_command"] = bench_command
                point = BenchPoint(
                    variant=variant.label,
                    request_rate=rate,
                    result_path=None if result_path is None else str(result_path),
                    raw_result=raw_result,
                    metrics=metrics,
                    resource_summary=resource_summary,
                )
                report.bench_points.append(point)
            if run_niah:
                probe_max_tokens = min(64, config.output_len)
                progress.advance(f"niah prompt_budget={config.context_len - probe_max_tokens}")
                prompt, prompt_tokens = build_niah_prompt(
                    model=config.model,
                    token_target=config.context_len - probe_max_tokens,
                    depth=0.5,
                    secret="GINKGOQ-8192",
                    local_files_only=local_files_only,
                )
                niah = run_stream_probe(
                    base_url=base_url,
                    model=config.model,
                    prompt=prompt,
                    expected="GINKGOQ-8192",
                    max_tokens=probe_max_tokens,
                    seed=config.seed,
                    out_path=out_dir / "quality_niah_8k.json",
                )
                niah["prompt_tokens"] = prompt_tokens
                report.quality.append({"name": "niah_8k", **niah})
            if quality_jsonl is not None:
                progress.advance(f"quality_jsonl examples<={quality_max_examples}")
                rows = run_quality_jsonl(
                    base_url=base_url,
                    model=config.model,
                    path=quality_jsonl,
                    max_examples=quality_max_examples,
                    max_tokens=min(128, config.output_len),
                    seed=config.seed,
                    out_dir=out_dir,
                )
                report.quality.extend({"name": f"quality_jsonl_{idx}", **row} for idx, row in enumerate(rows))
            report.resource_summary = sampler.summary().to_dict()
        after_metrics = get_prometheus_metrics(base_url)
        report.prometheus_after = filter_cache_metrics(after_metrics)
    log_text = Path(report.server_log).read_text(encoding="utf-8", errors="replace")
    report.server_log_kv = parse_server_log_kv(log_text)
    (out_dir / "variant_report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report


def check_vllm_cli_support(variants: list[Variant]) -> dict[str, Any]:
    proc = subprocess.run(
        [*VLLM_COMMAND, "serve", "--help=kv-cache-dtype"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    text = proc.stdout + "\n" + proc.stderr
    found = {variant.kv_cache_dtype: variant.kv_cache_dtype in text for variant in variants}
    return {
        "returncode": proc.returncode,
        "kv_cache_dtype_found": found,
        "help_excerpt": text[:12000],
    }


def require_vllm_cli_support(variants: list[Variant]) -> dict[str, Any]:
    support = check_vllm_cli_support(variants)
    missing = [
        dtype
        for dtype, present in support["kv_cache_dtype_found"].items()
        if not present
    ]
    if support["returncode"] != 0:
        excerpt = support.get("help_excerpt") or ""
        raise RuntimeError(f"vLLM CLI dtype check failed. Output:\n{excerpt}")
    if missing:
        raise RuntimeError(
            "The installed vLLM CLI does not list these --kv-cache-dtype values: "
            + ", ".join(missing)
            + ". Install a vLLM build that contains upstream TurboQuant cache dtypes, "
            + "or run only dtypes listed by `vllm serve --help=kv-cache-dtype`."
        )
    return support


def get_prometheus_metrics(base_url: str) -> dict[str, float]:
    try:
        text = http_text(f"{base_url}/metrics", timeout_s=10.0)
    except (urllib.error.URLError, TimeoutError, OSError):
        return {}
    return parse_prometheus(text)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "value"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare vLLM FP8 and TurboQuant KV cache paths.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-root", default="reports/vllm_fp8_tq")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--variants", default="fp8,tq4")
    parser.add_argument("--context-len", type=int, default=8192)
    parser.add_argument("--input-len", type=int)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--request-rates", default="1,4,inf")
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=8800)
    parser.add_argument("--startup-timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--serve-extra-arg",
        action="append",
        default=[],
        help="Append one raw argument to `vllm serve`; repeat for flag values.",
    )
    parser.add_argument(
        "--bench-extra-arg",
        action="append",
        default=[],
        help="Append one raw argument to `vllm bench serve`; repeat for flag values.",
    )
    parser.add_argument("--skip-cli-check", action="store_true")
    parser.add_argument("--skip-runtime-check", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--run-niah", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quality-jsonl")
    parser.add_argument("--quality-max-examples", type=int, default=32)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    variants = build_variants(args.variants)
    try:
        input_len = resolve_input_len(args.context_len, args.output_len, args.input_len)
        runtime_check = None if args.skip_runtime_check else require_torch_cuda_runtime()
        cli_support = None if args.skip_cli_check else require_vllm_cli_support(variants)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.out_root) / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    LOGGER.info("run_id=%s output_dir=%s", run_id, run_root)
    LOGGER.info(
        "context_len=%d input_len=%d output_len=%d variants=%s request_rates=%s",
        args.context_len,
        input_len,
        args.output_len,
        ",".join(variant.label for variant in variants),
        ",".join(item.strip() for item in args.request_rates.split(",") if item.strip()),
    )
    config = RunConfig(
        model=args.model,
        context_len=args.context_len,
        input_len=input_len,
        output_len=args.output_len,
        num_prompts=args.num_prompts,
        request_rates=tuple(item.strip() for item in args.request_rates.split(",") if item.strip()),
        max_concurrency=args.max_concurrency,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
        host=args.host,
        base_port=args.base_port,
        startup_timeout_s=args.startup_timeout_s,
        skip_cli_check=args.skip_cli_check,
        serve_extra_args=tuple(args.serve_extra_arg or ()),
        bench_extra_args=tuple(args.bench_extra_arg or ()),
    )
    manifest = {
        "run_id": run_id,
        "config": config.to_dict(),
        "variants": [variant.to_dict() for variant in variants],
        "environment": capture_env(),
        "vllm_cli_support": cli_support,
        "torch_cuda_runtime": runtime_check,
        "sources": {
            "vllm_bench_serve_docs": "https://docs.vllm.ai/en/latest/cli/bench/serve/",
            "vllm_turboquant_docs": "https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant/",
        },
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("manifest written: %s", run_root / "manifest.json")

    reports = []
    quality_jsonl = Path(args.quality_jsonl) if args.quality_jsonl else None
    for idx, variant in enumerate(variants):
        LOGGER.info("starting variant %s/%s: %s (%s)", idx + 1, len(variants), variant.label, variant.kv_cache_dtype)
        reports.append(
            run_variant(
                config=config,
                variant=variant,
                port=args.base_port + idx,
                run_root=run_root,
                run_niah=args.run_niah,
                quality_jsonl=quality_jsonl,
                quality_max_examples=args.quality_max_examples,
                local_files_only=args.local_files_only,
            )
        )
    summary = {
        "run_id": run_id,
        "manifest": manifest,
        "reports": [report.to_dict() for report in reports],
        "derived": derive_metrics(reports),
    }
    (run_root / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("report written: %s", run_root / "report.json")
    print(json.dumps({"run_id": run_id, "report": str(run_root / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
