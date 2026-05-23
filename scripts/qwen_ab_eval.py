"""Controlled HuggingFace vs TurboQuant Qwen evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from torbuquant.integration import KVQuantConfig
from torbuquant.integration.hf import DynamicCachePatch, HFDiagnosticCacheAdapter
from torbuquant.bench import DeviceProfile, ModelProfile, RunPoint, RunProfile, SystemProfile
from torbuquant.quality import trajectory_metrics


@dataclass(frozen=True)
class Scenario:
    name: str
    token_target: int
    prompt: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class ResourceSampler:
    def __init__(self, *, device_index: int = 0, interval_s: float = 0.05):
        self.device_index = device_index
        self.interval_s = interval_s
        self.psutil = _load_psutil()
        self.nvml = _load_nvml()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[dict[str, Any]] = []
        self._process = self.psutil.Process(os.getpid()) if self.psutil is not None else None
        self._handle = None
        if self.nvml is not None:
            try:
                self._handle = self.nvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception:
                self._handle = None

    def __enter__(self) -> "ResourceSampler":
        if self._process is not None:
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
            row: dict[str, Any] = {"time_s": time.time()}
            if self._process is not None and self.psutil is not None:
                with self._process.oneshot():
                    mem = self._process.memory_info()
                    row["process_cpu_percent"] = self._process.cpu_percent(interval=None)
                    row["process_rss_bytes"] = int(mem.rss)
                vm = self.psutil.virtual_memory()
                row["system_ram_used_bytes"] = int(vm.used)
                row["system_ram_percent"] = float(vm.percent)
            if self._handle is not None and self.nvml is not None:
                try:
                    util = self.nvml.nvmlDeviceGetUtilizationRates(self._handle)
                    mem_info = self.nvml.nvmlDeviceGetMemoryInfo(self._handle)
                    row["gpu_util_percent"] = int(util.gpu)
                    row["gpu_mem_util_percent"] = int(util.memory)
                    row["gpu_mem_used_bytes"] = int(mem_info.used)
                except Exception:
                    pass
            self.samples.append(row)
            self._stop.wait(self.interval_s)

    def summary(self) -> dict[str, Any]:
        def max_field(name: str) -> int | float | None:
            values = [row[name] for row in self.samples if name in row]
            return max(values) if values else None

        def mean_field(name: str) -> float | None:
            values = [float(row[name]) for row in self.samples if name in row]
            return float(sum(values) / len(values)) if values else None

        return {
            "samples": len(self.samples),
            "process_cpu_percent_mean": mean_field("process_cpu_percent"),
            "process_cpu_percent_max": max_field("process_cpu_percent"),
            "process_rss_bytes_max": max_field("process_rss_bytes"),
            "system_ram_used_bytes_max": max_field("system_ram_used_bytes"),
            "system_ram_percent_max": max_field("system_ram_percent"),
            "gpu_util_percent_mean": mean_field("gpu_util_percent"),
            "gpu_util_percent_max": max_field("gpu_util_percent"),
            "gpu_mem_util_percent_max": max_field("gpu_mem_util_percent"),
            "gpu_mem_used_bytes_max": max_field("gpu_mem_used_bytes"),
        }


def _load_transformers():
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    return AutoModelForCausalLM, AutoTokenizer, DynamicCache


def load_local_text() -> str:
    paths = [
        ROOT / "paper" / "main.tex",
        ROOT / "RESEARCH_NOTES.md",
        ROOT / "ARCHITECTURE.md",
        ROOT / "plan.md",
    ]
    parts = [path.read_text(encoding="utf-8", errors="replace") for path in paths if path.exists()]
    text = "\n\n".join(parts).strip()
    if not text:
        text = "TurboQuant compresses key and value caches for long context language model inference."
    return text


def build_scenarios() -> list[Scenario]:
    long_text = load_local_text()
    return [
        Scenario(
            name="short_prompt",
            token_target=64,
            prompt=(
                "Explain in two paragraphs why KV-cache compression can help serving "
                "large language models when context length grows."
            ),
        ),
        Scenario(
            name="medium_context",
            token_target=256,
            prompt=(
                "Summarize the following technical notes and identify the main risks for "
                "compressed attention:\n\n"
                + long_text[:6000]
            ),
        ),
        Scenario(
            name="long_context",
            token_target=1024,
            prompt=(
                "Use the following research context to answer: what evidence is needed before "
                "claiming a serving benefit from compressed KV cache?\n\n"
                + long_text
            ),
        ),
    ]


def build_context_scenarios(token_targets: list[int]) -> list[Scenario]:
    long_text = load_local_text()
    scenarios: list[Scenario] = []
    for target in token_targets:
        if target <= 0:
            raise ValueError("token targets must be positive")
        scenarios.append(
            Scenario(
                name=f"context_{target}",
                token_target=target,
                prompt=(
                    "Use the following research context to answer: what evidence is needed before "
                    "claiming a serving benefit from compressed KV cache?\n\n"
                    + long_text
                ),
            )
        )
    return scenarios


def prepare_prompt(tokenizer: Any, scenario: Scenario) -> dict[str, Any]:
    encoded = tokenizer(
        scenario.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=scenario.token_target,
    )
    tokens = int(encoded["input_ids"].shape[1])
    if tokens < scenario.token_target:
        repeat_count = max(2, scenario.token_target // max(tokens, 1) + 2)
        encoded = tokenizer(
            "\n\n".join([scenario.prompt] * repeat_count),
            return_tensors="pt",
            truncation=True,
            max_length=scenario.token_target,
        )
    prompt_text = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=True)
    return {
        "scenario": scenario.name,
        "token_target": scenario.token_target,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "prompt_text": prompt_text,
        "input_tokens": int(encoded["input_ids"].shape[1]),
    }


def tensor_digest(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def reset_device() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def load_model_and_tokenizer(model_id: str, *, dtype: torch.dtype, local_files_only: bool):
    AutoModelForCausalLM, AutoTokenizer, _DynamicCache = _load_transformers()
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    ).to("cuda")
    model.eval()
    return model, tokenizer


def dense_kv_estimate_bytes(model: Any, *, total_tokens: int, dtype: torch.dtype) -> int:
    cfg = model.config
    layers = int(getattr(cfg, "num_hidden_layers"))
    kv_heads = int(getattr(cfg, "num_key_value_heads"))
    q_heads = int(getattr(cfg, "num_attention_heads"))
    head_dim = int(getattr(cfg, "hidden_size")) // q_heads
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()
    return int(layers * 2 * total_tokens * kv_heads * head_dim * dtype_bytes)


def compressed_cache_bytes(adapter: HFDiagnosticCacheAdapter) -> int:
    report = adapter.report()
    memory = report.get("memory") or {}
    return int(sum(int(row["compressed_total_bytes"]) for row in memory.values()))


def run_generation(
    *,
    kind: str,
    model_id: str,
    prompt_pack: dict[str, Any],
    generation_settings: dict[str, Any],
    preset: str,
    seed: int,
    dtype: torch.dtype,
    local_files_only: bool,
    out_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    reset_device()
    out_dir.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load_model_and_tokenizer(model_id, dtype=dtype, local_files_only=local_files_only)
    batch = {
        "input_ids": prompt_pack["input_ids"].to("cuda"),
        "attention_mask": prompt_pack["attention_mask"].to("cuda"),
    }
    adapter: HFDiagnosticCacheAdapter | None = None
    cache_patch = None
    cache = None
    if kind == "tq":
        _AutoModelForCausalLM, _AutoTokenizer, DynamicCache = _load_transformers()
        cfg = model.config
        config = KVQuantConfig(
            preset=preset,
            backend="hf",
            mode="diagnostic",
            model_id=model_id,
            context_length=int(batch["input_ids"].shape[1] + generation_settings["max_new_tokens"]),
            batch_size=1,
            num_q_heads=int(cfg.num_attention_heads),
            num_kv_heads=int(cfg.num_key_value_heads),
            head_dim=int(cfg.hidden_size // cfg.num_attention_heads),
            num_layers=int(cfg.num_hidden_layers),
            recent_window=0,
            seed=seed,
            local_files_only=local_files_only,
        )
        adapter = HFDiagnosticCacheAdapter(config=config, device=torch.device("cuda"), dtype=dtype)
        cache = DynamicCache(config=model.config)
        cache_patch = DynamicCachePatch(cache, adapter)

    generated = None
    start = time.perf_counter()
    with torch.no_grad(), ResourceSampler(device_index=0) as sampler:
        if cache_patch is None:
            generated = model.generate(
                **batch,
                **generation_settings,
            )
        else:
            with cache_patch:
                generated = model.generate(
                    **batch,
                    past_key_values=cache,
                    **generation_settings,
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    runtime_s = time.perf_counter() - start
    resource_summary = sampler.summary()
    peak_cuda_bytes = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None

    prompt_tokens = int(batch["input_ids"].shape[1])
    output_tokens = int(generated.shape[1] - prompt_tokens)
    total_tokens = int(generated.shape[1])
    text_full = tokenizer.decode(generated[0], skip_special_tokens=True)
    text_new = tokenizer.decode(generated[0, prompt_tokens:], skip_special_tokens=True)
    dense_bytes = dense_kv_estimate_bytes(model, total_tokens=total_tokens, dtype=dtype)
    tq_bytes = compressed_cache_bytes(adapter) if adapter is not None else None
    report = {
        "kind": kind,
        "model_id": model_id,
        "scenario": prompt_pack["scenario"],
        "token_target": prompt_pack["token_target"],
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "generation_settings": generation_settings,
        "seed": seed,
        "dtype": str(dtype),
        "prompt_digest": tensor_digest(batch["input_ids"]),
        "generated_digest": tensor_digest(generated),
        "runtime_s": runtime_s,
        "latency_s_per_output_token": runtime_s / output_tokens if output_tokens else None,
        "throughput_output_tokens_per_s": output_tokens / runtime_s if runtime_s > 0 else None,
        "peak_cuda_allocated_bytes": peak_cuda_bytes,
        "resource_summary": resource_summary,
        "dense_kv_estimate_bytes": dense_bytes,
        "tq_cache_report": adapter.report() if adapter is not None else None,
        "tq_compressed_cache_bytes": tq_bytes,
        "tq_cache_storage_ratio": dense_bytes / tq_bytes if tq_bytes else None,
        "prompt_text": prompt_pack["prompt_text"],
        "raw_output_full": text_full,
        "raw_output_new": text_new,
        "tokens": generated.detach().cpu().tolist()[0],
        "new_token_ids": generated[0, prompt_tokens:].detach().cpu().tolist(),
    }
    (out_dir / "output.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "raw_output.txt").write_text(text_full, encoding="utf-8")
    del generated, batch, model, tokenizer, adapter, cache, cache_patch
    reset_device()
    return report


def compare_outputs(hf: dict[str, Any], tq: dict[str, Any]) -> dict[str, Any]:
    hf_tokens = hf["tokens"]
    tq_tokens = tq["tokens"]
    hf_new_tokens = hf["new_token_ids"]
    tq_new_tokens = tq["new_token_ids"]
    n = min(len(hf_tokens), len(tq_tokens))
    token_match = sum(1 for i in range(n) if hf_tokens[i] == tq_tokens[i])
    traj = trajectory_metrics([hf_new_tokens], [tq_new_tokens], expected_tokens=hf["output_tokens"])
    output_match = hf["raw_output_new"] == tq["raw_output_new"]
    peak_hf = hf["peak_cuda_allocated_bytes"] or 0
    peak_tq = tq["peak_cuda_allocated_bytes"] or 0
    rt_hf = float(hf["runtime_s"])
    rt_tq = float(tq["runtime_s"])
    return {
        "same_prompt_digest": hf["prompt_digest"] == tq["prompt_digest"],
        "raw_new_output_match": output_match,
        "token_match_ratio": token_match / n if n else 1.0,
        "generated_trajectory": traj.to_dict(),
        "peak_cuda_allocated_delta_bytes": peak_tq - peak_hf,
        "peak_cuda_allocated_ratio_tq_over_hf": peak_tq / peak_hf if peak_hf else None,
        "runtime_delta_s": rt_tq - rt_hf,
        "runtime_ratio_tq_over_hf": rt_tq / rt_hf if rt_hf else None,
        "throughput_ratio_tq_over_hf": (
            tq["throughput_output_tokens_per_s"] / hf["throughput_output_tokens_per_s"]
            if hf["throughput_output_tokens_per_s"]
            else None
        ),
        "cache_storage_ratio_dense_over_tq": tq["tq_cache_storage_ratio"],
    }


def artifact_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            rel = file_path.relative_to(path).as_posix()
            rows.append({"path": rel, "bytes": file_path.stat().st_size})
    return rows


def build_bench_profile(
    *,
    run_id: str,
    model_id: str,
    preset: str,
    run_reports: list[dict[str, Any]],
) -> RunProfile:
    points: list[RunPoint] = []
    model = ModelProfile(model_id=model_id)
    for pair in run_reports:
        for kind in ("hf", "tq"):
            row = pair[kind]
            tq_report = row.get("tq_cache_report") or {}
            k_format = "K16" if kind == "hf" else str(tq_report.get("k_format", preset.upper()))
            v_format = "V16" if kind == "hf" else str(tq_report.get("v_format", preset.upper()))
            config = tq_report.get("config") or {}
            if kind == "tq" and config:
                model = ModelProfile(
                    model_id=str(config.get("model_id", model_id)),
                    layers=int(config.get("num_layers") or 0),
                    query_heads=int(config.get("num_q_heads") or 0),
                    kv_heads=int(config.get("num_kv_heads") or 0),
                    head_dim=int(config.get("head_dim") or 0),
                )
            points.append(
                RunPoint(
                    label=f"{row['scenario']}:{kind}",
                    k_format=k_format,
                    v_format=v_format,
                    mode="generate",
                    context_tokens=int(row["input_tokens"]),
                    batch_size=1,
                    tokens_per_s=float(row["throughput_output_tokens_per_s"] or 0.0),
                    mean_ms=float(row["runtime_s"]) * 1000.0,
                    median_ms=float(row["runtime_s"]) * 1000.0,
                    peak_gpu_bytes=int(row["peak_cuda_allocated_bytes"] or 0),
                    env=kind,
                )
            )
    device = DeviceProfile(
        name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
        backend="cuda" if torch.cuda.is_available() else "cpu",
        cuda=torch.version.cuda,
    )
    system = SystemProfile(
        platform=platform.platform(),
        python=platform.python_version(),
        torch=torch.__version__,
        device=device,
    )
    return RunProfile(
        version=1,
        timestamp=run_id,
        system=system,
        model=model,
        points=tuple(points),
        notes=("same model, prompts, generation settings, seed, and process runner",),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--preset", default="k8_v4")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-root", default="reports/qwen_ab")
    parser.add_argument("--scenarios", default="short_prompt,medium_context,long_context")
    parser.add_argument("--token-targets", default="")
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation")
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.out_root) / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    settings = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "return_dict_in_generate": False,
        "pad_token_id": None,
    }
    _AutoModelForCausalLM, AutoTokenizer, _DynamicCache = _load_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    settings["pad_token_id"] = tokenizer.pad_token_id
    built_scenarios = build_scenarios()
    if args.token_targets.strip():
        targets = [int(x) for x in args.token_targets.split(",") if x.strip()]
        built_scenarios.extend(build_context_scenarios(targets))
    scenario_map = {scenario.name: scenario for scenario in built_scenarios}
    selected = (
        [f"context_{int(x)}" for x in args.token_targets.split(",") if x.strip()]
        if args.token_targets.strip()
        else [name.strip() for name in args.scenarios.split(",") if name.strip()]
    )
    prompt_packs = []
    for name in selected:
        if name not in scenario_map:
            raise ValueError(f"unknown scenario: {name}")
        prompt_packs.append(prepare_prompt(tokenizer, scenario_map[name]))
    del tokenizer
    reset_device()

    comparisons = []
    run_reports = []
    for prompt_pack in prompt_packs:
        scenario_dir = run_root / prompt_pack["scenario"]
        hf_report = run_generation(
            kind="hf",
            model_id=args.model_id,
            prompt_pack=prompt_pack,
            generation_settings=settings,
            preset=args.preset,
            seed=args.seed,
            dtype=torch.float16,
            local_files_only=args.local_files_only,
            out_dir=scenario_dir / "hf",
        )
        tq_report = run_generation(
            kind="tq",
            model_id=args.model_id,
            prompt_pack=prompt_pack,
            generation_settings=settings,
            preset=args.preset,
            seed=args.seed,
            dtype=torch.float16,
            local_files_only=args.local_files_only,
            out_dir=scenario_dir / "tq",
        )
        comparison = compare_outputs(hf_report, tq_report)
        (scenario_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        comparisons.append({"scenario": prompt_pack["scenario"], **comparison})
        run_reports.append({"hf": hf_report, "tq": tq_report})

    report = {
        "run_id": run_id,
        "model_id": args.model_id,
        "preset": args.preset,
        "seed": args.seed,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        },
        "generation_settings": settings,
        "runs": run_reports,
        "comparisons": comparisons,
        "artifact_manifest": [],
        "notes": [
            "TurboQuant path uses the HuggingFace diagnostic adapter.",
            "The adapter stores compressed cache state but returns dense tensors to HuggingFace attention.",
            "Observed CUDA peak memory is therefore not expected to show serving-cache reduction.",
        ],
    }
    bench_profile = build_bench_profile(
        run_id=run_id,
        model_id=args.model_id,
        preset=args.preset,
        run_reports=run_reports,
    )
    (run_root / "bench_profile.json").write_text(bench_profile.to_json(), encoding="utf-8")
    report["bench_profile"] = bench_profile.to_dict()
    report_path = run_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["artifact_manifest"] = artifact_manifest(run_root)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = {
        "run_id": run_id,
        "model_id": args.model_id,
        "preset": args.preset,
        "scenarios": selected,
        "comparisons": comparisons,
        "report_path": str(run_root / "report.json"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
