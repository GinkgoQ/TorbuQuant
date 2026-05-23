"""Command helpers for external evaluation engines."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class KVSpec:
    key_cache: str = "f16"
    value_cache: str = "f16"
    env: dict[str, str] = field(default_factory=dict)
    flags: dict[str, str] = field(default_factory=dict)

    @classmethod
    def parse(cls, text: str) -> "KVSpec":
        key_cache = "f16"
        value_cache = "f16"
        env: dict[str, str] = {}
        flags: dict[str, str] = {}
        for part in text.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"bad KV spec fragment: {part}")
            key, value = [piece.strip() for piece in part.split("=", 1)]
            if key == "ctk":
                key_cache = value
            elif key == "ctv":
                value_cache = value
            elif key.startswith("env."):
                env[key[4:]] = value
            else:
                flags[key] = value
        return cls(key_cache=key_cache, value_cache=value_cache, env=env, flags=flags)

    def cli_args(self) -> list[str]:
        args = ["-ctk", self.key_cache, "-ctv", self.value_cache]
        for key, value in self.flags.items():
            args.extend([f"--{key}", value])
        return args

    def label(self) -> str:
        parts = [f"ctk={self.key_cache}", f"ctv={self.value_cache}"]
        parts.extend(f"env.{key}={value}" for key, value in sorted(self.env.items()))
        parts.extend(f"{key}={value}" for key, value in sorted(self.flags.items()))
        return ",".join(parts)


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    env: dict[str, str]

    def shell_text(self) -> str:
        return " ".join(shlex.quote(item) for item in self.argv)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["argv"] = list(self.argv)
        return data


def extra_engine_flags(env_var: str = "TORBUQUANT_ENGINE_FLAGS") -> list[str]:
    raw = os.environ.get(env_var, "").strip()
    return shlex.split(raw) if raw else []


def build_llama_generation_command(
    *,
    bin_dir: Path,
    model: Path,
    prompt: str,
    kv: KVSpec,
    tokens: int = 128,
    context: int = 512,
    gpu_layers: int = 99,
    seed: int = 42,
    temperature: float = 0.0,
    apply_template: bool = True,
    system_prompt: str | None = None,
    engine_flags: list[str] | None = None,
) -> CommandSpec:
    argv = [
        str(bin_dir / "llama-cli"),
        "-m",
        str(model),
        "-p",
        prompt,
        "-n",
        str(tokens),
        "-c",
        str(context),
        "-ngl",
        str(gpu_layers),
        "--seed",
        str(seed),
        "--temp",
        str(temperature),
        "--single-turn",
        "--no-display-prompt",
        "-fa",
        "on",
    ]
    if apply_template:
        argv.append("--jinja")
        if system_prompt:
            argv.extend(["-sys", system_prompt])
    argv.extend(kv.cli_args())
    argv.extend(engine_flags if engine_flags is not None else extra_engine_flags())
    return CommandSpec(argv=tuple(argv), env=dict(kv.env))


def build_llama_trajectory_command(
    *,
    bin_dir: Path,
    model: Path,
    prompt: str,
    trajectory_path: Path,
    kv: KVSpec,
    tokens: int = 128,
    context: int = 512,
    gpu_layers: int = 99,
    seed: int = 42,
    temperature: float = 0.0,
    apply_template: bool = True,
    system_prompt: str | None = None,
    engine_flags: list[str] | None = None,
) -> CommandSpec:
    argv = [
        str(bin_dir / "llama-completion"),
        "-m",
        str(model),
        "-p",
        prompt,
        "-n",
        str(tokens),
        "-c",
        str(context),
        "-ngl",
        str(gpu_layers),
        "--seed",
        str(seed),
        "--temp",
        str(temperature),
        "-no-cnv",
        "--no-display-prompt",
        "-fa",
        "on",
    ]
    if apply_template:
        argv.append("--jinja")
        if system_prompt:
            argv.extend(["-sys", system_prompt])
    argv.extend(kv.cli_args())
    argv.extend(engine_flags if engine_flags is not None else extra_engine_flags())
    env = dict(kv.env)
    env["TORBUQUANT_TRAJECTORY"] = str(trajectory_path)
    return CommandSpec(argv=tuple(argv), env=env)


def build_llama_tokenize_command(*, bin_dir: Path, model: Path) -> CommandSpec:
    return CommandSpec(
        argv=(
            str(bin_dir / "llama-tokenize"),
            "-m",
            str(model),
            "--ids",
            "--no-bos",
            "--no-parse-special",
            "--log-disable",
            "--stdin",
        ),
        env={},
    )


def build_llama_kld_base_command(
    *,
    bin_dir: Path,
    model: Path,
    corpus: Path,
    base_path: Path,
    kv: KVSpec,
    chunks: int = 32,
    context: int = 512,
    gpu_layers: int = 99,
    engine_flags: list[str] | None = None,
) -> CommandSpec:
    argv = [
        str(bin_dir / "llama-perplexity"),
        "-m",
        str(model),
        "-f",
        str(corpus),
        "-c",
        str(context),
        "--chunks",
        str(chunks),
        "-ngl",
        str(gpu_layers),
        "-fa",
        "on",
        "--kl-divergence-base",
        str(base_path),
    ]
    argv.extend(kv.cli_args())
    argv.extend(engine_flags if engine_flags is not None else extra_engine_flags())
    return CommandSpec(argv=tuple(argv), env=dict(kv.env))


def build_llama_kld_command(
    *,
    bin_dir: Path,
    model: Path,
    corpus: Path,
    base_path: Path,
    kv: KVSpec,
    chunks: int = 32,
    context: int = 512,
    gpu_layers: int = 99,
    engine_flags: list[str] | None = None,
) -> CommandSpec:
    spec = build_llama_kld_base_command(
        bin_dir=bin_dir,
        model=model,
        corpus=corpus,
        base_path=base_path,
        kv=kv,
        chunks=chunks,
        context=context,
        gpu_layers=gpu_layers,
        engine_flags=engine_flags,
    )
    argv = list(spec.argv)
    insert_at = argv.index("--kl-divergence-base")
    argv.insert(insert_at, "--kl-divergence")
    return CommandSpec(argv=tuple(argv), env=spec.env)


_NOISE_PATTERNS = (
    re.compile(r"^\[End thinking\].*$", re.MULTILINE),
    re.compile(r"^\[ Prompt:.*\]$", re.MULTILINE),
    re.compile(r"^Exiting\.\.\..*$", re.MULTILINE),
    re.compile(r"^llama_perf_.*$", re.MULTILINE),
    re.compile(r"^Log end$", re.MULTILINE),
    re.compile(r"^Loading model\.\.\..*$", re.MULTILINE),
)
_GEN_LINE_RE = re.compile(r"^\|\s.*", re.MULTILINE)
_BLOCK_RE = re.compile(r"^[\s\u2580-\u259F]+$", re.MULTILINE)


def strip_engine_output(text: str) -> str:
    out = text.replace("\x08", "")
    for pattern in _NOISE_PATTERNS:
        out = pattern.sub("", out)
    matches = list(_GEN_LINE_RE.finditer(out))
    if matches:
        out = out[matches[0].start():]
        out = re.sub(r"^\|\s?", "", out, flags=re.MULTILINE)
    out = _BLOCK_RE.sub("", out)
    return out.strip()


_PPL_RE = re.compile(r"Final estimate:\s*PPL\s*=\s*([0-9.]+)")
_KLD_RE = re.compile(r"Mean\s+KLD:\s*([0-9.+\-eE]+)")
_RMS_RE = re.compile(r"RMS Δp:\s*([0-9.]+)\s*%", re.UNICODE)
_TOP_RE = re.compile(r"Same\s+top[-\s]?p:\s*([0-9.]+)\s*%")


def parse_kld_metrics(text: str) -> dict[str, float | None]:
    return {
        "ppl": _first_float(_PPL_RE, text),
        "mean_kld": _first_float(_KLD_RE, text),
        "rms_dp_pct": _first_float(_RMS_RE, text),
        "same_top_p_pct": _first_float(_TOP_RE, text),
    }


def parse_token_ids(text: str) -> list[int]:
    stripped = text.strip()
    if not stripped or not stripped.startswith("["):
        return []
    inner = stripped.strip("[] \n")
    if not inner:
        return []
    return [int(piece.strip()) for piece in inner.split(",") if piece.strip()]


def parse_trajectory_jsonl(text: str) -> list[int]:
    ids: list[int] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        ids.append(int(record["token_id"]))
    return ids


_VLLM_DTYPE_MAP = {
    ("f16", "f16"): "auto",
    ("bf16", "bf16"): "auto",
    ("q8_0", "q8_0"): "fp8_e4m3",
    ("q8_0", "turbo4"): "turboquant_k8v4",
    ("q8_0", "turbo3"): "turboquant_k8v3",
    ("turbo4", "turbo4"): "turboquant_4bit_nc",
    ("turbo3", "turbo3"): "turboquant_3bit_nc",
    ("turbo3", "turbo4"): "turboquant_k3v4_nc",
    ("turbo4", "turbo4_rv"): "turboquant_4bit_nc_rv",
    ("turbo4", "turbo4_cv"): "turboquant_4bit_nc_cv_rv",
}


def vllm_kv_dtype(kv: KVSpec) -> str:
    key = (kv.key_cache.lower(), kv.value_cache.lower())
    if key not in _VLLM_DTYPE_MAP:
        raise ValueError(f"vLLM KV dtype mapping is missing for ctk={key[0]}, ctv={key[1]}")
    return _VLLM_DTYPE_MAP[key]


def sglang_kv_pair(kv: KVSpec) -> tuple[str, str]:
    key = kv.key_cache.lower()
    value = kv.value_cache.lower()
    if (key, value) in {("f16", "f16"), ("bf16", "bf16"), ("q8_0", "q8_0")}:
        return key, value
    raise ValueError(f"SGLang KV mapping is missing for ctk={key}, ctv={value}")


def corpus_identity(path: Path, *, head_bytes: int = 1024 * 1024) -> dict[str, object]:
    size = path.stat().st_size
    h = hashlib.sha256()
    with path.open("rb") as handle:
        h.update(handle.read(head_bytes))
    return {
        "path": str(path),
        "size_bytes": size,
        "sha256_head": h.hexdigest(),
        "sha256_head_bytes": min(size, head_bytes),
    }


def write_corpus_sidecar(base_path: Path, corpus: Path) -> Path:
    sidecar = Path(str(base_path) + ".corpus.json")
    sidecar.write_text(json.dumps(corpus_identity(corpus), indent=2), encoding="utf-8")
    return sidecar


def assert_corpus_identity(base_path: Path, corpus: Path) -> None:
    sidecar = Path(str(base_path) + ".corpus.json")
    if not sidecar.exists():
        return
    expected = json.loads(sidecar.read_text(encoding="utf-8"))
    actual = corpus_identity(corpus)
    if expected.get("sha256_head") != actual["sha256_head"] or expected.get("size_bytes") != actual["size_bytes"]:
        raise RuntimeError("corpus identity mismatch for KL base")


def _first_float(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    return float(match.group(1)) if match else None
