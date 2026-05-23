"""Benchmark profile parsing and replay helpers."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DeviceProfile:
    name: str = "unknown"
    backend: str = "unknown"
    cuda: str | None = None
    compute_capability: str | None = None
    vram_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SystemProfile:
    platform: str = "unknown"
    python: str = "unknown"
    torch: str = "unknown"
    triton: str | None = None
    transformers: str | None = None
    vllm: str | None = None
    ram_bytes: int = 0
    device: DeviceProfile = field(default_factory=DeviceProfile)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelProfile:
    model_id: str = ""
    revision: str = ""
    layers: int = 0
    query_heads: int = 0
    kv_heads: int = 0
    head_dim: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunPoint:
    label: str
    k_format: str
    v_format: str
    mode: str
    context_tokens: int
    batch_size: int
    tokens_per_s: float
    mean_ms: float = 0.0
    median_ms: float = 0.0
    variance_ms2: float = 0.0
    peak_gpu_bytes: int = 0
    env: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunProfile:
    version: int = 1
    timestamp: str = ""
    system: SystemProfile = field(default_factory=SystemProfile)
    model: ModelProfile = field(default_factory=ModelProfile)
    points: tuple[RunPoint, ...] = tuple()
    notes: tuple[str, ...] = tuple()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["points"] = [point.to_dict() for point in self.points]
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "RunProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        system_data = data.get("system", {})
        device = DeviceProfile(**system_data.get("device", {}))
        system = SystemProfile(**{k: v for k, v in system_data.items() if k != "device"}, device=device)
        model = ModelProfile(**data.get("model", {}))
        points = tuple(RunPoint(**row) for row in data.get("points", []))
        return cls(
            version=int(data.get("version", 1)),
            timestamp=str(data.get("timestamp", "")),
            system=system,
            model=model,
            points=points,
            notes=tuple(data.get("notes", ())),
        )

    def curve(self, *, mode: str, k_format: str | None = None, v_format: str | None = None) -> dict[int, float]:
        rows: dict[int, float] = {}
        for point in self.points:
            if point.mode != mode:
                continue
            if k_format is not None and point.k_format != k_format:
                continue
            if v_format is not None and point.v_format != v_format:
                continue
            rows[point.context_tokens] = point.tokens_per_s
        return dict(sorted(rows.items()))

    def ratio_curve(
        self,
        *,
        numerator_k: str,
        numerator_v: str,
        denominator_k: str,
        denominator_v: str,
        mode: str,
    ) -> dict[int, float]:
        num = self.curve(mode=mode, k_format=numerator_k, v_format=numerator_v)
        den = self.curve(mode=mode, k_format=denominator_k, v_format=denominator_v)
        out: dict[int, float] = {}
        for context in sorted(set(num) & set(den)):
            if den[context] > 0:
                out[context] = num[context] / den[context]
        return out


def compare_run_profiles(
    reference: RunProfile,
    candidate: RunProfile,
    *,
    mode: str,
    k_format: str,
    v_format: str,
) -> dict[int, dict[str, float]]:
    ref = reference.curve(mode=mode, k_format=k_format, v_format=v_format)
    cand = candidate.curve(mode=mode, k_format=k_format, v_format=v_format)
    rows: dict[int, dict[str, float]] = {}
    for context in sorted(set(ref) & set(cand)):
        ref_value = ref[context]
        cand_value = cand[context]
        rows[context] = {
            "reference": ref_value,
            "candidate": cand_value,
            "ratio": cand_value / ref_value if ref_value else 0.0,
            "delta": cand_value - ref_value,
        }
    return rows


def parse_profile_text(text: str) -> RunProfile:
    version = 1
    timestamp = ""
    system: dict[str, Any] = {}
    device: dict[str, Any] = {}
    model: dict[str, Any] = {}
    points: list[RunPoint] = []
    notes: list[str] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("TBQ_PROFILE_VERSION="):
            version = _to_int(line.split("=", 1)[1])
        elif line.startswith("TBQ_PROFILE_TIMESTAMP="):
            timestamp = line.split("=", 1)[1]
        elif line.startswith("[SYS]"):
            system.update(_parse_fields(line[5:]))
        elif line.startswith("[GPU]"):
            device.update(_parse_fields(line[5:]))
        elif line.startswith("[MODEL]"):
            model.update(_parse_fields(line[7:]))
        elif line.startswith("[BENCH]"):
            fields = _parse_fields(line[7:])
            points.append(_run_point_from_fields(fields))
        elif line.startswith("[NOTE]"):
            notes.append(line[6:].strip())

    if device:
        system["device"] = DeviceProfile(
            name=str(device.get("name", "unknown")),
            backend=str(device.get("backend", "unknown")),
            cuda=_none_or_str(device.get("cuda")),
            compute_capability=_none_or_str(device.get("compute_capability")),
            vram_bytes=_to_int(device.get("vram_bytes", 0)),
        )
    return RunProfile(
        version=version,
        timestamp=timestamp,
        system=SystemProfile(
            platform=str(system.get("platform", "unknown")),
            python=str(system.get("python", "unknown")),
            torch=str(system.get("torch", "unknown")),
            triton=_none_or_str(system.get("triton")),
            transformers=_none_or_str(system.get("transformers")),
            vllm=_none_or_str(system.get("vllm")),
            ram_bytes=_to_int(system.get("ram_bytes", 0)),
            device=system.get("device", DeviceProfile()),
        ),
        model=ModelProfile(
            model_id=str(model.get("model_id", "")),
            revision=str(model.get("revision", "")),
            layers=_to_int(model.get("layers", 0)),
            query_heads=_to_int(model.get("query_heads", 0)),
            kv_heads=_to_int(model.get("kv_heads", 0)),
            head_dim=_to_int(model.get("head_dim", 0)),
        ),
        points=tuple(points),
        notes=tuple(notes),
    )


def _parse_fields(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in re.finditer(r'([A-Za-z0-9_]+)=(".*?"|\S+)', text):
        key = match.group(1)
        value = match.group(2)
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        out[key] = value
    return out


def _run_point_from_fields(fields: dict[str, str]) -> RunPoint:
    return RunPoint(
        label=str(fields.get("label", "")),
        k_format=str(fields.get("k", "")),
        v_format=str(fields.get("v", "")),
        mode=str(fields.get("mode", "")),
        context_tokens=_to_int(fields.get("context", 0)),
        batch_size=max(1, _to_int(fields.get("batch", 1))),
        tokens_per_s=_to_float(fields.get("tokens_per_s", 0.0)),
        mean_ms=_to_float(fields.get("mean_ms", 0.0)),
        median_ms=_to_float(fields.get("median_ms", 0.0)),
        variance_ms2=_to_float(fields.get("variance_ms2", 0.0)),
        peak_gpu_bytes=_to_int(fields.get("peak_gpu_bytes", 0)),
        env=str(fields.get("env", "")),
    )


def _to_int(value: object) -> int:
    try:
        return int(str(value).replace("_", ""))
    except (TypeError, ValueError):
        return 0


def _to_float(value: object) -> float:
    try:
        return float(str(value).replace("_", ""))
    except (TypeError, ValueError):
        return 0.0


def _none_or_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return None if text.lower() in {"", "none", "null"} else text
