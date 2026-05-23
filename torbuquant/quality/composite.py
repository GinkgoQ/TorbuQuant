"""Composite quality scoring for reference/candidate checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Mapping

from torbuquant.quality.trajectory import harmonic_mean, score_band


@dataclass(frozen=True)
class QualityScore:
    score: float
    band: str
    axes: dict[str, float]
    floor_score: float | None = None
    floor_ok: bool | None = None
    floor_min: float = 99.5
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compose_quality_score(
    axes: Mapping[str, float | None],
    *,
    floor_score: float | None = None,
    floor_min: float = 99.5,
) -> QualityScore:
    measured = {name: float(value) for name, value in axes.items() if value is not None}
    score = harmonic_mean(list(measured.values())) if measured else 0.0
    floor_ok = None if floor_score is None else floor_score >= floor_min
    notes = quality_diagnosis(measured)
    if floor_ok is False:
        notes.append(f"Reference floor {floor_score:.2f} is below {floor_min:.2f}")
    return QualityScore(
        score=score,
        band=score_band(score),
        axes=measured,
        floor_score=floor_score,
        floor_ok=floor_ok,
        floor_min=floor_min,
        notes=notes,
    )


def quality_diagnosis(axes: Mapping[str, float]) -> list[str]:
    if not axes:
        return ["No measured axes were provided"]
    low = {name: value for name, value in axes.items() if value < 80.0}
    if not low:
        return ["All measured axes track the reference on tested surfaces"]
    if axes and all(value < 60.0 for value in axes.values()):
        return ["All measured axes are below the mismatch band; use a higher-precision KV configuration"]

    notes: list[str] = []
    short_context = any(name in low for name in ("trajectory", "kl"))
    long_context = "retrieval" in low
    prompt_drift = "perturbation" in low
    if (
        axes.get("trajectory", 100.0) < 60.0
        and axes.get("kl", 100.0) < 60.0
        and not long_context
        and not prompt_drift
    ):
        notes.append(
            "Token distribution changed while long-context retrieval stayed in band; inspect K format, V format, and rotation policy"
        )
        return notes
    if short_context:
        notes.append("Short-context decode drift is above the acceptance band")
    if long_context:
        notes.append("Long-context retrieval falls behind the reference")
    if prompt_drift:
        notes.append("Prompt perturbations cause extra output drift")
    unknown = [name for name in low if name not in {"trajectory", "kl", "retrieval", "perturbation"}]
    if unknown:
        notes.append("Other axes are below the acceptance band: " + ", ".join(sorted(unknown)))
    return notes


def quality_notes(axes: Mapping[str, float]) -> list[str]:
    return quality_diagnosis(axes)


def quality_report_dict(
    *,
    model: str,
    reference_label: str,
    candidate_label: str,
    score: QualityScore,
    extras: Mapping[str, object] | None = None,
) -> dict[str, object]:
    report: dict[str, object] = {
        "schema": "torbuquant.quality.v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "reference": reference_label,
        "candidate": candidate_label,
        "score": score.to_dict(),
    }
    if extras:
        report["extras"] = dict(extras)
    return report


def quality_report_text(
    *,
    model: str,
    reference_label: str,
    candidate_label: str,
    score: QualityScore,
) -> str:
    lines = [
        "TurboQuant Quality Report",
        f"model: {model}",
        f"reference: {reference_label}",
        f"candidate: {candidate_label}",
        f"score: {score.score:.2f}",
        f"band: {score.band}",
    ]
    if score.floor_score is not None:
        lines.append(f"reference_floor: {score.floor_score:.2f}")
    for name, value in sorted(score.axes.items()):
        lines.append(f"axis.{name}: {value:.2f} ({score_band(value)})")
    for note in score.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines)
