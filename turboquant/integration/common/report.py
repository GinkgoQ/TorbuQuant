"""Integration report payloads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from turboquant.integration.common.backend import IntegrationCapability
from turboquant.integration.common.config import KVQuantConfig


@dataclass(frozen=True)
class IntegrationReport:
    backend: str
    mode: str
    model_id: str
    preset: str
    k_format: str
    v_format: str
    capability: dict[str, Any]
    config: dict[str, Any]
    memory: dict[str, Any] | None = None
    cache: dict[str, Any] | None = None
    kernel: dict[str, Any] | None = None
    fallback_count: int = 0
    fallback_mode: str | None = None
    diagnostic_label: str | None = None
    command: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def integration_report(
    *,
    config: KVQuantConfig,
    capability: IntegrationCapability,
    k_format: str,
    v_format: str,
    memory: dict[str, Any] | None = None,
    cache: dict[str, Any] | None = None,
    kernel: dict[str, Any] | None = None,
    fallback_count: int = 0,
    fallback_mode: str | None = None,
    diagnostic_label: str | None = None,
    command: str | None = None,
) -> IntegrationReport:
    return IntegrationReport(
        backend=config.backend,
        mode=config.mode,
        model_id=config.model_id,
        preset=config.preset,
        k_format=k_format,
        v_format=v_format,
        capability=capability.to_dict(),
        config=config.to_dict(),
        memory=memory,
        cache=cache,
        kernel=kernel,
        fallback_count=fallback_count,
        fallback_mode=fallback_mode,
        diagnostic_label=diagnostic_label,
        command=command,
    )

