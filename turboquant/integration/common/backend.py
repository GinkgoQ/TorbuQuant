"""Backend capability and version detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal

import torch

from turboquant.kv.policy import BackendCapability
from turboquant.triton import triton_available

BackendName = Literal["diagnostic", "hf", "vllm"]


@dataclass(frozen=True)
class RuntimeVersions:
    python: str
    torch: str
    cuda_available: bool
    cuda: str | None
    gpu: str | None
    triton: str | None
    transformers: str | None
    vllm: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IntegrationCapability:
    backend: BackendName
    available: bool
    reason: str | None
    policy_capability: BackendCapability
    versions: RuntimeVersions

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["policy_capability"] = self.policy_capability.__dict__
        data["versions"] = self.versions.to_dict()
        return data


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def runtime_versions() -> RuntimeVersions:
    import sys

    cuda_available = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if cuda_available else None
    return RuntimeVersions(
        python=sys.version.split()[0],
        torch=torch.__version__,
        cuda_available=cuda_available,
        cuda=torch.version.cuda,
        gpu=gpu,
        triton=_package_version("triton"),
        transformers=_package_version("transformers"),
        vllm=_package_version("vllm"),
    )


def has_module(name: str) -> bool:
    try:
        import_module(name)
        return True
    except Exception:
        return False


def detect_backend(backend: BackendName) -> IntegrationCapability:
    versions = runtime_versions()
    if backend == "diagnostic":
        return IntegrationCapability(
            backend=backend,
            available=True,
            reason=None,
            policy_capability=BackendCapability("diagnostic", supports_sparse_v=True),
            versions=versions,
        )

    if backend == "hf":
        available = has_module("transformers")
        return IntegrationCapability(
            backend=backend,
            available=available,
            reason=None if available else "transformers is not importable",
            policy_capability=BackendCapability(
                "hf",
                fused_decode=triton_available(),
                paged_cache=False,
                supports_v3=False,
                supports_sparse_v=False,
            ),
            versions=versions,
        )

    if backend == "vllm":
        available = has_module("vllm")
        return IntegrationCapability(
            backend=backend,
            available=available,
            reason=None if available else "vllm is not importable",
            policy_capability=BackendCapability(
                "vllm",
                fused_decode=triton_available(),
                paged_cache=True,
                supports_v3=False,
                supports_sparse_v=False,
            ),
            versions=versions,
        )

    raise ValueError(f"unknown backend: {backend}")
