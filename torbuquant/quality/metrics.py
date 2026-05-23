"""Quality and timing metrics for compressed attention validation."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TensorError:
    max_abs: float
    rmse: float
    cosine: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class LogitMetrics:
    kl_divergence: float
    argmax_match: float
    topk_overlap: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class TimingStats:
    runs: int
    mean_ms: float
    median_ms: float
    variance_ms2: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def tensor_error(candidate: torch.Tensor, reference: torch.Tensor) -> TensorError:
    if candidate.shape != reference.shape:
        raise ValueError(f"shape mismatch: {candidate.shape} vs {reference.shape}")
    cand = candidate.float().reshape(-1)
    ref = reference.float().reshape(-1)
    diff = cand - ref
    denom = cand.norm() * ref.norm()
    cosine = 1.0 if denom.item() == 0.0 else float(torch.dot(cand, ref).div(denom).item())
    return TensorError(
        max_abs=float(diff.abs().max().item()) if diff.numel() else 0.0,
        rmse=float(torch.sqrt(torch.mean(diff.square())).item()) if diff.numel() else 0.0,
        cosine=cosine,
    )


def distribution_kl(candidate_scores: torch.Tensor, reference_scores: torch.Tensor) -> float:
    if candidate_scores.shape != reference_scores.shape:
        raise ValueError(f"shape mismatch: {candidate_scores.shape} vs {reference_scores.shape}")
    log_cand = F.log_softmax(candidate_scores.float(), dim=-1)
    ref = F.softmax(reference_scores.float(), dim=-1)
    return max(0.0, float(F.kl_div(log_cand, ref, reduction="batchmean").item()))


def logits_metrics(candidate_logits: torch.Tensor, reference_logits: torch.Tensor, *, topk: int = 5) -> LogitMetrics:
    if candidate_logits.shape != reference_logits.shape:
        raise ValueError(f"shape mismatch: {candidate_logits.shape} vs {reference_logits.shape}")
    if candidate_logits.ndim < 2:
        raise ValueError("logits must have a vocabulary dimension")
    vocab = int(candidate_logits.shape[-1])
    if not 1 <= topk <= vocab:
        raise ValueError(f"topk must be in [1, {vocab}], got {topk}")
    flat_cand = candidate_logits.reshape(-1, vocab).float()
    flat_ref = reference_logits.reshape(-1, vocab).float()
    kl = float(F.kl_div(F.log_softmax(flat_cand, dim=-1), F.softmax(flat_ref, dim=-1), reduction="batchmean").item())
    argmax_match = float((flat_cand.argmax(dim=-1) == flat_ref.argmax(dim=-1)).float().mean().item())
    cand_top = flat_cand.topk(topk, dim=-1).indices
    ref_top = flat_ref.topk(topk, dim=-1).indices
    overlap = []
    for cand_row, ref_row in zip(cand_top, ref_top):
        overlap.append(len(set(cand_row.tolist()).intersection(ref_row.tolist())) / topk)
    return LogitMetrics(
        kl_divergence=kl,
        argmax_match=argmax_match,
        topk_overlap=float(sum(overlap) / len(overlap)) if overlap else 1.0,
    )


def timing_stats(milliseconds: list[float]) -> TimingStats:
    if not milliseconds:
        raise ValueError("timing list is empty")
    variance = statistics.variance(milliseconds) if len(milliseconds) > 1 else 0.0
    return TimingStats(
        runs=len(milliseconds),
        mean_ms=float(statistics.mean(milliseconds)),
        median_ms=float(statistics.median(milliseconds)),
        variance_ms2=float(variance),
        min_ms=float(min(milliseconds)),
        max_ms=float(max(milliseconds)),
    )


def cuda_time_ms(fn, *, warmup: int = 5, repeats: int = 20) -> tuple[Any, TimingStats]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA timing requires torch.cuda")
    result = None
    for _ in range(warmup):
        result = fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return result, timing_stats(times)


def capacity_from_bytes(
    *,
    memory_budget_bytes: int,
    bytes_per_token: float,
    batch_size: int = 1,
) -> int:
    if memory_budget_bytes <= 0:
        raise ValueError("memory_budget_bytes must be positive")
    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return int(math.floor(memory_budget_bytes / (bytes_per_token * batch_size)))
