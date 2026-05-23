"""KLD (Kullback-Leibler Divergence) quality metric for attention distributions.

Measures the KL divergence between compressed and reference attention
distributions. Lower KLD indicates better preservation of attention patterns.

Score mapping:
    KLD_score = 100 * exp(-mean_kld)

so 0 nats → 100, 0.7 nats → 50, 1.7 nats → ~18.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class KLDResult:
    """Result of KLD quality evaluation."""
    score: float                 # 0–100
    mean_kld: float              # nats
    ppl: Optional[float] = None
    rms_dp_pct: Optional[float] = None
    same_topp_pct: Optional[float] = None
    is_self_reference: bool = False  # True when candidate == reference; should give 0


def kld_to_score(kld: float) -> float:
    """Map mean KLD (nats) → 0–100 with score = 100 * exp(-kld)."""
    if kld < 0:
        kld = 0.0
    return 100.0 * math.exp(-kld)


def score_to_kld(score: float) -> float:
    """Inverse mapping: score → mean KLD (nats)."""
    if score <= 0:
        return float("inf")
    if score >= 100:
        return 0.0
    return -math.log(score / 100.0)


def compute_attention_kld(
    candidate_scores: torch.Tensor,
    reference_scores: torch.Tensor,
    reduction: str = "mean",
) -> float:
    """Compute KL divergence between attention score distributions.

    Args:
        candidate_scores: Attention logits from compressed cache [*, seq_len].
        reference_scores: Attention logits from reference cache [*, seq_len].
        reduction: "mean", "sum", or "none".

    Returns:
        KLD in nats. Lower is better.
    """
    if candidate_scores.shape != reference_scores.shape:
        raise ValueError(f"shape mismatch: {candidate_scores.shape} vs {reference_scores.shape}")

    log_cand = F.log_softmax(candidate_scores.float(), dim=-1)
    ref = F.softmax(reference_scores.float(), dim=-1)

    kld = F.kl_div(log_cand, ref, reduction=reduction)
    return max(0.0, float(kld.item() if kld.numel() == 1 else kld.mean().item()))


def compute_trajectory_kld(
    candidate_logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> KLDResult:
    """Compute per-step KLD across a token trajectory.

    For each token position, measures KLD between the candidate and
    reference next-token distributions.

    Args:
        candidate_logits: [seq_len, vocab_size] or [batch, seq_len, vocab_size]
        reference_logits: Same shape as candidate.

    Returns:
        KLDResult with mean KLD and derived score.
    """
    if candidate_logits.shape != reference_logits.shape:
        raise ValueError(f"shape mismatch: {candidate_logits.shape} vs {reference_logits.shape}")

    if candidate_logits.ndim == 2:
        candidate_logits = candidate_logits.unsqueeze(0)
        reference_logits = reference_logits.unsqueeze(0)

    # Flatten to [N, vocab]
    batch, seq_len, vocab = candidate_logits.shape
    cand_flat = candidate_logits.reshape(-1, vocab).float()
    ref_flat = reference_logits.reshape(-1, vocab).float()

    # Per-position KLD
    log_cand = F.log_softmax(cand_flat, dim=-1)
    ref_probs = F.softmax(ref_flat, dim=-1)

    kld_per_pos = F.kl_div(log_cand, ref_probs, reduction="none").sum(dim=-1)
    mean_kld = float(kld_per_pos.mean().item())

    # Same-top-p metric: what fraction of positions have same top-1 token
    cand_top1 = cand_flat.argmax(dim=-1)
    ref_top1 = ref_flat.argmax(dim=-1)
    same_topp = float((cand_top1 == ref_top1).float().mean().item()) * 100.0

    return KLDResult(
        score=kld_to_score(mean_kld),
        mean_kld=mean_kld,
        same_topp_pct=same_topp,
        is_self_reference=False,
    )


def compute_self_reference_kld(
    logits: torch.Tensor,
    perturbation: float = 1e-6,
) -> KLDResult:
    """Compute KLD with self as reference (should be ~0).

    Useful for validating that the metric pipeline works correctly.
    """
    perturbed = logits + torch.randn_like(logits) * perturbation
    return compute_trajectory_kld(perturbed, logits)


def compute_rms_delta_probability(
    candidate_logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> float:
    """Compute RMS of probability deltas (percentage).

    Measures the root-mean-square difference in probabilities
    between candidate and reference distributions.

    Returns:
        RMS delta probability as a percentage (0-100).
    """
    cand_probs = F.softmax(candidate_logits.float(), dim=-1)
    ref_probs = F.softmax(reference_logits.float(), dim=-1)

    delta = (cand_probs - ref_probs).pow(2)
    rms = delta.mean().sqrt()
    return float(rms.item()) * 100.0


class KLDTracker:
    """Online KLD tracker for streaming evaluation."""

    def __init__(self):
        self.kld_sum: float = 0.0
        self.count: int = 0
        self.top1_matches: int = 0

    def update(
        self,
        candidate_scores: torch.Tensor,
        reference_scores: torch.Tensor,
    ) -> None:
        """Add a batch of attention score pairs."""
        if candidate_scores.shape != reference_scores.shape:
            raise ValueError("shape mismatch")

        kld = compute_attention_kld(candidate_scores, reference_scores, reduction="sum")
        n = candidate_scores.numel() // candidate_scores.shape[-1]

        self.kld_sum += kld
        self.count += n

        # Track top-1 matches
        cand_top1 = candidate_scores.argmax(dim=-1)
        ref_top1 = reference_scores.argmax(dim=-1)
        self.top1_matches += int((cand_top1 == ref_top1).sum().item())

    def result(self) -> KLDResult:
        """Compute final KLD result."""
        if self.count == 0:
            return KLDResult(score=100.0, mean_kld=0.0)

        mean_kld = self.kld_sum / self.count
        same_topp_pct = (self.top1_matches / self.count) * 100.0 if self.count > 0 else 100.0

        return KLDResult(
            score=kld_to_score(mean_kld),
            mean_kld=mean_kld,
            same_topp_pct=same_topp_pct,
        )

    def reset(self) -> None:
        """Reset tracker state."""
        self.kld_sum = 0.0
        self.count = 0
        self.top1_matches = 0
