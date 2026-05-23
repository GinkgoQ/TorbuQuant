"""Token-trajectory and perturbation metrics for generation checks."""

from __future__ import annotations

import math
import random
import re
import statistics
from dataclasses import asdict, dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class SequenceMatch:
    first_divergence: int | None
    prefix_agreement_length: int
    reference_length: int
    candidate_length: int
    matched: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryMetrics:
    score: float
    full_match_rate: float
    median_first_divergence: float | None
    mean_prefix_agreement_length: float
    mean_candidate_length: float
    mean_reference_length: float
    per_sequence: list[SequenceMatch]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["per_sequence"] = [row.to_dict() for row in self.per_sequence]
        return data


@dataclass(frozen=True)
class PerturbationRecord:
    prompt_id: str
    perturbation: str
    perturbed_prompt: str
    reference_drift: float
    candidate_drift: float
    excess_drift: float
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PerturbationMetrics:
    score: float
    per_perturbation_score: dict[str, float]
    records: list[PerturbationRecord]
    skipped: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["records"] = [row.to_dict() for row in self.records]
        return data


def sequence_match(reference: Sequence[int], candidate: Sequence[int]) -> SequenceMatch:
    """Compare two token-id sequences in model-token units."""
    n = min(len(reference), len(candidate))
    for idx in range(n):
        if reference[idx] != candidate[idx]:
            return SequenceMatch(idx, idx, len(reference), len(candidate), False)
    if len(reference) == len(candidate):
        return SequenceMatch(None, n, len(reference), len(candidate), True)
    return SequenceMatch(n, n, len(reference), len(candidate), False)


def trajectory_metrics(
    references: Sequence[Sequence[int]],
    candidates: Sequence[Sequence[int]],
    *,
    expected_tokens: int | None = None,
) -> TrajectoryMetrics:
    """Compute prefix trajectory agreement over paired token-id sequences."""
    if len(references) != len(candidates):
        raise ValueError("references and candidates must have the same length")
    if not references:
        raise ValueError("at least one sequence pair is required")

    rows = [sequence_match(ref, cand) for ref, cand in zip(references, candidates)]
    mean_prefix = statistics.mean(row.prefix_agreement_length for row in rows)
    mean_candidate = statistics.mean(row.candidate_length for row in rows)
    mean_reference = statistics.mean(row.reference_length for row in rows)
    divergences = [row.first_divergence for row in rows if row.first_divergence is not None]
    score = 100.0 * mean_prefix / mean_candidate if mean_candidate > 0 else 0.0
    score = min(100.0, max(0.0, score))
    notes: list[str] = []
    if expected_tokens is not None:
        short = sum(1 for row in rows if row.candidate_length < expected_tokens)
        if short:
            notes.append(f"{short}/{len(rows)} candidates ended before {expected_tokens} generated tokens")
    return TrajectoryMetrics(
        score=score,
        full_match_rate=sum(1 for row in rows if row.matched) / len(rows),
        median_first_divergence=statistics.median(divergences) if divergences else None,
        mean_prefix_agreement_length=mean_prefix,
        mean_candidate_length=mean_candidate,
        mean_reference_length=mean_reference,
        per_sequence=rows,
        notes=notes,
    )


def levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    """Levenshtein distance over token ids."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for idx_a, val_a in enumerate(a, 1):
        cur = [idx_a] + [0] * len(b)
        for idx_b, val_b in enumerate(b, 1):
            cost = 0 if val_a == val_b else 1
            cur[idx_b] = min(prev[idx_b] + 1, cur[idx_b - 1] + 1, prev[idx_b - 1] + cost)
        prev = cur
    return prev[-1]


def normalized_drift(anchor: Sequence[int], perturbed: Sequence[int]) -> float:
    """Token edit distance divided by anchor length, clipped to [0, 1]."""
    if not anchor and not perturbed:
        return 0.0
    if not anchor:
        return 1.0
    return min(1.0, levenshtein(anchor, perturbed) / len(anchor))


_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "of", "for", "with", "by", "as", "and", "or",
    "but", "not", "no", "do", "does", "did", "i", "you", "he", "she", "it",
    "we", "they", "this", "that", "these", "those", "have", "has", "had",
}
_SYNONYMS = {
    "big": "large",
    "large": "big",
    "small": "tiny",
    "tiny": "small",
    "begin": "start",
    "start": "begin",
    "happy": "glad",
    "sad": "unhappy",
    "smart": "clever",
    "clever": "smart",
    "show": "display",
    "display": "show",
    "build": "construct",
    "create": "make",
    "make": "create",
    "find": "locate",
    "locate": "find",
}


def eligible_words(prompt: str) -> list[tuple[int, int, str]]:
    """Return non-stopword word spans from a prompt."""
    rows: list[tuple[int, int, str]] = []
    for match in _WORD_RE.finditer(prompt):
        word = match.group(0)
        if word.lower() not in _STOPWORDS:
            rows.append((match.start(), match.end(), word))
    return rows


def perturb_prompt(prompt: str, kind: str, *, seed: int = 42) -> str | None:
    """Apply one deterministic local prompt perturbation."""
    rng = random.Random(seed)
    if kind == "typo":
        rows = [(s, e, w) for s, e, w in eligible_words(prompt) if len(w) >= 4]
        if not rows:
            return None
        start, end, word = rng.choice(rows)
        pos = rng.randrange(0, len(word) - 1)
        swapped = word[:pos] + word[pos + 1] + word[pos] + word[pos + 2:]
        if swapped == word:
            return None
        return prompt[:start] + swapped + prompt[end:]
    if kind == "case":
        chunks: list[str] = []
        cursor = 0
        changed = False
        for match in _WORD_RE.finditer(prompt):
            chunks.append(prompt[cursor:match.start()])
            word = match.group(0)
            if word[0].isupper():
                chunks.append(word[0].lower() + word[1:])
                changed = True
            else:
                chunks.append(word)
            cursor = match.end()
        chunks.append(prompt[cursor:])
        return "".join(chunks) if changed else None
    if kind == "punct":
        stripped = prompt.rstrip()
        tail = prompt[len(stripped):]
        if stripped.endswith("?") or stripped.endswith("."):
            return stripped[:-1] + tail
        return stripped + "?" + tail
    if kind == "paraphrase":
        rows = [
            (s, e, w) for s, e, w in eligible_words(prompt)
            if len(w) >= 3 and w.lower() in _SYNONYMS
        ]
        if not rows:
            return None
        start, end, word = rng.choice(rows)
        sub = _SYNONYMS[word.lower()]
        if word[0].isupper():
            sub = sub[0].upper() + sub[1:]
        return prompt[:start] + sub + prompt[end:]
    raise ValueError(f"unknown perturbation kind: {kind}")


def perturbation_metrics(
    *,
    prompt_ids: Sequence[str],
    prompts: Sequence[str],
    reference_anchor: Sequence[Sequence[int]],
    candidate_anchor: Sequence[Sequence[int]],
    reference_perturbed: dict[tuple[str, str], Sequence[int]],
    candidate_perturbed: dict[tuple[str, str], Sequence[int]],
    perturbations: Sequence[str] = ("typo", "case", "punct", "paraphrase"),
    alpha: float = 5.0,
    seed: int = 42,
) -> PerturbationMetrics:
    """Score excess token drift under small prompt changes."""
    if not (len(prompt_ids) == len(prompts) == len(reference_anchor) == len(candidate_anchor)):
        raise ValueError("prompt ids, prompts, and anchor outputs must have matching lengths")
    records: list[PerturbationRecord] = []
    per_kind: dict[str, list[float]] = {kind: [] for kind in perturbations}
    skipped = 0
    for idx, prompt_id in enumerate(prompt_ids):
        prompt = prompts[idx]
        for kind in perturbations:
            perturbed_prompt = perturb_prompt(prompt, kind, seed=seed + idx)
            if perturbed_prompt is None:
                skipped += 1
                continue
            key = (prompt_id, kind)
            if key not in reference_perturbed or key not in candidate_perturbed:
                skipped += 1
                continue
            ref_drift = normalized_drift(reference_anchor[idx], reference_perturbed[key])
            cand_drift = normalized_drift(candidate_anchor[idx], candidate_perturbed[key])
            excess = max(0.0, cand_drift - ref_drift)
            score = 100.0 * math.exp(-alpha * excess)
            records.append(
                PerturbationRecord(
                    prompt_id=prompt_id,
                    perturbation=kind,
                    perturbed_prompt=perturbed_prompt,
                    reference_drift=ref_drift,
                    candidate_drift=cand_drift,
                    excess_drift=excess,
                    score=score,
                )
            )
            per_kind[kind].append(score)
    if not records:
        raise ValueError("no perturbation records were scored")
    per_summary = {
        kind: (statistics.mean(scores) if scores else float("nan"))
        for kind, scores in per_kind.items()
    }
    notes = [f"{skipped} perturbation cells were skipped"] if skipped else []
    return PerturbationMetrics(
        score=statistics.mean(row.score for row in records),
        per_perturbation_score=per_summary,
        records=records,
        skipped=skipped,
        notes=notes,
    )


def harmonic_mean(values: Sequence[float]) -> float:
    """Harmonic mean clipped to [0, 100]."""
    clean = [max(0.0, float(value)) for value in values]
    if not clean or any(value <= 0.0 for value in clean):
        return 0.0
    return min(100.0, max(0.0, len(clean) / sum(1.0 / value for value in clean)))


def score_band(score: float) -> str:
    """Map a 0-100 score to a coarse report band."""
    if score >= 90.0:
        return "match"
    if score >= 80.0:
        return "minor_drift"
    if score >= 60.0:
        return "drift"
    return "mismatch"
