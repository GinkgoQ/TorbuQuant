"""Needle-style retrieval prompt helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
import random
import re

FILLER_PARAGRAPHS: tuple[str, ...] = (
    "The city archives contain meeting notes about transit planning, library budgets, and seasonal repairs.",
    "Researchers cataloged mineral samples in labeled trays before recording their color, weight, and texture.",
    "A regional weather log described fog, rainfall, pressure changes, and wind direction across several weeks.",
    "The museum guide explained how textile dyes were prepared, stored, and tested by workshop apprentices.",
    "Engineers reviewed bridge inspection records with photographs, measurements, and maintenance annotations.",
    "The community garden report listed soil readings, compost batches, irrigation checks, and harvest weights.",
    "A shipping ledger tracked crate identifiers, warehouse doors, arrival times, and inspection signatures.",
    "The classroom schedule included reading groups, arithmetic drills, science demonstrations, and quiet study.",
    "A mountain trail note warned hikers about loose gravel, shaded switchbacks, creek crossings, and markers.",
    "The bakery notebook compared flour blends, proofing durations, oven temperatures, and customer feedback.",
    "An observatory entry summarized telescope alignment, cloud cover, exposure time, and calibration frames.",
    "The hospital inventory listed gloves, masks, saline bags, batteries, clipboards, and storage locations.",
    "A theater production log tracked lighting cues, costume repairs, prop placement, and rehearsal timing.",
    "The fisheries survey recorded water clarity, net depth, species counts, and boat positions.",
    "A farm equipment checklist mentioned tire pressure, fuel levels, blade wear, and spare parts.",
    "The county clerk indexed land records by parcel number, street name, survey note, and filing date.",
    "A software release note described input validation, telemetry fields, configuration parsing, and tests.",
    "The art studio journal compared canvas sizes, brush materials, pigment batches, and drying times.",
    "A rail depot form listed platform repairs, signal checks, ticket counters, and night shift notes.",
    "The recipe archive grouped soups, breads, sauces, salads, desserts, and preserving methods.",
    "A coastal survey described tide pools, erosion markers, dune grass, shell fragments, and access paths.",
    "The workshop manual covered tool storage, warning labels, torque values, and replacement intervals.",
    "A bookstore inventory arranged titles by author, shelf row, binding type, and order status.",
    "The orchestra librarian marked scores by movement, instrument section, bowing note, and rehearsal date.",
)

DISTRACTOR_KEYS: tuple[str, ...] = (
    "The secret password is",
    "The hidden code is",
    "The encrypted token is",
    "The backup reference is",
    "The archive marker is",
    "The control number is",
)


@dataclass(frozen=True)
class NeedleCase:
    context: str
    question: str
    answer: str
    depth: float
    context_tokens_estimate: int

    def to_dict(self) -> dict[str, str | float | int]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalCell:
    context_tokens: int
    depth: float
    reference_hits: int
    candidate_hits: int
    trials: int

    @property
    def reference_rate(self) -> float:
        return self.reference_hits / self.trials

    @property
    def candidate_rate(self) -> float:
        return self.candidate_hits / self.trials

    @property
    def loss(self) -> float:
        return max(0.0, self.reference_rate - self.candidate_rate)

    def to_dict(self) -> dict[str, float | int]:
        data = asdict(self)
        data.update(
            {
                "reference_rate": self.reference_rate,
                "candidate_rate": self.candidate_rate,
                "loss": self.loss,
            }
        )
        return data


@dataclass(frozen=True)
class RetrievalMatrix:
    score: float
    cells: tuple[RetrievalCell, ...]
    skipped: tuple[tuple[int, float], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "cells": [cell.to_dict() for cell in self.cells],
            "skipped": list(self.skipped),
        }


@dataclass(frozen=True)
class RetrievalNeedle:
    key: str
    value: str
    depth: float

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("key must be non-empty")
        if not self.value:
            raise ValueError("value must be non-empty")
        if not 0.0 <= self.depth <= 1.0:
            raise ValueError("depth must be between 0 and 1")

    @property
    def sentence(self) -> str:
        return f"{self.key}: {self.value}."

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalTrial:
    expected: str
    response: str
    found: bool
    depth: float = 0.0
    context_tokens: int = 0

    def to_dict(self) -> dict[str, str | bool | float | int]:
        return asdict(self)


@dataclass
class RetrievalConfigResult:
    mode: str
    context_tokens: int
    cache_format: str
    depth: float = 0.5
    needle_count: int = 1
    trials: list[RetrievalTrial] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        if not self.trials:
            return 0.0
        return 100.0 * sum(1 for trial in self.trials if trial.found) / len(self.trials)

    @property
    def passed(self) -> bool:
        return all(trial.found for trial in self.trials)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "context_tokens": self.context_tokens,
            "cache_format": self.cache_format,
            "depth": self.depth,
            "needle_count": self.needle_count,
            "hit_rate": self.hit_rate,
            "passed": self.passed,
            "trials": [trial.to_dict() for trial in self.trials],
        }


def build_needle_case(
    *,
    context_tokens: int,
    depth: float,
    answer: str,
    filler: str = "The archive entry describes a routine calibration note.",
) -> NeedleCase:
    if context_tokens <= 0:
        raise ValueError("context_tokens must be positive")
    if not 0.0 <= depth <= 1.0:
        raise ValueError("depth must be between 0 and 1")
    if not answer:
        raise ValueError("answer must be non-empty")
    filler_words = filler.split()
    if not filler_words:
        raise ValueError("filler must contain words")
    needle = f"The retrieval key is {answer}."
    needle_at = int(context_tokens * depth)
    words: list[str] = []
    while len(words) < context_tokens:
        if len(words) >= needle_at and needle not in " ".join(words[-16:]):
            words.extend(needle.split())
        words.extend(filler_words)
    context = " ".join(words[:context_tokens])
    question = "What is the retrieval key?"
    return NeedleCase(
        context=context,
        question=question,
        answer=answer,
        depth=depth,
        context_tokens_estimate=context_tokens,
    )


def exact_answer_match(generation: str, answer: str) -> bool:
    if not answer:
        raise ValueError("answer must be non-empty")
    return answer.lower() in generation.lower()


def extract_retrieval_target(needle: str) -> str:
    if not needle:
        raise ValueError("needle must be non-empty")
    candidates = re.findall(r"[A-Z][A-Z0-9-]{2,}", needle)
    if candidates:
        return max(candidates, key=len)
    parts = needle.split()
    return parts[-1].rstrip(".:,;!?") if parts else needle


def nearest_sentence_boundary(text: str, target_char: int, *, window: int = 200) -> int:
    if window < 0:
        raise ValueError("window must be non-negative")
    if not text or target_char <= 0:
        return 0
    if target_char >= len(text):
        return len(text)
    lo = max(0, target_char - window)
    hi = min(len(text), target_char + window)
    for delta in range(window + 1):
        for candidate in (target_char - delta, target_char + delta):
            if candidate < lo or candidate + 1 >= hi:
                continue
            if text[candidate] == "." and text[candidate + 1] in (" ", "\n"):
                return candidate + 2
    return target_char


def build_retrieval_prompt(
    *,
    haystack: str,
    needle: str,
    question: str,
    depth: float,
    boundary_window: int = 200,
) -> tuple[str, str]:
    if not haystack:
        raise ValueError("haystack must be non-empty")
    if not needle:
        raise ValueError("needle must be non-empty")
    if not question:
        raise ValueError("question must be non-empty")
    if not 0.0 <= depth <= 1.0:
        raise ValueError("depth must be between 0 and 1")
    target = int(len(haystack) * depth)
    insertion = nearest_sentence_boundary(haystack, target, window=boundary_window)
    context = f"{haystack[:insertion]} {needle} {haystack[insertion:]}"
    return context, question


def retrieval_matrix(
    cells: list[RetrievalCell],
    *,
    skipped: list[tuple[int, float]] | None = None,
) -> RetrievalMatrix:
    skipped = [] if skipped is None else skipped
    if not cells:
        return RetrievalMatrix(score=0.0, cells=tuple(), skipped=tuple(skipped))
    for cell in cells:
        if cell.trials <= 0:
            raise ValueError("cell trials must be positive")
        if not 0 <= cell.reference_hits <= cell.trials:
            raise ValueError("reference hits must be within trial count")
        if not 0 <= cell.candidate_hits <= cell.trials:
            raise ValueError("candidate hits must be within trial count")
    mean_loss = sum(cell.loss for cell in cells) / len(cells)
    return RetrievalMatrix(score=100.0 * (1.0 - mean_loss), cells=tuple(cells), skipped=tuple(skipped))


def make_retrieval_number(rng: random.Random) -> str:
    return str(rng.randint(1_000_000, 9_999_999))


def build_filler_paragraphs(
    target_chars: int,
    rng: random.Random,
    *,
    paragraphs: Sequence[str] = FILLER_PARAGRAPHS,
) -> list[str]:
    if target_chars <= 0:
        raise ValueError("target_chars must be positive")
    if not paragraphs:
        raise ValueError("paragraphs must be non-empty")
    shuffled = list(paragraphs)
    rng.shuffle(shuffled)
    result: list[str] = []
    total = 0
    idx = 0
    while total < target_chars:
        paragraph = shuffled[idx % len(shuffled)]
        result.append(paragraph)
        total += len(paragraph) + 2
        idx += 1
    return result


def insert_retrieval_needles(paragraphs: Sequence[str], needles: Sequence[RetrievalNeedle]) -> str:
    if not paragraphs:
        raise ValueError("paragraphs must be non-empty")
    items = list(paragraphs)
    n_items = len(items)
    insertions: list[tuple[int, str]] = []
    for needle in needles:
        index = int(needle.depth * n_items)
        index = max(0, min(index, n_items))
        insertions.append((index, needle.sentence))
    for index, sentence in sorted(insertions, key=lambda item: item[0], reverse=True):
        items.insert(index, sentence)
    return "\n\n".join(items)


def generate_single_haystack(
    needle: RetrievalNeedle,
    target_chars: int,
    rng: random.Random,
) -> str:
    paragraphs = build_filler_paragraphs(target_chars, rng)
    return insert_retrieval_needles(paragraphs, [needle])


def generate_multi_key_haystack(
    needle: RetrievalNeedle,
    distractors: Sequence[RetrievalNeedle],
    target_chars: int,
    rng: random.Random,
) -> str:
    paragraphs = build_filler_paragraphs(target_chars, rng)
    return insert_retrieval_needles(paragraphs, [needle, *distractors])


def generate_multi_value_haystack(
    needles: Sequence[RetrievalNeedle],
    target_chars: int,
    rng: random.Random,
) -> str:
    if not needles:
        raise ValueError("needles must be non-empty")
    paragraphs = build_filler_paragraphs(target_chars, rng)
    return insert_retrieval_needles(paragraphs, needles)


def make_distractor_needles(
    count: int,
    rng: random.Random,
    *,
    keys: Sequence[str] = DISTRACTOR_KEYS,
) -> list[RetrievalNeedle]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if count and not keys:
        raise ValueError("keys must be non-empty when count is positive")
    needles: list[RetrievalNeedle] = []
    for i in range(count):
        depth = (i + 1) / (count + 2)
        needles.append(RetrievalNeedle(key=keys[i % len(keys)], value=make_retrieval_number(rng), depth=depth))
    return needles


def score_single_number(response: str, expected: str) -> bool:
    if not expected or not re.fullmatch(r"\d{7}", expected):
        raise ValueError("expected must be a 7-digit string")
    match = re.search(r"\b(\d{7})\b", response)
    return match is not None and match.group(1) == expected


def score_multi_numbers(response: str, expected: Sequence[str]) -> list[bool]:
    for item in expected:
        if not item or not re.fullmatch(r"\d{7}", item):
            raise ValueError("expected values must be 7-digit strings")
    found = set(re.findall(r"\b(\d{7})\b", response))
    return [item in found for item in expected]


def retrieval_length_label(context_tokens: int) -> str:
    return f"{context_tokens // 1024}K" if context_tokens >= 1024 and context_tokens % 1024 == 0 else str(context_tokens)


def retrieval_heatmap_table(results: Sequence[RetrievalConfigResult], cache_format: str) -> str:
    selected = [result for result in results if result.cache_format == cache_format]
    if not selected:
        return f"## Single Needle Retrieval: {cache_format}\n\n(no results)\n"
    depths = sorted({int(result.depth * 100) for result in selected})
    lengths = sorted({result.context_tokens for result in selected})
    lookup = {(int(result.depth * 100), result.context_tokens): result.passed for result in selected}
    header = "| Depth |" + "".join(f" {retrieval_length_label(length):<5}|" for length in lengths)
    sep = "|-------|" + "------|" * len(lengths)
    lines = [f"## Single Needle Retrieval: {cache_format}", "", header, sep]
    for depth in depths:
        row = f"| {depth:<5}%|"
        for length in lengths:
            value = lookup.get((depth, length))
            cell = " hit " if value is True else " miss" if value is False else " n/a "
            row += f"{cell}|"
        lines.append(row)
    return "\n".join(lines)


def retrieval_delta_table(
    results: Sequence[RetrievalConfigResult],
    reference_format: str,
    candidate_format: str,
) -> str:
    reference = {
        (int(result.depth * 100), result.context_tokens): result.passed
        for result in results
        if result.cache_format == reference_format
    }
    candidate = {
        (int(result.depth * 100), result.context_tokens): result.passed
        for result in results
        if result.cache_format == candidate_format
    }
    if not reference or not candidate:
        return ""
    depths = sorted({key[0] for key in reference})
    lengths = sorted({key[1] for key in reference})
    header = "| Depth |" + "".join(f" {retrieval_length_label(length):<5}|" for length in lengths)
    sep = "|-------|" + "------|" * len(lengths)
    lines = [f"## Delta: {candidate_format} vs {reference_format}", "", header, sep]
    changed = False
    for depth in depths:
        row = f"| {depth:<5}%|"
        for length in lengths:
            ref_value = reference.get((depth, length))
            cand_value = candidate.get((depth, length))
            if ref_value is None or cand_value is None:
                cell = " n/a "
            elif ref_value == cand_value:
                cell = " same"
            elif cand_value and not ref_value:
                cell = " gain"
                changed = True
            else:
                cell = " loss"
                changed = True
            row += f"{cell}|"
        lines.append(row)
    if not changed:
        lines.extend(["", "No differences detected."])
    return "\n".join(lines)


def retrieval_multi_key_table(results: Sequence[RetrievalConfigResult]) -> str:
    if not results:
        return "## Multi-Key Retrieval\n\n(no results)\n"
    formats = sorted({result.cache_format for result in results})
    lengths = sorted({result.context_tokens for result in results})
    lookup = {(result.cache_format, result.context_tokens): result.passed for result in results}
    header = "| Cache |" + "".join(f" {retrieval_length_label(length):<5}|" for length in lengths)
    sep = "|-------|" + "------|" * len(lengths)
    lines = ["## Multi-Key Retrieval", "", header, sep]
    for cache_format in formats:
        row = f"| {cache_format:<5} |"
        for length in lengths:
            value = lookup.get((cache_format, length))
            cell = " hit " if value is True else " miss" if value is False else " n/a "
            row += f"{cell}|"
        lines.append(row)
    return "\n".join(lines)


def retrieval_multi_value_table(results: Sequence[RetrievalConfigResult]) -> str:
    if not results:
        return "## Multi-Value Retrieval\n\n(no results)\n"
    formats = sorted({result.cache_format for result in results})
    lengths = sorted({result.context_tokens for result in results})
    counts = sorted({result.needle_count for result in results})
    lines = ["## Multi-Value Retrieval", ""]
    for cache_format in formats:
        header = "| Values |" + "".join(f" {retrieval_length_label(length):<7}|" for length in lengths)
        sep = "|--------|" + "--------|" * len(lengths)
        lines.extend([f"### {cache_format}", "", header, sep])
        lookup = {
            (result.needle_count, result.context_tokens): result.hit_rate
            for result in results
            if result.cache_format == cache_format
        }
        for count in counts:
            row = f"| {count:<6} |"
            for length in lengths:
                rate = lookup.get((count, length))
                cell = " n/a   " if rate is None else f" {rate:5.1f}%"
                row += f"{cell}|"
            lines.append(row)
        lines.append("")
    return "\n".join(lines).rstrip()
