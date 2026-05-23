"""Prompt-set loading helpers for reference/candidate evaluations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class PromptCase:
    prompt_id: str
    category: str
    prompt: str
    license: str | None = None
    metadata: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        if data["metadata"] is None:
            data.pop("metadata")
        if data["license"] is None:
            data.pop("license")
        return data


def load_prompt_jsonl(path: str | Path) -> list[PromptCase]:
    source = Path(path)
    rows: list[PromptCase] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_no}: invalid JSON") from exc
            rows.append(prompt_case_from_mapping(payload, source=source, line_no=line_no))
    if not rows:
        raise ValueError(f"{source}: no prompts were loaded")
    return rows


def prompt_case_from_mapping(
    payload: dict[str, object],
    *,
    source: Path | None = None,
    line_no: int | None = None,
) -> PromptCase:
    prefix = "" if source is None else f"{source}:{line_no}: "
    prompt_id = payload.get("id", payload.get("prompt_id"))
    category = payload.get("category", "uncategorized")
    prompt = payload.get("prompt")
    if not isinstance(prompt_id, str) or not prompt_id.strip():
        raise ValueError(prefix + "prompt id must be a non-empty string")
    if not isinstance(category, str) or not category.strip():
        raise ValueError(prefix + "category must be a non-empty string")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(prefix + "prompt must be a non-empty string")
    license_value = payload.get("license")
    if license_value is not None and not isinstance(license_value, str):
        raise ValueError(prefix + "license must be a string when present")
    extra = {
        key: value
        for key, value in payload.items()
        if key not in {"id", "prompt_id", "category", "prompt", "license"}
    }
    return PromptCase(
        prompt_id=prompt_id,
        category=category,
        prompt=prompt,
        license=license_value,
        metadata=extra or None,
    )


def filter_prompt_cases(
    prompts: Sequence[PromptCase],
    *,
    categories: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[PromptCase]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    wanted = None if categories is None else {category for category in categories}
    rows = [case for case in prompts if wanted is None or case.category in wanted]
    if limit is not None:
        rows = rows[:limit]
    return rows


def group_prompt_cases(prompts: Sequence[PromptCase]) -> dict[str, list[PromptCase]]:
    grouped: dict[str, list[PromptCase]] = defaultdict(list)
    for case in prompts:
        grouped[case.category].append(case)
    return dict(grouped)


def prompt_ids(prompts: Sequence[PromptCase]) -> list[str]:
    return [case.prompt_id for case in prompts]


def prompt_texts(prompts: Sequence[PromptCase]) -> list[str]:
    return [case.prompt for case in prompts]
