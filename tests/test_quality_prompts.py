from pathlib import Path

import pytest

from torbuquant.quality import (
    PromptCase,
    filter_prompt_cases,
    group_prompt_cases,
    load_prompt_jsonl,
    prompt_case_from_mapping,
    prompt_ids,
    prompt_texts,
)


PROMPTS_PATH = Path("other_implemnetations/turboquant_plus/refract/prompts/v0.1.jsonl")


def test_load_prompt_jsonl_reads_refract_prompt_set():
    prompts = load_prompt_jsonl(PROMPTS_PATH)

    assert len(prompts) == 30
    assert prompts[0] == PromptCase(
        prompt_id="fact-001",
        category="factual",
        prompt="The capital of France is",
        license="CC0",
        metadata=None,
    )
    assert "dialogue" in group_prompt_cases(prompts)
    assert prompt_ids(prompts[:2]) == ["fact-001", "fact-002"]
    assert prompt_texts(prompts[:1]) == ["The capital of France is"]


def test_filter_prompt_cases_by_category_and_limit():
    prompts = load_prompt_jsonl(PROMPTS_PATH)

    filtered = filter_prompt_cases(prompts, categories=["code", "reasoning"], limit=3)

    assert len(filtered) == 3
    assert all(case.category in {"code", "reasoning"} for case in filtered)


def test_prompt_case_from_mapping_keeps_extra_fields():
    case = prompt_case_from_mapping(
        {
            "prompt_id": "x-1",
            "category": "small",
            "prompt": "Return the number",
            "source": "unit",
        }
    )

    assert case.prompt_id == "x-1"
    assert case.metadata == {"source": "unit"}
    assert case.to_dict()["metadata"] == {"source": "unit"}


def test_load_prompt_jsonl_reports_bad_lines(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"id": "x", "category": "c"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="prompt must be a non-empty string"):
        load_prompt_jsonl(path)


def test_filter_prompt_cases_rejects_negative_limit():
    with pytest.raises(ValueError):
        filter_prompt_cases([], limit=-1)
