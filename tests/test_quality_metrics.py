import pytest
import torch

from torbuquant.quality import (
    build_needle_case,
    build_retrieval_prompt,
    capacity_from_bytes,
    compose_quality_score,
    distribution_kl,
    exact_answer_match,
    extract_retrieval_target,
    logits_metrics,
    nearest_sentence_boundary,
    quality_diagnosis,
    quality_report_dict,
    RetrievalCell,
    retrieval_matrix,
    quality_report_text,
    tensor_error,
    timing_stats,
)


def test_tensor_error_reports_zero_for_identical_tensors():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    err = tensor_error(x, x)
    assert err.max_abs == 0.0
    assert err.rmse == 0.0
    assert err.cosine == pytest.approx(1.0)


def test_distribution_kl_is_zero_for_equal_scores():
    scores = torch.tensor([[[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]]])
    assert distribution_kl(scores, scores) == pytest.approx(0.0, abs=1e-7)


def test_logits_metrics_argmax_and_topk_overlap():
    ref = torch.tensor([[0.0, 3.0, 2.0, 1.0]])
    cand = torch.tensor([[0.0, 2.5, 2.1, 1.0]])
    metrics = logits_metrics(cand, ref, topk=2)
    assert metrics.argmax_match == 1.0
    assert metrics.topk_overlap == 1.0
    assert metrics.kl_divergence >= 0.0


def test_timing_stats_and_capacity_validation():
    stats = timing_stats([1.0, 2.0, 3.0])
    assert stats.runs == 3
    assert stats.median_ms == 2.0
    assert capacity_from_bytes(memory_budget_bytes=1024, bytes_per_token=4.0, batch_size=2) == 128
    with pytest.raises(ValueError):
        capacity_from_bytes(memory_budget_bytes=0, bytes_per_token=1.0)


def test_needle_case_and_exact_match():
    case = build_needle_case(context_tokens=64, depth=0.5, answer="BLUE-17")
    assert "BLUE-17" in case.context
    assert exact_answer_match("The answer is blue-17.", case.answer)
    assert not exact_answer_match("The answer is red.", case.answer)


def test_retrieval_prompt_inserts_needle_near_sentence_boundary():
    haystack = "Alpha sentence. Beta sentence. Gamma sentence. Delta sentence."
    context, question = build_retrieval_prompt(
        haystack=haystack,
        needle="Note: APRICOT-7-BLUE is the rare paint color.",
        question="What color?",
        depth=0.5,
    )

    assert "APRICOT-7-BLUE" in context
    assert question == "What color?"
    assert extract_retrieval_target("Note: APRICOT-7-BLUE is the rare paint color.") == "APRICOT-7-BLUE"
    assert nearest_sentence_boundary(haystack, haystack.index("Beta")) == len("Alpha sentence. ")


def test_retrieval_matrix_scores_against_reference_hits():
    cells = [
        RetrievalCell(context_tokens=4096, depth=0.1, reference_hits=1, candidate_hits=1, trials=1),
        RetrievalCell(context_tokens=4096, depth=0.5, reference_hits=1, candidate_hits=0, trials=1),
        RetrievalCell(context_tokens=8192, depth=0.9, reference_hits=0, candidate_hits=0, trials=1),
    ]

    matrix = retrieval_matrix(cells, skipped=[(16384, 0.5)])

    assert matrix.score == pytest.approx(100.0 * (1.0 - (0.0 + 1.0 + 0.0) / 3.0))
    assert matrix.skipped == ((16384, 0.5),)
    assert matrix.cells[1].loss == 1.0


def test_compose_quality_score_uses_harmonic_mean_and_notes():
    score = compose_quality_score(
        {
            "trajectory": 100.0,
            "kl": 90.0,
            "retrieval": 50.0,
            "perturbation": None,
        },
        floor_score=99.7,
    )

    assert score.band == "drift"
    assert score.floor_ok is True
    assert "retrieval" in score.axes
    assert any("Long-context retrieval" in note for note in score.notes)

    text = quality_report_text(
        model="Qwen/Qwen2.5-3B",
        reference_label="hf",
        candidate_label="tq",
        score=score,
    )
    assert "axis.retrieval: 50.00" in text
    assert "candidate: tq" in text


def test_quality_diagnosis_pattern_cases_and_report_dict():
    assert any("track the reference" in note for note in quality_diagnosis({"trajectory": 95.0, "kl": 94.0}))
    assert any(
        "higher-precision" in note
        for note in quality_diagnosis({"trajectory": 10.0, "kl": 20.0, "retrieval": 30.0})
    )
    assert any(
        "Token distribution changed" in note
        for note in quality_diagnosis({"trajectory": 20.0, "kl": 30.0, "retrieval": 95.0})
    )

    score = compose_quality_score({"trajectory": 99.0, "kl": 98.0}, floor_score=99.7)
    report = quality_report_dict(
        model="Qwen/Qwen2.5-3B",
        reference_label="hf",
        candidate_label="tq",
        score=score,
        extras={"context_tokens": 4096},
    )

    assert report["schema"] == "torbuquant.quality.v1"
    assert report["model"] == "Qwen/Qwen2.5-3B"
    assert report["score"]["floor_ok"] is True
    assert report["extras"] == {"context_tokens": 4096}
