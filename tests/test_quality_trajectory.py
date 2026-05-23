import math

import pytest

from torbuquant.quality import (
    harmonic_mean,
    levenshtein,
    normalized_drift,
    perturb_prompt,
    perturbation_metrics,
    sequence_match,
    trajectory_metrics,
)


def test_sequence_match_reports_prefix_and_divergence():
    row = sequence_match([1, 2, 3, 4], [1, 2, 9, 4])

    assert row.first_divergence == 2
    assert row.prefix_agreement_length == 2
    assert row.reference_length == 4
    assert row.candidate_length == 4
    assert row.matched is False


def test_trajectory_metrics_normalizes_by_candidate_length():
    metrics = trajectory_metrics(
        references=[[1, 2, 3], [4, 5, 6]],
        candidates=[[1, 2, 3, 7, 8], [4, 9, 6, 7, 8]],
        expected_tokens=3,
    )

    assert metrics.full_match_rate == 0.0
    assert metrics.mean_prefix_agreement_length == 2.0
    assert metrics.mean_candidate_length == 5.0
    assert metrics.score == 40.0
    assert metrics.median_first_divergence == 2


def test_levenshtein_and_normalized_drift():
    assert levenshtein([1, 2, 3], [1, 9, 3, 4]) == 2
    assert normalized_drift([1, 2, 3], [1, 9, 3, 4]) == pytest.approx(2 / 3)
    assert normalized_drift([], []) == 0.0
    assert normalized_drift([], [1]) == 1.0


def test_perturb_prompt_variants_are_deterministic():
    prompt = "Build a Large system?"

    assert perturb_prompt(prompt, "case", seed=1) == "build a large system?"
    assert perturb_prompt(prompt, "punct", seed=1) == "Build a Large system"
    assert perturb_prompt("build a large system", "punct", seed=1) == "build a large system?"
    assert perturb_prompt(prompt, "paraphrase", seed=1) in {
        "Construct a Large system?",
        "Build a big system?",
    }
    assert perturb_prompt(prompt, "typo", seed=1) is not None


def test_perturbation_metrics_scores_excess_drift():
    metrics = perturbation_metrics(
        prompt_ids=["p0"],
        prompts=["Build a Large system?"],
        reference_anchor=[[1, 2, 3, 4]],
        candidate_anchor=[[1, 2, 3, 4]],
        reference_perturbed={
            ("p0", "case"): [1, 2, 3, 9],
            ("p0", "punct"): [1, 2, 3, 4],
        },
        candidate_perturbed={
            ("p0", "case"): [1, 8, 8, 8],
            ("p0", "punct"): [1, 2, 3, 4],
        },
        perturbations=("case", "punct"),
        alpha=5.0,
        seed=1,
    )

    assert len(metrics.records) == 2
    assert metrics.records[0].excess_drift == pytest.approx(0.5)
    assert metrics.records[0].score == pytest.approx(100.0 * math.exp(-2.5))
    assert metrics.records[1].score == 100.0
    assert metrics.score < 100.0


def test_harmonic_mean_and_empty_inputs():
    assert harmonic_mean([100.0, 50.0]) == pytest.approx(66.6666666667)
    assert harmonic_mean([100.0, 0.0]) == 0.0
    with pytest.raises(ValueError):
        trajectory_metrics([], [])
