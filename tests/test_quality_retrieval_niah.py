import random

import pytest

from torbuquant.quality import (
    DISTRACTOR_KEYS,
    RetrievalConfigResult,
    RetrievalNeedle,
    RetrievalTrial,
    build_filler_paragraphs,
    generate_multi_key_haystack,
    generate_multi_value_haystack,
    generate_single_haystack,
    insert_retrieval_needles,
    make_distractor_needles,
    make_retrieval_number,
    retrieval_delta_table,
    retrieval_heatmap_table,
    retrieval_multi_key_table,
    retrieval_multi_value_table,
    score_multi_numbers,
    score_single_number,
)


def test_make_retrieval_number_has_seven_digits_and_seed_control():
    rng_a = random.Random(42)
    rng_b = random.Random(42)

    number = make_retrieval_number(rng_a)

    assert number == make_retrieval_number(rng_b)
    assert len(number) == 7
    assert number.isdigit()
    assert 1_000_000 <= int(number) <= 9_999_999


def test_retrieval_needle_sentence_and_validation():
    needle = RetrievalNeedle(key="The special magic number is", value="1234567", depth=0.5)

    assert needle.sentence == "The special magic number is: 1234567."

    with pytest.raises(ValueError):
        RetrievalNeedle(key="", value="1234567", depth=0.5)
    with pytest.raises(ValueError):
        RetrievalNeedle(key="K", value="1234567", depth=1.1)


def test_build_filler_paragraphs_reaches_size_and_shuffles():
    paragraphs_a = build_filler_paragraphs(2000, random.Random(1))
    paragraphs_b = build_filler_paragraphs(2000, random.Random(99))

    assert sum(len(item) + 2 for item in paragraphs_a) >= 2000
    assert paragraphs_a[0] != paragraphs_b[0]
    assert len(set(build_filler_paragraphs(10_000, random.Random(2))[:24])) == 24


def test_insert_retrieval_needles_places_edges_and_keeps_input_list():
    paragraphs = ["aaa", "bbb", "ccc", "ddd"]
    original = list(paragraphs)
    start = RetrievalNeedle(key="K0", value="1111111", depth=0.0)
    end = RetrievalNeedle(key="K1", value="2222222", depth=1.0)

    result = insert_retrieval_needles(paragraphs, [start, end])
    parts = result.split("\n\n")

    assert parts[0] == start.sentence
    assert parts[-1] == end.sentence
    assert paragraphs == original


def test_generate_single_multi_key_and_multi_value_haystacks():
    rng = random.Random(42)
    needle = RetrievalNeedle(key="The special magic number is", value="1234567", depth=0.5)
    distractors = [
        RetrievalNeedle(key="The secret password is", value="7654321", depth=0.25),
        RetrievalNeedle(key="The hidden code is", value="1111111", depth=0.75),
    ]
    value_needles = [
        RetrievalNeedle(key="The special magic number is", value="1111111", depth=0.25),
        RetrievalNeedle(key="The special magic number is", value="2222222", depth=0.5),
        RetrievalNeedle(key="The special magic number is", value="3333333", depth=0.75),
    ]

    single = generate_single_haystack(needle, 4000, rng)
    multi_key = generate_multi_key_haystack(needle, distractors, 4000, random.Random(43))
    multi_value = generate_multi_value_haystack(value_needles, 4000, random.Random(44))

    assert len(single) >= 4000
    assert needle.sentence in single
    assert needle.sentence in multi_key
    assert all(item.sentence in multi_key for item in distractors)
    assert all(item.sentence in multi_value for item in value_needles)


def test_make_distractor_needles_uses_keys_and_depths():
    distractors = make_distractor_needles(3, random.Random(42))

    assert [item.key for item in distractors] == list(DISTRACTOR_KEYS[:3])
    assert [item.depth for item in distractors] == pytest.approx([0.2, 0.4, 0.6])
    assert all(len(item.value) == 7 and item.value.isdigit() for item in distractors)


def test_score_single_number_is_strict_about_seven_digit_tokens():
    assert score_single_number("1234567", "1234567") is True
    assert score_single_number("The number is 1234567.", "1234567") is True
    assert score_single_number("123456", "1234567") is False
    assert score_single_number("12345678", "1234567") is False
    assert score_single_number("1111111 2222222", "3333333") is False

    with pytest.raises(ValueError):
        score_single_number("1234567", "123456")


def test_score_multi_numbers_returns_per_value_hits():
    assert score_multi_numbers("1234567, 7654321, 1111111", ["1234567", "7654321", "1111111"]) == [
        True,
        True,
        True,
    ]
    assert score_multi_numbers("1234567 and text", ["1234567", "7654321"]) == [True, False]
    assert score_multi_numbers("123456", ["1234567"]) == [False]


def test_retrieval_config_result_and_tables():
    ref = RetrievalConfigResult(mode="single", context_tokens=4096, cache_format="bf16", depth=0.5)
    ref.trials = [RetrievalTrial(expected="1234567", response="1234567", found=True)]
    cand = RetrievalConfigResult(mode="single", context_tokens=4096, cache_format="k4v4", depth=0.5)
    cand.trials = [RetrievalTrial(expected="1234567", response="wrong", found=False)]
    multi_key = RetrievalConfigResult(mode="multi-key", context_tokens=8192, cache_format="k4v4", depth=0.5)
    multi_key.trials = [RetrievalTrial(expected="1234567", response="1234567", found=True)]
    multi_value = RetrievalConfigResult(mode="multi-value", context_tokens=8192, cache_format="k4v4", needle_count=2)
    multi_value.trials = [
        RetrievalTrial(expected="1111111", response="1111111, 2222222", found=True),
        RetrievalTrial(expected="2222222", response="1111111, 2222222", found=True),
    ]

    assert cand.hit_rate == 0.0
    assert cand.passed is False
    assert multi_value.hit_rate == 100.0
    assert "miss" in retrieval_heatmap_table([ref, cand], "k4v4")
    assert "loss" in retrieval_delta_table([ref, cand], "bf16", "k4v4")
    assert "Multi-Key" in retrieval_multi_key_table([multi_key])
    assert "100.0%" in retrieval_multi_value_table([multi_value])
