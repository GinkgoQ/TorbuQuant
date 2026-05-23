from pathlib import Path

import pytest

from torbuquant.bench import (
    KVSpec,
    assert_corpus_identity,
    build_llama_generation_command,
    build_llama_kld_base_command,
    build_llama_kld_command,
    build_llama_tokenize_command,
    build_llama_trajectory_command,
    parse_kld_metrics,
    parse_token_ids,
    parse_trajectory_jsonl,
    sglang_kv_pair,
    strip_engine_output,
    vllm_kv_dtype,
    write_corpus_sidecar,
)


def test_kv_spec_parse_label_and_cli_args():
    kv = KVSpec.parse("ctk=q8_0,ctv=turbo4,env.TURBO_SPARSE_V=1,threads=8")

    assert kv.key_cache == "q8_0"
    assert kv.value_cache == "turbo4"
    assert kv.env == {"TURBO_SPARSE_V": "1"}
    assert kv.cli_args() == ["-ctk", "q8_0", "-ctv", "turbo4", "--threads", "8"]
    assert "env.TURBO_SPARSE_V=1" in kv.label()


def test_llama_generation_command_has_required_noninteractive_flags(tmp_path):
    kv = KVSpec.parse("ctk=f16,ctv=f16")
    spec = build_llama_generation_command(
        bin_dir=tmp_path,
        model=Path("model.gguf"),
        prompt="hello",
        kv=kv,
        tokens=8,
        engine_flags=["-ub", "64"],
    )

    assert str(tmp_path / "llama-cli") == spec.argv[0]
    assert "--single-turn" in spec.argv
    assert "--no-display-prompt" in spec.argv
    assert "--no-conversation" not in spec.argv
    assert spec.argv[-2:] == ("-ub", "64")


def test_llama_kld_commands_have_expected_flags(tmp_path):
    kv = KVSpec.parse("ctk=q8_0,ctv=q8_0")
    base = build_llama_kld_base_command(
        bin_dir=tmp_path,
        model=Path("model.gguf"),
        corpus=Path("wiki.raw"),
        base_path=Path("base.bin"),
        kv=kv,
    )
    scored = build_llama_kld_command(
        bin_dir=tmp_path,
        model=Path("model.gguf"),
        corpus=Path("wiki.raw"),
        base_path=Path("base.bin"),
        kv=kv,
    )

    assert "--kl-divergence-base" in base.argv
    assert "--kl-divergence" not in base.argv
    assert "--kl-divergence" in scored.argv
    assert scored.argv.index("--kl-divergence") < scored.argv.index("--kl-divergence-base")


def test_llama_trajectory_and_tokenize_commands(tmp_path):
    kv = KVSpec.parse("ctk=f16,ctv=f16")
    traj = tmp_path / "tokens.jsonl"
    spec = build_llama_trajectory_command(
        bin_dir=tmp_path,
        model=Path("model.gguf"),
        prompt="hello",
        trajectory_path=traj,
        kv=kv,
    )
    tok = build_llama_tokenize_command(bin_dir=tmp_path, model=Path("model.gguf"))

    assert spec.argv[0] == str(tmp_path / "llama-completion")
    assert spec.env["TORBUQUANT_TRAJECTORY"] == str(traj)
    assert "-no-cnv" in spec.argv
    assert tok.argv[0] == str(tmp_path / "llama-tokenize")
    assert "--stdin" in tok.argv


def test_strip_engine_output_and_parse_kld_metrics():
    text = "Loading model...\n██ ██\n| The answer is blue.\nllama_perf_context_print: 1\n"
    assert strip_engine_output(text) == "The answer is blue."

    metrics = parse_kld_metrics(
        "Mean    KLD:   0.0015\nFinal estimate: PPL = 6.25 +/- 0.1\nRMS Δp: 1.3 %\nSame top-p: 98.0 %"
    )
    assert metrics["mean_kld"] == 0.0015
    assert metrics["ppl"] == 6.25
    assert metrics["rms_dp_pct"] == 1.3
    assert metrics["same_top_p_pct"] == 98.0


def test_parse_token_and_trajectory_outputs():
    assert parse_token_ids("[1, 2, 3]") == [1, 2, 3]
    assert parse_token_ids("") == []
    assert parse_trajectory_jsonl('{"step": 0, "token_id": 42}\n{"step": 1, "token_id": 43}\n') == [42, 43]


def test_external_backend_kv_mappings():
    assert vllm_kv_dtype(KVSpec.parse("ctk=q8_0,ctv=turbo4")) == "turboquant_k8v4"
    assert vllm_kv_dtype(KVSpec.parse("ctk=turbo3,ctv=turbo3")) == "turboquant_3bit_nc"
    assert sglang_kv_pair(KVSpec.parse("ctk=q8_0,ctv=q8_0")) == ("q8_0", "q8_0")

    with pytest.raises(ValueError, match="vLLM KV dtype mapping"):
        vllm_kv_dtype(KVSpec.parse("ctk=x,ctv=y"))
    with pytest.raises(ValueError, match="SGLang KV mapping"):
        sglang_kv_pair(KVSpec.parse("ctk=turbo4,ctv=turbo4"))


def test_corpus_sidecar_detects_mismatch(tmp_path):
    corpus = tmp_path / "a.txt"
    corpus.write_text("alpha beta", encoding="utf-8")
    base = tmp_path / "base.bin"
    base.write_bytes(b"base")

    write_corpus_sidecar(base, corpus)
    assert_corpus_identity(base, corpus)

    corpus.write_text("alpha gamma", encoding="utf-8")
    with pytest.raises(RuntimeError, match="corpus identity mismatch"):
        assert_corpus_identity(base, corpus)
