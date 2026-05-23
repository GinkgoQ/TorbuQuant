import importlib.util
from pathlib import Path
import sys

from torbuquant.bench import RunProfile, compare_run_profiles, parse_profile_text


def _load_qwen_ab_eval():
    path = Path(__file__).resolve().parents[1] / "scripts" / "qwen_ab_eval.py"
    spec = importlib.util.spec_from_file_location("qwen_ab_eval_local", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PROFILE_TEXT = """
TBQ_PROFILE_VERSION=1
TBQ_PROFILE_TIMESTAMP=2026-05-22T10:00:00Z
[SYS] platform=Linux python=3.10 torch=2.9.1 triton=3.5.1 transformers=4.57.0 ram_bytes=128000000000
[GPU] name="NVIDIA RTX" backend=cuda cuda=12.8 compute_capability=8.9 vram_bytes=24000000000
[MODEL] model_id=Qwen/Qwen2.5-3B revision=local layers=36 query_heads=16 kv_heads=2 head_dim=128
[BENCH] label="dense decode 4k" k=K16 v=V16 mode=decode context=4096 batch=1 tokens_per_s=100.0 mean_ms=10.0 peak_gpu_bytes=9000000000
[BENCH] label="tq decode 4k" k=K16 v=V4 mode=decode context=4096 batch=1 tokens_per_s=130.0 mean_ms=7.7 peak_gpu_bytes=6000000000
[BENCH] label="dense decode 8k" k=K16 v=V16 mode=decode context=8192 batch=1 tokens_per_s=80.0
[BENCH] label="tq decode 8k" k=K16 v=V4 mode=decode context=8192 batch=1 tokens_per_s=120.0
[NOTE] same prompts and generation settings
"""


def test_parse_profile_text_and_curves(tmp_path):
    profile = parse_profile_text(PROFILE_TEXT)

    assert profile.version == 1
    assert profile.system.device.backend == "cuda"
    assert profile.model.model_id == "Qwen/Qwen2.5-3B"
    assert profile.curve(mode="decode", k_format="K16", v_format="V4") == {4096: 130.0, 8192: 120.0}
    assert profile.ratio_curve(
        numerator_k="K16",
        numerator_v="V4",
        denominator_k="K16",
        denominator_v="V16",
        mode="decode",
    ) == {4096: 1.3, 8192: 1.5}

    path = tmp_path / "profile.json"
    profile.save(path)
    loaded = RunProfile.from_json(path)
    assert loaded.points[0].label == "dense decode 4k"
    assert loaded.notes == ("same prompts and generation settings",)


def test_compare_run_profiles_uses_shared_contexts_only():
    reference = parse_profile_text(PROFILE_TEXT)
    candidate = parse_profile_text(
        PROFILE_TEXT.replace("tokens_per_s=130.0", "tokens_per_s=117.0").replace(
            "tokens_per_s=120.0", "tokens_per_s=108.0"
        )
    )

    rows = compare_run_profiles(reference, candidate, mode="decode", k_format="K16", v_format="V4")

    assert rows[4096]["ratio"] == 0.9
    assert rows[8192]["delta"] == -12.0


def test_qwen_ab_profile_builder_records_hf_and_tq_points():
    reports = [
        {
            "hf": {
                "scenario": "context_4096",
                "input_tokens": 4096,
                "throughput_output_tokens_per_s": 2.0,
                "runtime_s": 1.5,
                "peak_cuda_allocated_bytes": 1000,
            },
            "tq": {
                "scenario": "context_4096",
                "input_tokens": 4096,
                "throughput_output_tokens_per_s": 1.8,
                "runtime_s": 1.7,
                "peak_cuda_allocated_bytes": 900,
                "tq_cache_report": {
                    "k_format": "K8",
                    "v_format": "V4",
                    "config": {
                        "model_id": "Qwen/Qwen2.5-3B",
                        "num_layers": 36,
                        "num_q_heads": 16,
                        "num_kv_heads": 2,
                        "head_dim": 128,
                    },
                },
            },
        }
    ]

    qwen_ab_eval = _load_qwen_ab_eval()
    profile = qwen_ab_eval.build_bench_profile(
        run_id="unit",
        model_id="Qwen/Qwen2.5-3B",
        preset="k8_v4",
        run_reports=reports,
    )

    assert len(profile.points) == 2
    assert profile.points[0].k_format == "K16"
    assert profile.points[1].v_format == "V4"
    assert profile.model.layers == 36
