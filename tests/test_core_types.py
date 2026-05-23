from torbuquant.core.types import CodebookSpec, TransformSpec


def test_transform_spec_round_trip_dict():
    spec = TransformSpec(
        kind="rht",
        dim=128,
        seed=123,
        dtype="float32",
        device_type="cuda",
        pad_dim=128,
    )

    assert TransformSpec.from_dict(spec.to_dict()) == spec


def test_codebook_spec_round_trip_dict():
    spec = CodebookSpec(
        dim=128,
        bits=4,
        distribution="beta_sphere",
        cache_key="d128_b4_exact",
    )

    assert CodebookSpec.from_dict(spec.to_dict()) == spec
