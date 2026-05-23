import torch

from torbuquant.integration.vllm import (
    build_metadata,
    build_query_groups,
    build_recipe_tables,
    pack_recipe_vectors,
    recipe_bits,
    recipe_cuda_allowed,
    recipe_group_dims,
    recipe_kernel_meta,
    recipe_packed_dim,
    scatter_recipe_output,
)


def test_recipe_public_names_match_layout_values():
    assert recipe_bits("turboquant25") == 2.5
    assert recipe_bits("turboquant35") == 3.5
    assert recipe_group_dims(128, "turboquant25") == (32, 96)
    assert recipe_group_dims(128, "turboquant35") == (64, 64)
    assert recipe_packed_dim(128, "turboquant25") == 44
    assert recipe_packed_dim(128, "turboquant35") == 64
    assert recipe_cuda_allowed((8, 6))
    assert not recipe_cuda_allowed((9, 0))
    assert recipe_kernel_meta((8, 6), 128).update_tile == 16


def test_recipe_query_group_gather_and_scatter():
    metadata = build_metadata(
        recipe="turboquant25",
        head_dim=128,
        num_kv_heads=2,
        layer_names=["layer0"],
    )
    groups = metadata.get_layer("layer0").key.get_group_indices(
        device=torch.device("cpu"),
        head_dim=128,
        recipe="turboquant25",
    )
    query = torch.randn(3, 4, 128)
    kv_for_q = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    high, low = build_query_groups(query, groups, kv_head_for_query_head=kv_for_q)
    out = scatter_recipe_output(
        head_dim=128,
        group_outputs=(high, low),
        group_indices=(groups[0].index_select(0, kv_for_q), groups[1].index_select(0, kv_for_q)),
        dtype=torch.float32,
    )

    assert high.shape == (3, 4, 32)
    assert low.shape == (3, 4, 96)
    torch.testing.assert_close(out, query)


def test_recipe_vector_contract_across_head_dims():
    torch.manual_seed(11)
    for head_dim in (64, 96, 128, 256):
        metadata = build_metadata(
            recipe="turboquant35",
            head_dim=head_dim,
            num_kv_heads=2,
            layer_names=["layer0"],
        )
        groups = metadata.get_layer("layer0").key.get_group_indices(
            device=torch.device("cpu"),
            head_dim=head_dim,
            recipe="turboquant35",
        )
        tables = build_recipe_tables(
            recipe="turboquant35",
            head_dim=head_dim,
            group_indices=groups,
            device=torch.device("cpu"),
        )
        x = torch.randn(2, 2, head_dim)
        packed = pack_recipe_vectors(x, "turboquant35", tables)
        assert packed.shape == (2, 2, recipe_packed_dim(head_dim, "turboquant35"))
