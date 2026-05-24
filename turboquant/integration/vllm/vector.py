"""Recipe vector packing for vLLM TurboQuant cache entries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch

from turboquant.core.codebook import get_codebook_tensors
from turboquant.integration.vllm.recipe import (
    QJL_SCALE,
    QJL_SEED_OFFSET,
    SEED,
    RecipeLayout,
    recipe_layout,
)
from turboquant.packing.bits import pack_unsigned, unpack_unsigned


@dataclass(frozen=True)
class RecipeTables:
    layout: RecipeLayout
    group_indices: tuple[torch.Tensor, torch.Tensor]
    rotations: tuple[torch.Tensor, torch.Tensor]
    qjl_matrices: tuple[torch.Tensor, torch.Tensor]
    centroids: dict[int, torch.Tensor]


def build_recipe_tables(
    *,
    recipe: str,
    head_dim: int,
    group_indices: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    use_exact: bool = True,
) -> RecipeTables:
    layout = recipe_layout(recipe, head_dim)
    group_dims = tuple(group.dim for group in layout.groups)
    rotations = tuple(
        structured_signs(device=device, dim=dim, seed=SEED + offset + dim)
        for dim, offset in zip(group_dims, (101, 211), strict=True)
    )
    qjl_matrices = tuple(
        structured_signs(
            device=device,
            dim=dim,
            seed=SEED + QJL_SEED_OFFSET + offset + dim,
        )
        for dim, offset in zip(group_dims, (307, 401), strict=True)
    )
    centroid_dims = {
        group.mse_bits: group.dim
        for group in layout.groups
        if group.mse_bits > 0
    }
    centroids = {
        bits: get_codebook_tensors(
            dim,
            bits,
            device,
            dtype=torch.float32,
            use_exact=use_exact,
        )[0]
        for bits, dim in centroid_dims.items()
    }
    return RecipeTables(
        layout=layout,
        group_indices=(
            group_indices[0].to(device=device, dtype=torch.int64).contiguous(),
            group_indices[1].to(device=device, dtype=torch.int64).contiguous(),
        ),
        rotations=(rotations[0].contiguous(), rotations[1].contiguous()),
        qjl_matrices=(qjl_matrices[0].contiguous(), qjl_matrices[1].contiguous()),
        centroids=centroids,
    )


@lru_cache(maxsize=None)
def _hadamard_block_sizes(dim: int) -> tuple[int, ...]:
    sizes: list[int] = []
    remaining = dim
    while remaining > 0:
        block = 1 << (remaining.bit_length() - 1)
        sizes.append(block)
        remaining -= block
    return tuple(sizes)


def _fwht_pow2(x: torch.Tensor) -> torch.Tensor:
    shape = x.shape
    size = shape[-1]
    out = x.reshape(-1, size)
    block = 1
    while block < size:
        out = out.reshape(out.shape[0], -1, block * 2)
        left = out[..., :block]
        right = out[..., block : 2 * block]
        out = torch.cat((left + right, left - right), dim=-1)
        out = out.reshape(-1, size)
        block *= 2
    return out.reshape(shape)


def structured_signs(*, device: torch.device, dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    signs = torch.randint(0, 2, (dim,), generator=generator, dtype=torch.int64)
    signs = signs.to(dtype=torch.float32).mul_(2.0).sub_(1.0)
    return signs.to(device=device)


def apply_block_hadamard(
    x: torch.Tensor,
    signs: torch.Tensor,
    *,
    normalized: bool,
    inverse: bool,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    cursor = 0
    for block_size in _hadamard_block_sizes(x.shape[-1]):
        block = x[..., cursor : cursor + block_size]
        block_signs = signs[cursor : cursor + block_size]
        if inverse:
            block = _fwht_pow2(block)
            block = block * block_signs
        else:
            block = block * block_signs
            block = _fwht_pow2(block)
        if normalized:
            block = block / math.sqrt(block_size)
        outputs.append(block)
        cursor += block_size
    return torch.cat(outputs, dim=-1)


def pack_recipe_vectors(x: torch.Tensor, recipe: str, tables: RecipeTables) -> torch.Tensor:
    if x.shape[-1] != tables.layout.head_dim:
        raise ValueError(f"last dimension must be {tables.layout.head_dim}")
    if recipe != tables.layout.recipe:
        raise ValueError(f"tables use {tables.layout.recipe}, got {recipe}")

    groups = tuple(_gather_group(x.to(torch.float32), indices) for indices in tables.group_indices)
    packed_groups: list[torch.Tensor] = []
    for group_x, group_layout, rotation, qjl_matrix in zip(
        groups,
        tables.layout.groups,
        tables.rotations,
        tables.qjl_matrices,
        strict=True,
    ):
        vector_norms = group_x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        unit = group_x / vector_norms
        rotated = apply_block_hadamard(unit, rotation, normalized=True, inverse=False)
        indices = torch.zeros_like(rotated, dtype=torch.int64)
        rotated_hat = torch.zeros_like(rotated, dtype=torch.float32)
        if group_layout.mse_bits > 0:
            centroids = tables.centroids[group_layout.mse_bits]
            mse_indices = torch.abs(rotated.unsqueeze(-1) - centroids).argmin(dim=-1)
            indices = mse_indices.to(torch.int64)
            rotated_hat = centroids[mse_indices.long()]

        mse_hat = apply_block_hadamard(rotated_hat, rotation, normalized=True, inverse=True)
        residual = unit - mse_hat
        residual_norms = residual.norm(dim=-1, keepdim=True)
        qjl_bits = (
            apply_block_hadamard(residual, qjl_matrix, normalized=False, inverse=False) >= 0
        ).to(torch.int64)
        packed_groups.append(
            torch.cat(
                (
                    pack_unsigned(indices, group_layout.mse_bits),
                    pack_unsigned(qjl_bits, 1),
                    _norms_to_bytes(vector_norms.squeeze(-1)),
                    _norms_to_bytes(residual_norms.squeeze(-1)),
                ),
                dim=-1,
            )
        )
    return torch.cat(packed_groups, dim=-1)


def unpack_recipe_vectors(
    packed: torch.Tensor,
    recipe: str,
    tables: RecipeTables,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    if recipe != tables.layout.recipe:
        raise ValueError(f"tables use {tables.layout.recipe}, got {recipe}")
    if packed.shape[-1] != tables.layout.packed_dim:
        raise ValueError(f"packed dimension must be {tables.layout.packed_dim}")

    group_outputs: list[torch.Tensor] = []
    cursor = 0
    for group_layout, rotation, qjl_matrix in zip(
        tables.layout.groups,
        tables.rotations,
        tables.qjl_matrices,
        strict=True,
    ):
        group_packed = packed[..., cursor : cursor + group_layout.packed_bytes]
        cursor += group_layout.packed_bytes
        local = 0
        mse_indices = unpack_unsigned(
            group_packed[..., local : local + group_layout.mse_payload_bytes],
            group_layout.mse_bits,
            group_layout.dim,
        )
        local += group_layout.mse_payload_bytes
        qjl_bits = unpack_unsigned(
            group_packed[..., local : local + group_layout.qjl_payload_bytes],
            1,
            group_layout.dim,
        )
        local += group_layout.qjl_payload_bytes
        vector_norms = _bytes_to_norms(group_packed[..., local : local + 2])
        local += 2
        residual_norms = _bytes_to_norms(group_packed[..., local : local + 2])

        rotated_hat = torch.zeros(
            (*packed.shape[:-1], group_layout.dim),
            dtype=torch.float32,
            device=packed.device,
        )
        if group_layout.mse_bits > 0:
            rotated_hat = tables.centroids[group_layout.mse_bits][mse_indices.long()]

        mse_hat = apply_block_hadamard(rotated_hat, rotation, normalized=True, inverse=True)
        qjl_signs = qjl_bits.to(torch.float32).mul_(2.0).sub_(1.0)
        qjl_hat = apply_block_hadamard(qjl_signs, qjl_matrix, normalized=False, inverse=True)
        qjl_hat = qjl_hat * (QJL_SCALE / group_layout.dim)
        group_outputs.append((mse_hat + qjl_hat * residual_norms) * vector_norms)

    return _scatter_groups(
        head_dim=tables.layout.head_dim,
        group_outputs=(group_outputs[0], group_outputs[1]),
        group_indices=tables.group_indices,
        dtype=dtype,
    )


def build_query_groups(
    query: torch.Tensor,
    group_indices: tuple[torch.Tensor, torch.Tensor],
    *,
    kv_head_for_query_head: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query.ndim != 3:
        raise ValueError(f"query must have shape [tokens, heads, dim], got {tuple(query.shape)}")
    gathered = tuple(
        group.to(device=query.device, dtype=torch.int64).index_select(0, kv_head_for_query_head)
        for group in group_indices
    )
    return tuple(_gather_group(query.to(torch.float32), indices) for indices in gathered)  # type: ignore[return-value]


def scatter_recipe_output(
    *,
    head_dim: int,
    group_outputs: tuple[torch.Tensor, torch.Tensor],
    group_indices: tuple[torch.Tensor, torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    return _scatter_groups(
        head_dim=head_dim,
        group_outputs=group_outputs,
        group_indices=group_indices,
        dtype=dtype,
    )


def _gather_group(x: torch.Tensor, group_indices: torch.Tensor) -> torch.Tensor:
    prefix = x.shape[:-2]
    heads = x.shape[-2]
    if group_indices.shape[0] != heads:
        raise ValueError(f"group metadata has {group_indices.shape[0]} heads, tensor has {heads}")
    index = group_indices.reshape(*((1,) * len(prefix)), *group_indices.shape)
    index = index.expand(*prefix, *group_indices.shape)
    return torch.gather(x, dim=-1, index=index)


def _scatter_groups(
    *,
    head_dim: int,
    group_outputs: tuple[torch.Tensor, torch.Tensor],
    group_indices: tuple[torch.Tensor, torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    out = torch.zeros(
        (*group_outputs[0].shape[:-1], head_dim),
        dtype=torch.float32,
        device=group_outputs[0].device,
    )
    for values, indices in zip(group_outputs, group_indices, strict=True):
        prefix = values.shape[:-2]
        index = indices.reshape(*((1,) * len(prefix)), *indices.shape)
        index = index.expand(*prefix, *indices.shape)
        out.scatter_add_(-1, index, values.float())
    return out.to(dtype=dtype)


def _norms_to_bytes(norms: torch.Tensor) -> torch.Tensor:
    norm_half = norms.to(torch.float16).contiguous()
    return norm_half.reshape(-1).view(torch.uint8).reshape(*norm_half.shape, 2)


def _bytes_to_norms(norm_bytes: torch.Tensor) -> torch.Tensor:
    raw = norm_bytes.contiguous().reshape(-1, 2).view(torch.float16)
    return raw.reshape(*norm_bytes.shape[:-1], 1).to(torch.float32)
