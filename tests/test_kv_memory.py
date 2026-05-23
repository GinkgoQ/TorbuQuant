import torch

from torbuquant.kv import ByteLedger, MemoryReport, bytes_from_tensors, report_from_components


def test_memory_report_totals_and_ratio():
    report = MemoryReport(
        dense_kv_bytes=1024,
        compressed_k_bytes=100,
        compressed_v_bytes=120,
        scales_bytes=16,
        zeros_bytes=16,
        norms_bytes=32,
        centroids_bytes=8,
        rotation_bytes=64,
        recent_window_bytes=128,
        boundary_bytes=4,
        sparse_bytes=12,
        workspace_bytes=20,
        peak_allocated_bytes=256,
    )

    assert report.compressed_total_bytes == 520
    assert report.measured_total_bytes == 776
    assert report.compression_ratio == 1024 / 520
    assert report.as_dict()["compressed_total_bytes"] == 520


def test_bytes_from_tensors_uses_storage_size():
    tensors = {
        "a": torch.zeros(3, 4, dtype=torch.float16),
        "b": torch.zeros(5, dtype=torch.uint8),
    }

    assert bytes_from_tensors(tensors) == {"a": 24, "b": 5}


def test_byte_ledger_matches_tensor_storage():
    k = torch.zeros(2, 7, dtype=torch.uint8)
    v = torch.zeros(2, 9, dtype=torch.uint8)
    scales = torch.zeros(2, 3, dtype=torch.float16)

    ledger = ByteLedger(dense_kv_bytes=512)
    ledger.add_tensor("compressed_k_bytes", k)
    ledger.add_tensor("compressed_v_bytes", v)
    ledger.add_tensor("scales_bytes", scales)
    report = ledger.to_report()

    assert report.compressed_k_bytes == 14
    assert report.compressed_v_bytes == 18
    assert report.scales_bytes == 12
    assert report.compressed_total_bytes == 44


def test_report_changes_when_metadata_changes():
    k = torch.zeros(10, dtype=torch.uint8)
    base = report_from_components(dense_kv_bytes=100, compressed_k=k, metadata_bytes=0)
    with_meta = report_from_components(dense_kv_bytes=100, compressed_k=k, metadata_bytes=12)

    assert base.compression_ratio == 10
    assert with_meta.compression_ratio == 100 / 22
