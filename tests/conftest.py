"""Test configuration and fixtures."""

import pytest
import polars as pl
from pathlib import Path


@pytest.fixture
def sample_metadata():
    """Sample metadata matching HuggingFace dataset structure."""
    return [
        {
            "id": "1",
            "title": "Luật Đất đai 2024",
            "so_ky_hieu": "15/2024/QH15",
            "loai_van_ban": "Luật",
            "ngay_ban_hanh": "2024-01-01",
            "ngay_co_hieu_luc": "2024-08-01",
            "nganh": "Tài nguyên",
            "url": "https://example.com/1"
        },
        {
            "id": "2",
            "title": "Nghị định về đất đai",
            "so_ky_hieu": "01/2024/NĐ-CP",
            "loai_van_ban": "Nghị định",
            "ngay_ban_hanh": "2024-02-01",
            "ngay_co_hieu_luc": "2024-03-01",
            "nganh": "Tài nguyên",
            "url": "https://example.com/2"
        },
        {
            "id": "3",
            "title": "Document without content",
            "so_ky_hieu": "03/2024/ABC",
            "loai_van_ban": "Thông tư",
            "ngay_ban_hanh": "2024-04-01",
            "ngay_co_hieu_luc": "",
            "nganh": "",
            "url": "https://example.com/3"
        },
    ]


@pytest.fixture
def sample_content():
    """Sample content HTML (using 'id' field like HuggingFace dataset)."""
    return {
        "1": "<html><body><h1>Luật Đất đai 2024</h1><p>Full law content here...</p></body></html>",
        "2": "<html><body><h1>Nghị định về đất đai</h1><p>Decree content here...</p></body></html>",
        # Note: doc "3" has no content (simulates missing content)
    }


@pytest.fixture
def output_dir(tmp_path):
    """Temporary output directory for tests."""
    return tmp_path / "output"
