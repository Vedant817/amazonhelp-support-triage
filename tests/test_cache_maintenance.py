"""Tests for pipeline cache maintenance.

The cache key is a fingerprint of configuration, supervision and source files, so
every experiment writes a new ~42 MB artifact and nothing ever removes the old
ones. Twelve had accumulated in this repository (about 500 MB) before the
retention policy existed. These tests pin the eviction behaviour, including the
failure modes that matter: never evicting the cache that was just written, and
never leaving a cache file without its digest manifest.
"""

import os
import time
from pathlib import Path

from src.model_factory import _prune_pipeline_caches


def _make_cache(directory: Path, name: str, age_seconds: float = 0.0) -> Path:
    cache = directory / name
    cache.write_bytes(b"payload")
    manifest = cache.with_suffix(cache.suffix + ".sha256")
    manifest.write_text("digest  name\n", encoding="utf-8")
    stamp = time.time() - age_seconds
    os.utime(cache, (stamp, stamp))
    return cache


class TestPrunePipelineCaches:
    def test_keeps_the_requested_number_of_caches(self, tmp_path):
        for index in range(6):
            _make_cache(tmp_path, f"pipeline-{index:02d}.joblib", age_seconds=100 - index)
        _prune_pipeline_caches(tmp_path, keep="pipeline-00.joblib", retain=3)
        remaining = sorted(path.name for path in tmp_path.glob("pipeline-*.joblib"))
        assert len(remaining) == 3

    def test_never_evicts_the_cache_just_written(self, tmp_path):
        for index in range(6):
            _make_cache(tmp_path, f"pipeline-{index:02d}.joblib", age_seconds=100 - index)
        # Oldest file, deliberately passed as `keep`.
        _prune_pipeline_caches(tmp_path, keep="pipeline-05.joblib", retain=2)
        assert (tmp_path / "pipeline-05.joblib").exists()

    def test_removes_the_manifest_alongside_its_cache(self, tmp_path):
        for index in range(5):
            _make_cache(tmp_path, f"pipeline-{index:02d}.joblib", age_seconds=100 - index)
        _prune_pipeline_caches(tmp_path, keep="pipeline-00.joblib", retain=2)
        for cache in tmp_path.glob("pipeline-*.joblib"):
            assert cache.with_suffix(cache.suffix + ".sha256").exists(), (
                "a cache without its manifest can never be trusted"
            )

    def test_reports_what_it_removed(self, tmp_path):
        for index in range(4):
            _make_cache(tmp_path, f"pipeline-{index:02d}.joblib", age_seconds=100 - index)
        removed = _prune_pipeline_caches(tmp_path, keep="pipeline-00.joblib", retain=2)
        assert len(removed) == 2

    def test_no_op_when_under_the_retention_limit(self, tmp_path):
        for index in range(2):
            _make_cache(tmp_path, f"pipeline-{index:02d}.joblib", age_seconds=100 - index)
        assert _prune_pipeline_caches(tmp_path, keep="pipeline-00.joblib", retain=5) == []
        assert len(list(tmp_path.glob("pipeline-*.joblib"))) == 2

    def test_missing_directory_is_not_an_error(self, tmp_path):
        assert _prune_pipeline_caches(tmp_path / "absent", keep="x.joblib") == []

    def test_ignores_unrelated_files(self, tmp_path):
        _make_cache(tmp_path, "pipeline-00.joblib")
        (tmp_path / "notes.txt").write_text("keep me", encoding="utf-8")
        _prune_pipeline_caches(tmp_path, keep="pipeline-00.joblib", retain=1)
        assert (tmp_path / "notes.txt").exists()
