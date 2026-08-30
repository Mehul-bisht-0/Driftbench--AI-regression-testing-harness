"""Tests for driftbench.store"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from driftbench.store import Store, write_jsonl, append_jsonl, read_jsonl


@pytest.fixture
def tmp_store(tmp_path):
    return Store(tmp_path / "runs")


@pytest.fixture
def sample_run():
    from driftbench.runner import run_suite
    from driftbench.variant import load as load_variant
    from driftbench.suite import by_id
    v = load_variant("prompts/v1_baseline.txt", variant_id="v1",
                     agent="scripted")
    task = by_id("ret-pricing-growth")
    return run_suite(v, tasks=[task], replicates=2, master_seed=42,
                     progress=None)


class TestJsonl:
    def test_write_read(self, tmp_path):
        path = tmp_path / "test.jsonl"
        rows = [{"a": 1}, {"b": 2}, {"c": 3}]
        write_jsonl(path, rows)
        loaded = list(read_jsonl(path))
        assert loaded == rows

    def test_append(self, tmp_path):
        path = tmp_path / "test.jsonl"
        write_jsonl(path, [{"a": 1}])
        append_jsonl(path, {"b": 2})
        loaded = list(read_jsonl(path))
        assert len(loaded) == 2

    def test_skip_blank_lines(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text('{"a": 1}\n\n{"b": 2}\n\n', encoding="utf-8")
        loaded = list(read_jsonl(path))
        assert len(loaded) == 2

    def test_bad_json_raises(self, tmp_path):
        path = tmp_path / "test.jsonl"
        path.write_text('not json\n', encoding="utf-8")
        with pytest.raises(ValueError, match="line 1"):
            list(read_jsonl(path))


class TestStore:
    def test_save_and_load(self, tmp_store, sample_run):
        tmp_store.save(sample_run)
        loaded = tmp_store.load(sample_run.manifest.run_id)
        assert loaded.manifest.run_id == sample_run.manifest.run_id
        assert len(loaded.results) == len(sample_run.results)

    def test_load_nonexistent(self, tmp_store):
        with pytest.raises(FileNotFoundError):
            tmp_store.load("nonexistent-run")

    def test_runs_list(self, tmp_store, sample_run):
        tmp_store.save(sample_run)
        rows = tmp_store.runs()
        assert len(rows) == 1
        assert rows[0].run_id == sample_run.manifest.run_id

    def test_runs_filter_by_variant(self, tmp_store, sample_run):
        tmp_store.save(sample_run)
        rows = tmp_store.runs(variant_id="v1")
        assert len(rows) == 1
        rows = tmp_store.runs(variant_id="nonexistent")
        assert len(rows) == 0

    def test_latest(self, tmp_store, sample_run):
        tmp_store.save(sample_run)
        latest = tmp_store.latest()
        assert latest is not None
        assert latest.run_id == sample_run.manifest.run_id

    def test_resolve_by_run_id(self, tmp_store, sample_run):
        tmp_store.save(sample_run)
        loaded = tmp_store.resolve(sample_run.manifest.run_id)
        assert loaded.manifest.run_id == sample_run.manifest.run_id

    def test_resolve_by_variant(self, tmp_store, sample_run):
        tmp_store.save(sample_run)
        loaded = tmp_store.resolve("v1")
        assert loaded.manifest.variant_id == "v1"

    def test_resolve_by_path(self, tmp_store, sample_run):
        path = tmp_store.save(sample_run)
        loaded = tmp_store.resolve(str(path))
        assert loaded.manifest.run_id == sample_run.manifest.run_id

    def test_resolve_nonexistent(self, tmp_store):
        with pytest.raises(FileNotFoundError):
            tmp_store.resolve("nonexistent")

    def test_reindex(self, tmp_store, sample_run):
        tmp_store.save(sample_run)
        n = tmp_store.reindex()
        assert n == 1
        rows = tmp_store.runs()
        assert len(rows) == 1

    def test_path_for(self, tmp_store):
        p = tmp_store.path_for("my-run-id")
        assert p.name == "my-run-id.jsonl"

    def test_pass_counts(self, tmp_store, sample_run):
        tmp_store.save(sample_run)
        counts = tmp_store.pass_counts(sample_run.manifest.run_id)
        assert "ret-pricing-growth" in counts
        p, n = counts["ret-pricing-growth"]
        assert p == n == 2


class TestRunRow:
    def test_line(self):
        from driftbench.store import RunRow
        row = RunRow(run_id="test-run-20260101", created_at=0,
                     variant_id="v1", variant_digest="abc",
                     suite_digest="def", agent="scripted",
                     model="test", effort="medium", replicates=5,
                     n_tasks=10, passes=50, cells=50, cost_usd=0.0,
                     notes="", path="runs/test.jsonl")
        line = row.line()
        assert "test-run" in line
        assert "100%" in line
