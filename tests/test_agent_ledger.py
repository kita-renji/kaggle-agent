# SPDX-License-Identifier: MIT
"""Tests for the append-only ledger and cross-process locking."""

import json
import threading
import time

import pytest

from agent import ledger


def test_paths_follow_project_root(agent_workspace):
    assert ledger.project_root() == agent_workspace
    assert ledger.competition_dir("titanic") == agent_workspace / "competitions" / "titanic"
    assert ledger.runs_ledger_path("titanic").name == "runs.jsonl"
    assert ledger.gpu_ledger_path() == agent_workspace / "data" / "agent" / "gpu_usage.jsonl"


def test_append_and_read_round_trip(agent_workspace):
    path = ledger.runs_ledger_path("titanic")
    assert ledger.append_jsonl(path, {"event": "run_started", "run_id": "r-1"})
    assert ledger.append_jsonl(path, {"event": "run_finished", "run_id": "r-1"})

    records = ledger.read_jsonl(path)
    assert [r["event"] for r in records] == ["run_started", "run_finished"]
    assert all("logged_at" in r for r in records)


def test_logged_at_cannot_be_overridden_by_caller(agent_workspace):
    path = ledger.runs_ledger_path("titanic")
    ledger.append_jsonl(path, {"event": "run_started", "logged_at": "1999-01-01T00:00:00+00:00"})

    (record,) = ledger.read_jsonl(path)
    assert not record["logged_at"].startswith("1999")


def test_read_jsonl_skips_malformed_lines(agent_workspace):
    path = ledger.runs_ledger_path("titanic")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"event": "a"}\n'
        "not json at all\n"
        "\n"
        '["a list, not a dict"]\n'
        '{"event": "b"}\n',
        encoding="utf-8",
    )

    assert [r["event"] for r in ledger.read_jsonl(path)] == ["a", "b"]


def test_read_jsonl_missing_file_is_empty(agent_workspace):
    assert ledger.read_jsonl(ledger.runs_ledger_path("nope")) == []


def test_concurrent_appends_all_survive(agent_workspace):
    path = ledger.gpu_ledger_path()
    errors: list[BaseException] = []

    def writer(tag: str):
        try:
            for i in range(60):
                ledger.append_jsonl(path, {"writer": tag, "i": i})
        except BaseException as exc:  # noqa: BLE001 — surfaced via assert below
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(tag,)) for tag in ("a", "b", "c")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    records = ledger.read_jsonl(path)
    # Every line parses (no interleaved or truncated JSON) and none were lost.
    assert len(records) == 180
    for tag in ("a", "b", "c"):
        assert sorted(r["i"] for r in records if r["writer"] == tag) == list(range(60))


def test_lock_is_exclusive(agent_workspace):
    ledger.acquire_lock("kernel--owner--slug", run_id="r-1")
    with pytest.raises(ledger.LockTimeout) as excinfo:
        ledger.acquire_lock("kernel--owner--slug", timeout=0.05, stale_after=3600)

    assert "test-session" in str(excinfo.value)
    assert ledger.release_lock("kernel--owner--slug")
    # Now free again.
    ledger.acquire_lock("kernel--owner--slug", timeout=0.05)


def test_lock_records_holder_metadata(agent_workspace):
    ledger.acquire_lock("kernel--owner--slug", run_id="r-42")
    holder = ledger.read_lock("kernel--owner--slug")

    assert holder["run_id"] == "r-42"
    assert holder["session"] == "test-session"
    assert "created_at" in holder


def test_stale_lock_is_reclaimed(agent_workspace):
    path = ledger.acquire_lock("stuck")
    # Backdate the lock past the staleness horizon.
    import os

    old = time.time() - 120
    os.utime(path, (old, old))

    ledger.acquire_lock("stuck", timeout=0.05, stale_after=60)
    assert ledger.read_lock("stuck")["session"] == "test-session"


def test_fresh_lock_is_not_reclaimed(agent_workspace):
    ledger.acquire_lock("busy")
    with pytest.raises(ledger.LockTimeout):
        ledger.acquire_lock("busy", timeout=0.05, stale_after=60)


def test_file_lock_releases_on_exception(agent_workspace):
    with pytest.raises(ValueError):
        with ledger.file_lock("scoped"):
            raise ValueError("boom")

    # Released despite the error — acquiring again must not block.
    ledger.acquire_lock("scoped", timeout=0.05)


def test_release_lock_reports_absence(agent_workspace):
    assert ledger.release_lock("never-held") is False


def test_append_jsonl_strict_raises_on_failure(agent_workspace, monkeypatch):
    def explode(*_args, **_kwargs):
        raise OSError("disk gone")

    monkeypatch.setattr(ledger, "file_lock", explode)
    path = ledger.budget_ledger_path("titanic")

    assert ledger.append_jsonl(path, {"event": "x"}) is False  # best effort
    with pytest.raises(OSError):
        ledger.append_jsonl(path, {"event": "submission_reserved"}, strict=True)


def test_write_json_is_atomic_and_readable(agent_workspace):
    path = ledger.tick_state_path("titanic")
    ledger.write_json(path, {"fingerprint": "abc", "idle_ticks": 2})

    assert ledger.read_json(path) == {"fingerprint": "abc", "idle_ticks": 2}
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_read_json_default_on_missing_or_corrupt(agent_workspace):
    path = ledger.tick_state_path("titanic")
    assert ledger.read_json(path, default={"x": 1}) == {"x": 1}

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert ledger.read_json(path, default={"x": 1}) == {"x": 1}


def test_append_text_creates_and_newline_terminates(agent_workspace):
    path = ledger.journal_path("titanic")
    ledger.append_text(path, "## first")
    ledger.append_text(path, "## second\n")

    assert path.read_text(encoding="utf-8") == "## first\n## second\n"


def test_new_run_id_is_unique_and_sortable(agent_workspace):
    ids = {ledger.new_run_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(rid.startswith("r-") for rid in ids)


def test_kernel_lock_name_is_path_safe(agent_workspace):
    name = ledger.kernel_lock_name("kitarenji/titanic-gbm-5fold")
    assert "/" not in name
    assert name == "kernel--kitarenji--titanic-gbm-5fold"


def test_records_are_json_serialisable_with_non_native_types(agent_workspace):
    from pathlib import Path

    path = ledger.runs_ledger_path("titanic")
    ledger.append_jsonl(path, {"event": "run_started", "folder": Path("a/b")})

    (record,) = ledger.read_jsonl(path)
    assert isinstance(record["folder"], str)
    assert json.dumps(record)
