# SPDX-License-Identifier: MIT
"""Tests for the run lifecycle: folding, dedupe guards, launch, poll, collect."""

import json
import types
from datetime import timedelta

import pytest

from agent import budget, ledger, runs


# --------------------------------------------------------------------------
# Fakes — the Kaggle API surface these tests touch, and nothing more
# --------------------------------------------------------------------------

class FakePushResult:
    def __init__(self, version=1, error=None):
        self.version_number = version
        self.error = error
        self.url = "https://www.kaggle.com/code/tester/demo"


class FakeApi:
    def __init__(self, *, version=1, push_error=None, status="COMPLETE", failure=None):
        self._version = version
        self._push_error = push_error
        self._status = status
        self._failure = failure
        self.pushed: list[str] = []
        self.outputs: list[tuple[str, str]] = []

    def kernels_push(self, path):
        self.pushed.append(path)
        return FakePushResult(self._version, self._push_error)

    def kernels_status(self, slug):
        return types.SimpleNamespace(
            status=types.SimpleNamespace(name=self._status),
            failure_message=self._failure,
        )

    def kernels_output(self, slug, path, force=True, quiet=False):
        self.outputs.append((slug, path))
        out = ledger.Path(path)
        out.mkdir(parents=True, exist_ok=True)
        (out / "submission.csv").write_text("id,y\n1,0\n", encoding="utf-8")
        (out / "metrics.json").write_text(
            json.dumps({"cv_score": 0.8412, "metric": "accuracy",
                        "folds": [0.83, 0.85], "config": {"folds": 2}}),
            encoding="utf-8",
        )
        (out / "demo.log").write_text("epoch 1\nCV: 0.8412\n", encoding="utf-8")


@pytest.fixture
def kernel_folder(agent_workspace):
    folder = ledger.kernels_dir("titanic") / "demo"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": "tester/demo",
        "title": "Demo Kernel",
        "code_file": "train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": False,
        "enable_internet": False,
        "competition_sources": ["titanic"],
    }), encoding="utf-8")
    (folder / "train.py").write_text("print('hello')\n", encoding="utf-8")
    return folder


def _gpu_folder(folder, hours_meta=True):
    meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta["enable_gpu"] = True
    (folder / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return folder


# --------------------------------------------------------------------------
# fold_runs
# --------------------------------------------------------------------------

def test_fold_layers_events_into_one_row(agent_workspace):
    rows = runs.fold_runs([
        {"event": "run_started", "run_id": "r-1", "kernel": "t/d", "gpu": True,
         "config": {"lr": 0.03}, "logged_at": "2026-08-11T07:00:00+00:00"},
        {"event": "run_observed", "run_id": "r-1", "status": "running", "elapsed_s": 600,
         "logged_at": "2026-08-11T07:10:00+00:00"},
        {"event": "run_finished", "run_id": "r-1", "status": "complete",
         "files": ["submission.csv"], "logged_at": "2026-08-11T08:00:00+00:00"},
        {"event": "run_scored", "run_id": "r-1", "cv_score": 0.84, "verdict": "keep",
         "logged_at": "2026-08-11T08:05:00+00:00"},
    ])

    (row,) = rows
    assert row["kernel"] == "t/d"
    assert row["config"] == {"lr": 0.03}
    assert row["finished"] and row["scored"]
    assert row["cv_score"] == 0.84
    assert row["active"] is False


def test_finished_but_unscored_run_is_still_active(agent_workspace):
    """Kaggle finishing is not terminal — the result still needs harvesting."""
    rows = runs.fold_runs([
        {"event": "run_started", "run_id": "r-1", "kernel": "t/d", "logged_at": "2026-08-11T07:00:00+00:00"},
        {"event": "run_finished", "run_id": "r-1", "status": "complete", "logged_at": "2026-08-11T08:00:00+00:00"},
    ])
    assert rows[0]["active"] is True
    assert runs.fold_runs([]) == []


def test_abandoned_run_is_terminal(agent_workspace):
    rows = runs.fold_runs([
        {"event": "run_started", "run_id": "r-1", "kernel": "t/d", "logged_at": "2026-08-11T07:00:00+00:00"},
        {"event": "run_abandoned", "run_id": "r-1", "status": "error", "reason": "OOM",
         "logged_at": "2026-08-11T08:00:00+00:00"},
    ])
    assert rows[0]["active"] is False
    assert rows[0]["abandoned_reason"] == "OOM"


def test_orphan_events_are_dropped_not_invented(agent_workspace):
    rows = runs.fold_runs([
        {"event": "run_finished", "run_id": "ghost", "status": "complete"},
        {"event": "run_scored", "run_id": "ghost", "cv_score": 1.0},
    ])
    assert rows == []


def test_awaiting_harvest_selects_finished_unscored(agent_workspace):
    path = ledger.runs_ledger_path("titanic")
    for record in [
        {"event": "run_started", "run_id": "r-1", "kernel": "t/a"},
        {"event": "run_finished", "run_id": "r-1", "status": "complete"},
        {"event": "run_started", "run_id": "r-2", "kernel": "t/b"},
    ]:
        ledger.append_jsonl(path, record)

    assert [r["run_id"] for r in runs.awaiting_harvest("titanic")] == ["r-1"]


def test_consecutive_kernel_errors_stops_at_a_success(agent_workspace):
    rows = runs.fold_runs([
        {"event": "run_started", "run_id": "r-1", "kernel": "t/a"},
        {"event": "run_scored", "run_id": "r-1", "cv_score": 0.8, "verdict": "keep"},
        {"event": "run_started", "run_id": "r-2", "kernel": "t/a"},
        {"event": "run_abandoned", "run_id": "r-2", "status": "error", "reason": "x"},
        {"event": "run_started", "run_id": "r-3", "kernel": "t/a"},
        {"event": "run_abandoned", "run_id": "r-3", "status": "error", "reason": "y"},
    ])
    assert runs.consecutive_kernel_errors(rows) == 2


# --------------------------------------------------------------------------
# config_hash
# --------------------------------------------------------------------------

def test_config_hash_is_stable_under_key_reordering(kernel_folder):
    code = kernel_folder / "train.py"
    a = runs.config_hash(code, {"lr": 0.03, "folds": 5})
    b = runs.config_hash(code, {"folds": 5, "lr": 0.03})
    assert a == b


def test_config_hash_changes_with_code_bytes(kernel_folder):
    code = kernel_folder / "train.py"
    before = runs.config_hash(code, {"lr": 0.03})
    code.write_text("print('changed')\n", encoding="utf-8")
    assert runs.config_hash(code, {"lr": 0.03}) != before


def test_config_hash_missing_code_raises_launch_error(kernel_folder):
    with pytest.raises(runs.LaunchError):
        runs.config_hash(kernel_folder / "nope.py", {})


# --------------------------------------------------------------------------
# launch
# --------------------------------------------------------------------------

def test_launch_pushes_and_records(kernel_folder):
    api = FakeApi(version=7)
    result = runs.launch(kernel_folder, slug="titanic", hypothesis_id="H-004",
                         config={"lr": 0.03}, api=api)

    assert result["version"] == 7
    assert api.pushed == [str(kernel_folder)]

    (row,) = runs.load_runs("titanic")
    assert row["hypothesis_id"] == "H-004"
    assert row["kernel"] == "tester/demo"
    assert row["active"] is True
    # The kernel lock is held for the run's whole life, not just the push.
    assert ledger.read_lock(ledger.kernel_lock_name("tester/demo"))["run_id"] == row["run_id"]


def test_launch_dry_run_writes_nothing(kernel_folder):
    api = FakeApi()
    result = runs.launch(kernel_folder, slug="titanic", config={"lr": 0.03},
                         api=api, dry_run=True)

    assert result["dry_run"] is True
    assert result["config_hash"]
    assert api.pushed == []
    assert runs.load_runs("titanic") == []
    assert ledger.read_lock(ledger.kernel_lock_name("tester/demo")) is None


def test_dry_run_env_var_makes_launch_inert(kernel_folder, monkeypatch):
    monkeypatch.setenv("KAGGLE_AGENT_DRY_RUN", "1")
    api = FakeApi()
    result = runs.launch(kernel_folder, slug="titanic", api=api)

    assert result["dry_run"] is True
    assert api.pushed == []


def test_launch_refuses_when_a_run_is_active(kernel_folder):
    api = FakeApi()
    runs.launch(kernel_folder, slug="titanic", config={"a": 1}, api=api)

    with pytest.raises(runs.LaunchError, match="max_lanes"):
        runs.launch(kernel_folder, slug="titanic", config={"a": 2}, api=api)


def test_launch_refuses_a_duplicate_experiment(kernel_folder):
    api = FakeApi()
    first = runs.launch(kernel_folder, slug="titanic", config={"lr": 0.03}, api=api)
    runs.score("titanic", first["run_id"], cv_score=0.8, verdict="reject")

    with pytest.raises(runs.LaunchError, match="identical experiment"):
        runs.launch(kernel_folder, slug="titanic", config={"lr": 0.03}, api=api)


def test_force_overrides_duplicate_guard(kernel_folder):
    api = FakeApi()
    first = runs.launch(kernel_folder, slug="titanic", config={"lr": 0.03}, api=api)
    runs.score("titanic", first["run_id"], cv_score=0.8, verdict="reject")
    ledger.release_lock(ledger.kernel_lock_name("tester/demo"))

    second = runs.launch(kernel_folder, slug="titanic", config={"lr": 0.03},
                         api=api, force=True)
    assert second["run_id"] != first["run_id"]


def test_launch_refuses_when_gpu_budget_is_spent(kernel_folder):
    _gpu_folder(kernel_folder)
    mission = dict(budget.MISSION_DEFAULTS, gpu_weekly_hours=4.0)
    budget.reserve_gpu("titanic", run_id="other", estimated_hours=3.5)
    api = FakeApi()

    with pytest.raises(runs.LaunchError, match="GPU budget"):
        runs.launch(kernel_folder, slug="titanic", estimated_hours=2.0,
                    api=api, mission=mission)
    assert api.pushed == []


def test_gpu_launch_reserves_hours_immediately(kernel_folder):
    _gpu_folder(kernel_folder)
    api = FakeApi()
    runs.launch(kernel_folder, slug="titanic", estimated_hours=3.0, api=api,
                mission=dict(budget.MISSION_DEFAULTS, gpu_weekly_hours=15.0))

    # Visible to a concurrent competition loop before the run finishes.
    assert budget.gpu_week(competition="titanic")["competition_hours"] == 3.0


def test_launch_traps_sys_exit_from_bad_metadata(agent_workspace):
    folder = ledger.kernels_dir("titanic") / "broken"
    folder.mkdir(parents=True, exist_ok=True)

    with pytest.raises(runs.LaunchError, match="invalid kernel folder"):
        runs.launch(folder, slug="titanic", api=FakeApi())


def test_launch_traps_sys_exit_from_failed_push(kernel_folder):
    api = FakeApi(push_error="quota exceeded")

    with pytest.raises(runs.LaunchError, match="rejected the push"):
        runs.launch(kernel_folder, slug="titanic", api=api)
    # A failed push must not leave the kernel locked forever.
    assert ledger.read_lock(ledger.kernel_lock_name("tester/demo")) is None
    assert runs.load_runs("titanic") == []


def test_second_process_cannot_push_the_same_kernel(kernel_folder):
    """The lock guard, seen from a concurrent session."""
    api = FakeApi()
    runs.launch(kernel_folder, slug="titanic", config={"a": 1}, api=api)

    # A different competition workspace, same kernel ref, lanes free.
    with pytest.raises(ledger.LockTimeout):
        ledger.acquire_lock(ledger.kernel_lock_name("tester/demo"), timeout=0.05, stale_after=3600)


# --------------------------------------------------------------------------
# poll
# --------------------------------------------------------------------------

def test_poll_records_a_status_change(kernel_folder):
    api = FakeApi(status="RUNNING")
    launched = runs.launch(kernel_folder, slug="titanic", api=api)

    result = runs.poll("titanic", launched["run_id"], api=api)
    assert result["status"] == "running"
    assert result["terminal"] is False
    assert result.get("recorded") is True


def test_poll_does_not_spam_the_ledger_when_nothing_moved(kernel_folder):
    api = FakeApi(status="RUNNING")
    launched = runs.launch(kernel_folder, slug="titanic", api=api)
    runs.poll("titanic", launched["run_id"], api=api)
    before = len(ledger.read_jsonl(ledger.runs_ledger_path("titanic")))

    runs.poll("titanic", launched["run_id"], api=api)
    assert len(ledger.read_jsonl(ledger.runs_ledger_path("titanic"))) == before


def test_poll_flags_terminal_status(kernel_folder):
    api = FakeApi(status="COMPLETE")
    launched = runs.launch(kernel_folder, slug="titanic", api=api)

    assert runs.poll("titanic", launched["run_id"], api=api)["terminal"] is True


def test_poll_survives_an_api_error(kernel_folder):
    api = FakeApi(status="RUNNING")
    launched = runs.launch(kernel_folder, slug="titanic", api=api)

    def boom(_slug):
        raise RuntimeError("kaggle is down")

    api.kernels_status = boom
    result = runs.poll("titanic", launched["run_id"], api=api)

    assert result["status"] == "unknown"
    assert result["terminal"] is False


def test_is_overdue_uses_the_per_run_ceiling(agent_workspace):
    mission = dict(budget.MISSION_DEFAULTS, max_gpu_hours_per_run=8.0)
    started = (ledger.utc_now() - timedelta(hours=11)).isoformat(timespec="seconds")

    assert runs.is_overdue({"started_at": started}, mission) is True
    fresh = (ledger.utc_now() - timedelta(hours=1)).isoformat(timespec="seconds")
    assert runs.is_overdue({"started_at": fresh}, mission) is False


# --------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------

def test_collect_downloads_records_and_frees_the_lock(kernel_folder):
    api = FakeApi(status="COMPLETE")
    launched = runs.launch(kernel_folder, slug="titanic", api=api)
    run_id = launched["run_id"]

    result = runs.collect("titanic", run_id, api=api)

    assert "submission.csv" in result["files"]
    assert result["metrics"]["cv_score"] == 0.8412
    assert result["log_path"].endswith("demo.log")
    assert ledger.read_lock(ledger.kernel_lock_name("tester/demo")) is None

    row = runs.find_run("titanic", run_id)
    assert row["finished"] is True
    assert row["active"] is True  # still needs a verdict


def test_collect_prunes_oversized_artifacts(kernel_folder):
    api = FakeApi(status="COMPLETE")
    launched = runs.launch(kernel_folder, slug="titanic", api=api)
    run_id = launched["run_id"]

    original = api.kernels_output

    def with_a_huge_file(slug, path, force=True, quiet=False):
        original(slug, path, force, quiet)
        (ledger.Path(path) / "checkpoint.pt").write_bytes(b"0" * (2 * 1024 * 1024))

    api.kernels_output = with_a_huge_file
    result = runs.collect("titanic", run_id, api=api, max_mb=1)

    assert any("checkpoint.pt" in p for p in result["pruned_files"])
    assert not (ledger.run_output_dir(run_id) / "checkpoint.pt").exists()


def test_collect_of_a_gpu_run_settles_the_reservation(kernel_folder):
    _gpu_folder(kernel_folder)
    api = FakeApi(status="COMPLETE")
    launched = runs.launch(kernel_folder, slug="titanic", estimated_hours=3.0, api=api,
                           mission=dict(budget.MISSION_DEFAULTS, gpu_weekly_hours=15.0))
    assert budget.gpu_week(competition="titanic")["competition_hours"] == 3.0

    runs.collect("titanic", launched["run_id"], api=api)

    # The actual reading supersedes the estimate rather than double-counting it.
    week = budget.gpu_week(competition="titanic")
    assert week["competition_hours"] < 1.0
    assert week["open_reserved_hours"] == 0.0


def test_abandon_closes_the_run_and_frees_the_lock(kernel_folder):
    api = FakeApi(status="ERROR")
    launched = runs.launch(kernel_folder, slug="titanic", api=api)

    runs.abandon("titanic", launched["run_id"], reason="CUDA OOM at epoch 12")

    row = runs.find_run("titanic", launched["run_id"])
    assert row["active"] is False
    assert row["abandoned_reason"] == "CUDA OOM at epoch 12"
    assert ledger.read_lock(ledger.kernel_lock_name("tester/demo")) is None


def test_lane_is_free_again_after_abandon(kernel_folder):
    api = FakeApi()
    first = runs.launch(kernel_folder, slug="titanic", config={"a": 1}, api=api)
    runs.abandon("titanic", first["run_id"], reason="stuck")

    second = runs.launch(kernel_folder, slug="titanic", config={"a": 2}, api=api)
    assert second["run_id"] != first["run_id"]
