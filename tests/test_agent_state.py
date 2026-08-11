# SPDX-License-Identifier: MIT
"""Tests for the phase ladder — the highest-value test in the agent suite.

Each test builds a workspace in a specific situation and asserts the one phase
the loop must choose. If the ladder ever regresses, this is what catches it.
"""

from datetime import timedelta

import pytest

from agent import budget, ledger, runs, state


# --------------------------------------------------------------------------
# Workspace builders
# --------------------------------------------------------------------------

def _quota(limit=5, used=0):
    remaining = None if used is None else max(limit - used, 0)
    return {"competition": "titanic", "limit": limit, "limit_source": "sdk", "used": used,
            "remaining": remaining, "exhausted": remaining is not None and remaining <= 0}


def _mission(slug="titanic", **overrides):
    values = {
        "metric": "accuracy", "direction": "higher_better", "target_lb": "",
        "submit_threshold_pct": 1.0, "max_lanes": 1, "gpu_weekly_hours": 15,
        "max_gpu_hours_per_run": 8, "stop_before_hours": 12, "max_idle_ticks": 12,
        "max_consecutive_kernel_errors": 3, "human_submission_reserve": 2,
        "agent_submission_cap": 3,
    }
    values.update(overrides)
    path = ledger.mission_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# Mission — {slug}\n\n"
        + "\n".join(f"{k}: {v}" for k, v in values.items() if v != "")
        + "\n",
        encoding="utf-8",
    )


def _research(slug="titanic"):
    path = ledger.research_dir(slug) / "brief.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Brief\nEvaluation is accuracy.\n", encoding="utf-8")


def _backlog(slug="titanic", items=("H-001 high ready", "H-002 med ready", "H-003 low ready")):
    """Items are "<id> <priority> <status>", optionally suffixed with " gpu"."""
    lines = [f"# Backlog — {slug}", ""]
    for item in items:
        hid, priority, status, *rest = item.split()
        cost = "gpu-4h" if rest == ["gpu"] else "cpu-15m"
        lines += [f"## {hid} — Try something [priority: {priority}] [cost: {cost}] [status: {status}]",
                  "**Why:** grounded in the brief.", ""]
    path = ledger.backlog_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture
def warm(agent_workspace):
    """A bootstrapped workspace with a full backlog and no runs."""
    _mission()
    _research()
    _backlog()
    return agent_workspace


def _started(slug="titanic", run_id="r-1", **extra):
    ledger.append_jsonl(ledger.runs_ledger_path(slug), {
        "event": "run_started", "run_id": run_id, "competition": slug,
        "kernel": "tester/demo", "version": 1, "gpu": False, "lane": "default",
        "hypothesis_id": "H-001", "config": {}, "config_hash": "abc", **extra})


def _finished(slug="titanic", run_id="r-1", files=("submission.csv", "metrics.json")):
    ledger.append_jsonl(ledger.runs_ledger_path(slug), {
        "event": "run_finished", "run_id": run_id, "status": "complete",
        "files": list(files), "output_dir": str(ledger.run_output_dir(run_id))})


def _scored(slug="titanic", run_id="r-1", cv=0.84, verdict="keep"):
    ledger.append_jsonl(ledger.runs_ledger_path(slug), {
        "event": "run_scored", "run_id": run_id, "cv_score": cv,
        "cv_metric": "accuracy", "verdict": verdict})


def _tick(slug="titanic", quota=None, **kwargs):
    return state.tick_state(slug, offline=True, quota=quota or _quota(), **kwargs)


# --------------------------------------------------------------------------
# Backlog parsing
# --------------------------------------------------------------------------

def test_parse_backlog_reads_tags(agent_workspace):
    items = state.parse_backlog(
        "# Backlog\n"
        "## H-005 — Target-encode Cabin deck  [priority: high] [cost: cpu-15m] [status: ready]\n"
        "**Why:** a public kernel reports a gain.\n"
        "## H-006 — Ten folds  [priority: low] [cost: gpu-4h] [status: blocked]\n"
    )
    assert [i["id"] for i in items] == ["H-005", "H-006"]
    assert items[0]["title"] == "Target-encode Cabin deck"
    assert items[0]["status"] == "ready"
    assert items[1]["needs_gpu"] is True


def test_parse_backlog_stops_at_retired(agent_workspace):
    items = state.parse_backlog(
        "## H-001 — live  [status: ready]\n"
        "## Retired\n"
        "## H-000 — dead  [status: ready]\n"
    )
    assert [i["id"] for i in items] == ["H-001"]


def test_ready_items_are_priority_ordered_and_gpu_filtered(agent_workspace):
    items = state.parse_backlog(
        "## H-1 — a [priority: low] [cost: cpu-1m] [status: ready]\n"
        "## H-2 — b [priority: high] [cost: gpu-2h] [status: ready]\n"
        "## H-3 — c [priority: med] [cost: cpu-1m] [status: running]\n"
    )
    assert [i["id"] for i in state.ready_items(items)] == ["H-2", "H-1"]
    assert [i["id"] for i in state.ready_items(items, gpu_allowed=False)] == ["H-1"]


# --------------------------------------------------------------------------
# Score helpers
# --------------------------------------------------------------------------

def test_is_better_respects_direction(agent_workspace):
    assert state.is_better(0.9, 0.8, "higher_better")
    assert not state.is_better(0.7, 0.8, "higher_better")
    assert state.is_better(0.07, 0.08, "lower_better")
    assert state.is_better(0.5, None, "higher_better")


def test_relative_gain_is_sign_aware(agent_workspace):
    assert state.relative_gain(0.101, 0.100, "higher_better") == pytest.approx(1.0)
    assert state.relative_gain(0.099, 0.100, "lower_better") == pytest.approx(1.0)
    assert state.relative_gain(0.099, 0.100, "higher_better") == pytest.approx(-1.0)


def test_champion_ignores_rejected_runs(warm):
    _started(run_id="r-1"); _finished(run_id="r-1"); _scored(run_id="r-1", cv=0.90, verdict="reject")
    _started(run_id="r-2"); _finished(run_id="r-2"); _scored(run_id="r-2", cv=0.84, verdict="keep")

    champ = state.champion(runs.load_runs("titanic"), "higher_better")
    assert champ["run_id"] == "r-2"


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------

def test_cold_workspace_bootstraps(agent_workspace):
    result = _tick()
    assert result["next_phase"] == "BOOTSTRAP"
    assert "MISSION.md" in result["next_action"]


def test_partially_bootstrapped_still_bootstraps(agent_workspace):
    _mission()
    result = _tick()
    assert result["next_phase"] == "BOOTSTRAP"
    assert "research/brief.md" in result["next_action"]


def test_warm_workspace_with_backlog_launches(warm):
    result = _tick()
    assert result["next_phase"] == "LAUNCH"
    assert "H-001" in result["next_action"]


def test_finished_run_harvests_before_launching(warm):
    _started(); _finished()
    result = _tick()

    assert result["next_phase"] == "HARVEST"
    assert "r-1" in result["next_action"]


def test_running_run_waits_and_blocks_a_second_launch(warm):
    _started()
    result = _tick()

    assert result["next_phase"] == "WAIT"
    assert "do not push another kernel" in result["next_action"]


def test_scored_keep_run_submits(warm):
    _started(); _finished(); _scored(cv=0.84)
    result = _tick()

    assert result["next_phase"] == "SUBMIT"
    assert result["submit_candidate"]["run_id"] == "r-1"


def test_rejected_run_is_not_submitted(warm):
    _started(); _finished(); _scored(cv=0.84, verdict="reject")
    result = _tick()

    assert result["next_phase"] == "LAUNCH"
    assert result["submit_candidate"] is None


def test_run_without_submission_file_is_not_submitted(warm):
    _started(); _finished(files=["metrics.json", "oof.csv"]); _scored(cv=0.84)
    result = _tick()

    assert result["submit_candidate"] is None
    assert result["next_phase"] == "LAUNCH"


def test_already_submitted_run_is_not_resubmitted(warm):
    _started(); _finished(); _scored(cv=0.84)
    budget.reserve_submission("titanic", run_id="r-1", message="r-1 H-001 demo")

    result = _tick()
    assert result["submit_candidate"] is None
    assert result["next_phase"] == "LAUNCH"


def test_marginal_gain_does_not_spend_a_slot(warm):
    _started(run_id="r-1"); _finished(run_id="r-1"); _scored(run_id="r-1", cv=0.8400)
    budget.reserve_submission("titanic", run_id="r-1", message="r-1 H-001 demo")
    # +0.06% — well under the 1% threshold in MISSION.md.
    _started(run_id="r-2"); _finished(run_id="r-2"); _scored(run_id="r-2", cv=0.8405)

    result = _tick()
    assert result["submit_candidate"] is None


def test_clear_gain_over_the_submitted_best_does_submit(warm):
    _started(run_id="r-1"); _finished(run_id="r-1"); _scored(run_id="r-1", cv=0.8400)
    budget.reserve_submission("titanic", run_id="r-1", message="r-1 H-001 demo")
    _started(run_id="r-2"); _finished(run_id="r-2"); _scored(run_id="r-2", cv=0.8600)

    result = _tick()
    assert result["next_phase"] == "SUBMIT"
    assert result["submit_candidate"]["run_id"] == "r-2"


def test_budget_exhausted_falls_through_to_launch(warm):
    _started(); _finished(); _scored(cv=0.84)
    for i in range(3):
        budget.reserve_submission("titanic", run_id=f"other-{i}", message=f"other-{i} x y")

    result = _tick()
    assert result["submission_budget"]["allowed"] is False
    assert result["next_phase"] == "LAUNCH"


def test_thin_backlog_refills(warm):
    _backlog(items=("H-001 high ready",))
    result = _tick()
    # One ready item is launchable, but the queue is too thin to keep the loop fed.
    assert result["next_phase"] == "LAUNCH"

    _backlog(items=("H-001 high running",))
    assert _tick()["next_phase"] == "REFILL"


def test_empty_backlog_refills(warm):
    _backlog(items=())
    assert _tick()["next_phase"] == "REFILL"


def test_gpu_starved_backlog_waits_rather_than_launching(warm):
    _mission(gpu_weekly_hours=2)
    _backlog(items=("H-001 high ready gpu", "H-002 med ready gpu", "H-003 low ready gpu"))
    budget.record_gpu_manual(3.0, competition="titanic")

    result = _tick()
    assert result["gpu_headroom"] is False
    assert result["next_phase"] == "WAIT"
    assert "weekly reset" in result["next_action"]


# --------------------------------------------------------------------------
# Stop conditions
# --------------------------------------------------------------------------

def test_halt_file_stops_the_loop(warm):
    ledger.halt_path("titanic").write_text("stop please\n", encoding="utf-8")
    result = _tick()

    assert result["next_phase"] == "STOP"
    assert "kill switch" in result["next_action"]
    assert result["wake_seconds"] == 0


def test_deadline_margin_stops_the_loop(warm):
    soon = ledger.utc_now() + timedelta(hours=6)
    _mission(deadline=soon.isoformat(timespec="seconds"), stop_before_hours=12)

    assert _tick()["next_phase"] == "STOP"


def test_target_lb_reached_stops_the_loop(warm):
    _mission(target_lb=0.80)
    _started(); _finished(); _scored(cv=0.84)
    budget.reserve_submission("titanic", run_id="r-1", message="r-1 H-001 demo")
    budget.record_submission_result("titanic", run_id="r-1", accepted=True,
                                    eval_status="complete", public_score="0.82")

    result = _tick()
    assert result["best_lb"]["lb_score"] == 0.82
    assert result["next_phase"] == "STOP"


def test_consecutive_kernel_errors_stop_the_loop(warm):
    for i in range(3):
        _started(run_id=f"r-{i}")
        ledger.append_jsonl(ledger.runs_ledger_path("titanic"), {
            "event": "run_abandoned", "run_id": f"r-{i}", "status": "error", "reason": "OOM"})

    result = _tick()
    assert result["next_phase"] == "STOP"
    assert "failed in a row" in result["next_action"]


def test_idle_overflow_stops_the_loop(warm):
    _mission(max_idle_ticks=2)
    _started()  # a run is active, so the ladder would otherwise WAIT forever

    first = _tick()
    state.write_tick_state("titanic", first)
    assert first["next_phase"] == "WAIT"

    for _ in range(3):
        result = _tick()
        state.write_tick_state("titanic", result)

    assert result["next_phase"] == "STOP"
    assert "idle ticks" in result["next_action"]


# --------------------------------------------------------------------------
# Fingerprint, idle counter, pacing
# --------------------------------------------------------------------------

def test_fingerprint_is_stable_when_nothing_changes(warm):
    _started()
    first = _tick()
    state.write_tick_state("titanic", first)
    second = _tick()

    assert second["fingerprint"] == first["fingerprint"]
    assert second["fingerprint_unchanged"] is True
    assert second["idle_ticks"] == 1


def test_fingerprint_moves_when_a_run_finishes(warm):
    _started()
    first = _tick()
    state.write_tick_state("titanic", first)
    _finished()

    second = _tick()
    assert second["fingerprint"] != first["fingerprint"]
    assert second["idle_ticks"] == 0


def test_idle_backoff_grows_then_caps(warm):
    _started()
    result = _tick()
    state.write_tick_state("titanic", result)
    delays = []
    for _ in range(6):
        result = _tick()
        state.write_tick_state("titanic", result)
        if result["next_phase"] != "WAIT":
            break
        delays.append(result["wake_seconds"])

    assert delays == sorted(delays)
    assert max(delays) <= state.WAKE_MAX


def test_actionable_phase_wakes_immediately(warm):
    assert _tick()["wake_seconds"] == state.WAKE_WORK_QUEUED


def test_pending_evaluation_wakes_at_ten_minutes(warm):
    _started(); _finished(); _scored(cv=0.84)
    budget.reserve_submission("titanic", run_id="r-1", message="r-1 H-001 demo")
    _backlog(items=())  # nothing else to do

    result = _tick()
    assert result["next_phase"] == "REFILL"  # REFILL still outranks waiting
    _backlog()
    _started(run_id="r-2")
    assert _tick()["wake_seconds"] == state.WAKE_AWAITING_EVAL


# --------------------------------------------------------------------------
# STATE.md
# --------------------------------------------------------------------------

def test_render_state_md_covers_the_resumability_contract(warm):
    _started(); _finished(); _scored(cv=0.84)
    result = _tick()
    text = state.render_state_md(result)

    for expected in ["# STATE — titanic", "## Scores", "## Runs", "## Budget",
                     "## Backlog", "## Next action", "LOWER BOUND", "do not edit"]:
        assert expected in text
    assert result["next_phase"] in text


def test_state_md_names_the_tight_limit_escalation(warm):
    result = _tick(quota=_quota(limit=2, used=0))
    text = state.render_state_md(result)

    assert "agent may submit: NO" in text
    assert "submit manually" in text


def test_write_state_md_lands_in_the_workspace(warm):
    result = _tick()
    state.write_state_md("titanic", result)

    assert ledger.state_md_path("titanic").exists()
    assert "# STATE — titanic" in ledger.state_md_path("titanic").read_text(encoding="utf-8")


def test_journal_entry_is_one_dated_block(warm):
    result = _tick()
    entry = state.journal_entry(result, ["Did the thing.", "Learned something."])

    assert entry.startswith(f"## {result['generated_at']} — tick {result['tick']}")
    assert "- Did the thing." in entry
    assert "Wakeup" in entry


# --------------------------------------------------------------------------
# Offline sense
# --------------------------------------------------------------------------

def test_offline_sense_makes_no_kaggle_calls(warm):
    # No api passed and offline=True: any remote call would raise.
    result = state.tick_state("titanic", offline=True)

    assert result["offline"] is True
    assert result["reconciled"] == []
    # Offline degrades to the local-ledger guard rather than claiming headroom.
    assert result["submission_budget"]["quota_unknown"] is True
    assert result["submission_budget"]["limit_source"] == "offline"


def test_offline_sense_still_enforces_the_agent_cap(warm):
    for i in range(3):
        budget.reserve_submission("titanic", run_id=f"r-{i}", message=f"r-{i} H x")

    result = state.tick_state("titanic", offline=True)
    assert result["submission_budget"]["allowed"] is False
    assert "daily cap" in result["submission_budget"]["reason"]


def test_quota_fetch_failure_degrades_instead_of_crashing(warm, monkeypatch):
    """A Kaggle outage must not end the loop — it must narrow it."""
    import submission_quota

    def boom(*_a, **_k):
        raise RuntimeError("kaggle is down")

    monkeypatch.setattr(submission_quota, "submission_quota", boom)
    verdict = budget.submission_budget("titanic")

    assert verdict["quota_unknown"] is True
    assert verdict["allowed"] is True  # local ledger still has room
    assert verdict["limit_source"] == "unavailable"
