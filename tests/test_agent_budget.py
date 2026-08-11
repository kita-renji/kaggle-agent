# SPDX-License-Identifier: MIT
"""Tests for submission and GPU budget arithmetic.

``submission_quota`` is always injected, never called: these tests must stay
hermetic, and the whole point of the module is that its decisions are
reproducible without Kaggle.
"""

from datetime import datetime, timedelta, timezone

import pytest

from agent import budget, ledger


def _quota(limit=5, used=0, exhausted=None, source="sdk"):
    remaining = None if used is None else max(limit - used, 0)
    return {
        "competition": "titanic",
        "limit": limit,
        "limit_source": source,
        "used": used,
        "remaining": remaining,
        "exhausted": exhausted if exhausted is not None else (remaining is not None and remaining <= 0),
    }


def _reserve(slug, n, *, now=None):
    for i in range(n):
        budget.reserve_submission(slug, run_id=f"r-{i}", message=f"r-{i} H-1 x", now=now)


# --------------------------------------------------------------------------
# MISSION.md parsing
# --------------------------------------------------------------------------

def test_parse_mission_reads_values_and_keeps_defaults(agent_workspace):
    mission = budget.parse_mission(
        "# Mission — titanic\n"
        "metric: accuracy\n"
        "direction: higher_better\n"
        "target_lb: 0.82\n"
        "agent_submission_cap: 3\n"
        "\n"
        "## Out of bounds\n"
        "- Never submit a worse CV.\n"
    )
    assert mission["metric"] == "accuracy"
    assert mission["target_lb"] == 0.82
    assert mission["agent_submission_cap"] == 3
    assert mission["human_submission_reserve"] == 2  # default preserved


def test_parse_mission_ignores_prose_code_and_unknown_keys(agent_workspace):
    mission = budget.parse_mission(
        "max_lanes: 2  # inline comment\n"
        "```\n"
        "max_lanes: 99\n"
        "```\n"
        "- bullet: 42\n"
        "not_a_real_key: 7\n"
        "gpu_weekly_hours: not-a-number\n"
    )
    assert mission["max_lanes"] == 2
    assert "not_a_real_key" not in mission
    # Unparsable numeric falls back rather than crashing the tick.
    assert mission["gpu_weekly_hours"] == budget.MISSION_DEFAULTS["gpu_weekly_hours"]


def test_read_mission_missing_file_is_all_defaults(agent_workspace):
    assert budget.read_mission("titanic") == budget.MISSION_DEFAULTS


def test_mission_deadline_parses_z_suffix(agent_workspace):
    parsed = budget.mission_deadline({"deadline": "2026-12-01T23:59:00Z"})
    assert parsed == datetime(2026, 12, 1, 23, 59, tzinfo=timezone.utc)
    assert budget.mission_deadline({"deadline": "garbage"}) is None
    assert budget.mission_deadline({"deadline": None}) is None


def test_write_mission_template_does_not_clobber(agent_workspace):
    path = budget.write_mission_template("titanic", metric="accuracy")
    path.write_text("metric: edited_by_human\n", encoding="utf-8")
    budget.write_mission_template("titanic", metric="accuracy")
    assert "edited_by_human" in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Submission budget — the matrix that protects the human's slots
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "limit,kaggle_used,agent_reserved,expected_allowed,expected_available",
    [
        (5, 0, 0, True, 3),    # fresh day: agent's effective ceiling is 3
        (5, 3, 3, False, None),  # agent hit its own cap
        (5, 3, 0, True, 2),    # human already used their reserve; rest is fair game
        (5, 4, 1, True, 1),
        (5, 5, 0, False, None),  # exhausted
        (2, 0, 0, False, None),  # limit <= reserve: agent must never submit here
    ],
)
def test_submission_budget_matrix(agent_workspace, limit, kaggle_used, agent_reserved,
                                  expected_allowed, expected_available):
    _reserve("titanic", agent_reserved)
    result = budget.submission_budget("titanic", quota=_quota(limit=limit, used=kaggle_used))

    assert result["allowed"] is expected_allowed
    if expected_available is not None:
        assert result["available"] == expected_available
    assert result["reason"]


def test_tight_limit_competition_escalates_to_human(agent_workspace):
    result = budget.submission_budget("titanic", quota=_quota(limit=2, used=0))
    assert result["allowed"] is False
    assert "submit manually" in result["reason"]


def test_unknown_kaggle_usage_falls_back_to_local_ledger(agent_workspace):
    _reserve("titanic", 2)
    result = budget.submission_budget("titanic", quota=_quota(limit=5, used=None))

    assert result["allowed"] is True
    assert result["quota_unknown"] is True
    assert result["agent_used"] == 2
    assert "backstop" in result["reason"]


def test_unknown_kaggle_usage_still_respects_agent_cap(agent_workspace):
    _reserve("titanic", 3)
    result = budget.submission_budget("titanic", quota=_quota(limit=5, used=None))

    assert result["allowed"] is False
    assert "daily cap" in result["reason"]


def test_reserve_then_release_frees_the_slot(agent_workspace):
    _reserve("titanic", 3)
    assert budget.submission_budget("titanic", quota=_quota(used=3))["allowed"] is False

    budget.release_submission("titanic", run_id="r-2", reason="kaggle rejected: no submission.csv")
    result = budget.submission_budget("titanic", quota=_quota(used=2))

    assert result["agent_used"] == 2
    assert result["allowed"] is True


def test_reservations_are_scoped_to_the_utc_day(agent_workspace):
    yesterday = ledger.utc_now() - timedelta(days=1)
    _reserve("titanic", 3, now=yesterday)

    # Yesterday's spend does not constrain today.
    assert budget.agent_submissions_today("titanic") == 0
    assert budget.submission_budget("titanic", quota=_quota(used=0))["allowed"] is True


def test_reservations_are_scoped_per_competition(agent_workspace):
    _reserve("titanic", 3)
    assert budget.agent_submissions_today("birdclef-2025") == 0


def test_reserve_submission_is_strict(agent_workspace, monkeypatch):
    def explode(*_a, **_k):
        raise OSError("disk gone")

    monkeypatch.setattr(ledger, "file_lock", explode)
    with pytest.raises(OSError):
        budget.reserve_submission("titanic", run_id="r-1", message="r-1 H-1 x")


def test_mission_overrides_cap_and_reserve(agent_workspace):
    mission = dict(budget.MISSION_DEFAULTS, agent_submission_cap=1, human_submission_reserve=0)
    _reserve("titanic", 1)
    result = budget.submission_budget("titanic", quota=_quota(used=1), mission=mission)

    assert result["allowed"] is False
    assert result["agent_cap"] == 1


# --------------------------------------------------------------------------
# GPU budget
# --------------------------------------------------------------------------

def test_gpu_week_anchor_is_previous_saturday(agent_workspace):
    wednesday = datetime(2026, 8, 12, 15, 0, tzinfo=timezone.utc)  # a Wednesday
    anchor = budget.gpu_week_anchor(wednesday)

    assert anchor.weekday() == 5
    assert anchor == datetime(2026, 8, 8, tzinfo=timezone.utc)


def test_gpu_week_anchor_env_override(agent_workspace, monkeypatch):
    monkeypatch.setenv("GPU_WEEK_ANCHOR", "0")  # Monday
    anchor = budget.gpu_week_anchor(datetime(2026, 8, 12, 15, 0, tzinfo=timezone.utc))
    assert anchor.weekday() == 0


def test_gpu_week_sums_actuals_and_open_reserves(agent_workspace):
    budget.reserve_gpu("titanic", run_id="r-1", estimated_hours=3.5)
    budget.record_gpu_actual("titanic", run_id="r-1", hours_upper=3.3, hours_lower=1.5)
    budget.reserve_gpu("birdclef-2025", run_id="r-2", estimated_hours=4.0)  # still open

    week = budget.gpu_week(competition="titanic")

    # The settled run counts at its actual, not its estimate; the open one at its estimate.
    assert week["by_competition"] == {"birdclef-2025": 4.0, "titanic": 3.3}
    assert week["account_hours"] == 7.3
    assert week["open_reserved_hours"] == 4.0
    assert week["competition_hours"] == 3.3


def test_gpu_week_ignores_records_before_the_anchor(agent_workspace):
    path = ledger.gpu_ledger_path()
    stale = (budget.gpu_week_anchor() - timedelta(days=1)).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"event": "gpu_actual", "competition": "titanic", "run_id": "old", '
        f'"hours_upper": 20.0, "logged_at": "{stale}"}}\n',
        encoding="utf-8",
    )
    assert budget.gpu_week()["account_hours"] == 0.0


def test_gpu_manual_counts_toward_the_account_guard(agent_workspace):
    budget.record_gpu_manual(9.0, note="ran a notebook interactively")
    assert budget.gpu_week()["account_hours"] == 9.0


def test_gpu_allows_within_both_caps(agent_workspace):
    mission = dict(budget.MISSION_DEFAULTS, gpu_weekly_hours=15.0)
    verdict = budget.gpu_allows("titanic", 3.0, mission=mission)

    assert verdict["allowed"] is True
    assert verdict["ledger_is_lower_bound"] is True


def test_gpu_denied_by_per_competition_slice(agent_workspace):
    mission = dict(budget.MISSION_DEFAULTS, gpu_weekly_hours=5.0)
    budget.reserve_gpu("titanic", run_id="r-1", estimated_hours=4.0)

    verdict = budget.gpu_allows("titanic", 3.0, mission=mission)
    assert verdict["allowed"] is False
    assert "weekly slice" in verdict["reason"]


def test_gpu_denied_by_account_guard_across_competitions(agent_workspace):
    # A competition with a generous private slice still cannot break the shared guard.
    mission = dict(budget.MISSION_DEFAULTS, gpu_weekly_hours=30.0)
    budget.reserve_gpu("birdclef-2025", run_id="r-a", estimated_hours=24.0)

    verdict = budget.gpu_allows("titanic", 3.0, mission=mission)
    assert verdict["allowed"] is False
    assert "account would reach" in verdict["reason"]


def test_gpu_denied_by_per_run_cap(agent_workspace):
    mission = dict(budget.MISSION_DEFAULTS, max_gpu_hours_per_run=8.0)
    verdict = budget.gpu_allows("titanic", 9.5, mission=mission)

    assert verdict["allowed"] is False
    assert "checkpointed stages" in verdict["reason"]


def test_two_competitions_cannot_both_reserve_the_same_hours(agent_workspace):
    """The concurrency case the reserve-at-launch rule exists to prevent."""
    mission = dict(budget.MISSION_DEFAULTS, gpu_weekly_hours=30.0)
    # 21h already committed account-wide leaves 5h under the 26h guard.
    budget.record_gpu_manual(21.0)

    first = budget.gpu_allows("titanic", 3.0, mission=mission)
    assert first["allowed"] is True
    budget.reserve_gpu("titanic", run_id="r-1", estimated_hours=3.0)

    # The second loop sees the commitment immediately, before r-1 finishes.
    second = budget.gpu_allows("birdclef-2025", 3.0, mission=mission)
    assert second["allowed"] is False
