# SPDX-License-Identifier: MIT
"""Submission and GPU budget arithmetic — the loop's only spending authority.

Two scarce, externally visible resources are guarded here, and nowhere else:

* **Competition submissions.** A daily, team-wide cap that resets at 00:00 UTC.
  The agent may spend at most ``agent_submission_cap`` of them and must leave
  ``human_submission_reserve`` for the human operator.
* **GPU hours.** ~30h/week across the whole Kaggle account, so concurrent
  competition loops are spending from one pot. Each competition gets a slice
  (``gpu_weekly_hours``) under a global account guard.

Deliberately kept out of markdown: an agent that can re-derive its own budget
can also talk itself past it.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

from agent import ledger

logger = logging.getLogger(__name__)

# Submissions -------------------------------------------------------------
SUBMISSION_RESERVED = "submission_reserved"
SUBMISSION_RESULT = "submission_result"
SUBMISSION_RELEASED = "submission_released"

# GPU ---------------------------------------------------------------------
GPU_RESERVE = "gpu_reserve"
GPU_ACTUAL = "gpu_actual"
GPU_MANUAL = "gpu_manual"

# Kaggle publishes ~30 GPU h/week. Guard below it: the ledger cannot see
# interactive notebook usage, so it is a lower bound on true consumption.
GPU_NOMINAL_WEEKLY_HOURS = 30.0
GPU_GUARD_WEEKLY_HOURS = 26.0
DEFAULT_GPU_WEEK_ANCHOR = 5  # Saturday (Monday=0), matching Kaggle's usual reset

MISSION_DEFAULTS: dict[str, object] = {
    "metric": None,
    "direction": "higher_better",
    "deadline": None,
    "target_lb": None,
    "submit_threshold_pct": 1.0,
    "max_lanes": 1,
    "gpu_weekly_hours": 15.0,
    "max_gpu_hours_per_run": 8.0,
    "stop_before_hours": 12.0,
    "max_idle_ticks": 12,
    "max_consecutive_kernel_errors": 3,
    "human_submission_reserve": 2,
    "agent_submission_cap": 3,
}

_NUMERIC_KEYS = {
    "target_lb", "submit_threshold_pct", "gpu_weekly_hours",
    "max_gpu_hours_per_run", "stop_before_hours",
}
_INT_KEYS = {"max_lanes", "max_idle_ticks", "max_consecutive_kernel_errors",
             "human_submission_reserve", "agent_submission_cap"}

_KV_RE = re.compile(r"^([a-z_][a-z0-9_]*)\s*:\s*(.+?)\s*(?:#.*)?$", re.IGNORECASE)


# --------------------------------------------------------------------------
# MISSION.md
# --------------------------------------------------------------------------

def parse_mission(text: str) -> dict:
    """Parse ``key: value`` lines out of MISSION.md, ignoring prose.

    Lenient by design: the file is human-authored, and an unparsable line must
    degrade to the default rather than crash a tick mid-flight.
    """
    mission = dict(MISSION_DEFAULTS)
    in_code_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence or stripped.startswith(("#", "-", "*", ">")):
            continue
        match = _KV_RE.match(stripped)
        if not match:
            continue
        key, raw = match.group(1).lower(), match.group(2).strip()
        if key not in MISSION_DEFAULTS:
            continue
        if raw.lower() in {"none", "null", ""}:
            mission[key] = None
            continue
        try:
            if key in _INT_KEYS:
                mission[key] = int(float(raw))
            elif key in _NUMERIC_KEYS:
                mission[key] = float(raw)
            else:
                mission[key] = raw
        except ValueError:
            continue  # keep the default; STATE.md will show it
    return mission


def read_mission(slug: str) -> dict:
    path = ledger.mission_path(slug)
    if not path.exists():
        return dict(MISSION_DEFAULTS)
    return parse_mission(path.read_text(encoding="utf-8"))


def mission_deadline(mission: dict) -> datetime | None:
    raw = mission.get("deadline")
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# Submission budget
# --------------------------------------------------------------------------

def _utc_day(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%d")


def agent_submissions_today(slug: str, *, now: datetime | None = None) -> int:
    """Net reservations the agent holds today: reserved minus released."""
    day = _utc_day(now or ledger.utc_now())
    reserved = released = 0
    for record in ledger.read_jsonl(ledger.budget_ledger_path(slug)):
        if record.get("day") != day or record.get("actor") != "agent":
            continue
        if record.get("event") == SUBMISSION_RESERVED:
            reserved += 1
        elif record.get("event") == SUBMISSION_RELEASED:
            released += 1
    return max(0, reserved - released)


def unknown_quota(slug: str, *, limit_fallback: int = 5, source: str = "unavailable") -> dict:
    """A quota shape meaning "Kaggle could not tell us". Never claims headroom."""
    return {"competition": slug, "limit": limit_fallback, "limit_source": source,
            "used": None, "remaining": None, "exhausted": False}


def fetch_quota(slug: str, *, now: datetime | None = None) -> dict:
    """Ask Kaggle for today's quota, degrading rather than crashing the tick.

    A Kaggle outage, a missing token, or a forbidden submissions list must not
    end the loop. Falling back to an unknown quota is strictly conservative:
    the agent's own daily cap still applies, and Kaggle's quota error remains
    the authoritative backstop at submit time.
    """
    try:
        from submission_quota import submission_quota as _submission_quota

        return _submission_quota(slug, by_user=True, now=now)
    except Exception as exc:  # noqa: BLE001 — see docstring
        logger.warning("submission quota unavailable for %s: %s", slug, exc)
        return unknown_quota(slug)


def submission_budget(
    slug: str,
    *,
    now: datetime | None = None,
    quota: dict | None = None,
    mission: dict | None = None,
) -> dict:
    """Decide whether the agent may spend a submission slot right now.

    ``quota`` is injectable so tests never touch the network; in production it
    comes from ``submission_quota.submission_quota(slug, by_user=True)``.

    Two independent caps must both hold: the agent's own daily allowance, and
    the human's untouched reserve. Where Kaggle's usage count is unknown (a
    documented failure mode — account not joined, submissions list forbidden)
    the local ledger alone decides, and the result is flagged so STATE.md can
    say so plainly rather than implying certainty.
    """
    now = now or ledger.utc_now()
    mission = mission if mission is not None else read_mission(slug)
    if quota is None:
        quota = fetch_quota(slug, now=now)

    limit = int(quota.get("limit") or 0)
    kaggle_used = quota.get("used")
    reserve = int(mission.get("human_submission_reserve", 2))
    cap = int(mission.get("agent_submission_cap", 3))
    agent_used = agent_submissions_today(slug, now=now)
    quota_unknown = kaggle_used is None

    result = {
        "competition": slug,
        "day": _utc_day(now),
        "limit": limit,
        "limit_source": quota.get("limit_source"),
        "kaggle_used": kaggle_used,
        "agent_used": agent_used,
        "agent_cap": cap,
        "human_reserve": reserve,
        "human_used": None,
        "available": None,
        "quota_unknown": quota_unknown,
        "allowed": False,
        "reason": "",
    }

    if quota.get("exhausted"):
        result["reason"] = f"Kaggle reports the daily limit ({limit}) is exhausted"
        return result

    if limit and limit <= reserve:
        # Not a bug: on a tight-limit competition every slot belongs to the
        # human, so the agent must escalate rather than quietly take one.
        result["reason"] = (
            f"daily limit {limit} <= human reserve {reserve}; the agent never submits here — "
            "submit manually"
        )
        return result

    if agent_used >= cap:
        result["reason"] = f"agent already spent its daily cap ({agent_used}/{cap})"
        return result

    if quota_unknown:
        result["allowed"] = True
        result["reason"] = (
            f"Kaggle usage unknown; allowing on the local ledger alone ({agent_used}/{cap} spent). "
            "Kaggle's own quota error is the authoritative backstop."
        )
        return result

    human_used = max(0, int(kaggle_used) - agent_used)
    human_headroom = max(0, reserve - human_used)
    available = limit - int(kaggle_used) - human_headroom
    result["human_used"] = human_used
    result["available"] = available

    if available < 1:
        result["reason"] = (
            f"no slot free after reserving {human_headroom} for the human "
            f"(limit {limit}, used {kaggle_used})"
        )
        return result

    result["allowed"] = True
    result["reason"] = f"agent {agent_used}/{cap} spent, {available} slot(s) free after human reserve"
    return result


def reserve_submission(slug: str, *, run_id: str, message: str, hypothesis_id: str | None = None,
                       quota_snapshot: dict | None = None, now: datetime | None = None) -> None:
    """Claim a slot BEFORE calling Kaggle. Raises if it cannot be persisted.

    Ordering matters: a crash between this write and the API call burns one
    agent slot, whereas the reverse ordering would risk double-spending a real
    submission. Fail towards under-spending.
    """
    ledger.append_jsonl(
        ledger.budget_ledger_path(slug),
        {
            "event": SUBMISSION_RESERVED,
            "day": _utc_day(now or ledger.utc_now()),
            "actor": "agent",
            "run_id": run_id,
            "hypothesis_id": hypothesis_id,
            "message": message,
            "quota_snapshot": quota_snapshot,
        },
        strict=True,
    )


def release_submission(slug: str, *, run_id: str, reason: str, now: datetime | None = None) -> None:
    """Give a slot back after Kaggle rejected the submission outright."""
    ledger.append_jsonl(
        ledger.budget_ledger_path(slug),
        {
            "event": SUBMISSION_RELEASED,
            "day": _utc_day(now or ledger.utc_now()),
            "actor": "agent",
            "run_id": run_id,
            "reason": reason,
        },
    )


def record_submission_result(slug: str, *, run_id: str, accepted: bool,
                             eval_status: str | None = None, public_score: str | None = None,
                             now: datetime | None = None) -> None:
    ledger.append_jsonl(
        ledger.budget_ledger_path(slug),
        {
            "event": SUBMISSION_RESULT,
            "day": _utc_day(now or ledger.utc_now()),
            "actor": "agent",
            "run_id": run_id,
            "accepted": accepted,
            "eval_status": eval_status,
            "public_score": public_score,
        },
    )


# --------------------------------------------------------------------------
# GPU budget
# --------------------------------------------------------------------------

def gpu_week_anchor(now: datetime | None = None, *, weekday: int | None = None) -> datetime:
    """Start of the current GPU week (default Saturday 00:00 UTC)."""
    import os

    now = (now or ledger.utc_now()).astimezone(timezone.utc)
    if weekday is None:
        raw = os.environ.get("GPU_WEEK_ANCHOR")
        try:
            weekday = int(raw) if raw is not None else DEFAULT_GPU_WEEK_ANCHOR
        except ValueError:
            weekday = DEFAULT_GPU_WEEK_ANCHOR
    weekday %= 7
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    days_since = (midnight.weekday() - weekday) % 7
    return midnight - timedelta(days=days_since)


def _parse_stamp(raw) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def gpu_week(*, now: datetime | None = None, competition: str | None = None) -> dict:
    """Account-wide and per-competition GPU hours consumed this week.

    An open ``gpu_reserve`` (no matching ``gpu_actual`` yet) counts at its
    estimate. That is what stops two concurrent competition loops each
    launching a 6h run into 5h of remaining quota: the commitment is visible
    the moment the kernel is pushed, not when it finishes.
    """
    now = now or ledger.utc_now()
    anchor = gpu_week_anchor(now)
    reserves: dict[str, dict] = {}
    actual_runs: set[str] = set()
    by_competition: dict[str, float] = {}

    def add(comp: str, hours: float) -> None:
        by_competition[comp] = by_competition.get(comp, 0.0) + max(0.0, hours)

    records = ledger.read_jsonl(ledger.gpu_ledger_path())
    for record in records:
        stamp = _parse_stamp(record.get("logged_at"))
        if stamp is None or stamp < anchor:
            continue
        event = record.get("event")
        comp = record.get("competition") or "*"
        if event == GPU_RESERVE:
            reserves[record.get("run_id")] = record
        elif event == GPU_ACTUAL:
            actual_runs.add(record.get("run_id"))
            add(comp, float(record.get("hours_upper") or 0.0))
        elif event == GPU_MANUAL:
            add(comp, float(record.get("hours_upper") or 0.0))

    open_reserved = 0.0
    for run_id, record in reserves.items():
        if run_id in actual_runs:
            continue  # settled; the actual reading supersedes the estimate
        hours = float(record.get("estimated_hours") or 0.0)
        open_reserved += hours
        add(record.get("competition") or "*", hours)

    account_hours = sum(by_competition.values())
    next_reset = anchor + timedelta(days=7)
    state = {
        "anchor": anchor.isoformat(timespec="seconds"),
        "resets_at": next_reset.isoformat(timespec="seconds"),
        "hours_until_reset": round((next_reset - now).total_seconds() / 3600.0, 2),
        "account_hours": round(account_hours, 2),
        "account_guard": GPU_GUARD_WEEKLY_HOURS,
        "account_nominal": GPU_NOMINAL_WEEKLY_HOURS,
        "account_remaining": round(max(0.0, GPU_GUARD_WEEKLY_HOURS - account_hours), 2),
        "open_reserved_hours": round(open_reserved, 2),
        "by_competition": {k: round(v, 2) for k, v in sorted(by_competition.items())},
    }
    if competition is not None:
        state["competition"] = competition
        state["competition_hours"] = round(by_competition.get(competition, 0.0), 2)
    return state


def gpu_allows(slug: str, estimated_hours: float, *, now: datetime | None = None,
               mission: dict | None = None, week: dict | None = None) -> dict:
    """Check a proposed GPU run against both the per-competition cap and the guard."""
    mission = mission if mission is not None else read_mission(slug)
    week = week if week is not None else gpu_week(now=now, competition=slug)
    cap = float(mission.get("gpu_weekly_hours") or 0.0)
    per_run_cap = float(mission.get("max_gpu_hours_per_run") or 0.0)
    used = float(week.get("competition_hours", week.get("by_competition", {}).get(slug, 0.0)))
    account = float(week["account_hours"])

    result = {
        "allowed": True,
        "reason": "",
        "estimated_hours": estimated_hours,
        "competition_used": used,
        "competition_cap": cap,
        "account_used": account,
        "account_guard": GPU_GUARD_WEEKLY_HOURS,
        "resets_at": week["resets_at"],
        "ledger_is_lower_bound": True,
    }

    if per_run_cap and estimated_hours > per_run_cap:
        result.update(allowed=False, reason=(
            f"{estimated_hours}h exceeds max_gpu_hours_per_run ({per_run_cap}h); "
            "split the job into checkpointed stages"))
    elif cap and used + estimated_hours > cap:
        result.update(allowed=False, reason=(
            f"{slug} would reach {used + estimated_hours:.1f}h of its {cap}h weekly slice "
            f"(resets {week['resets_at']})"))
    elif account + estimated_hours > GPU_GUARD_WEEKLY_HOURS:
        result.update(allowed=False, reason=(
            f"account would reach {account + estimated_hours:.1f}h of the "
            f"{GPU_GUARD_WEEKLY_HOURS}h guard shared by all competition loops"))
    else:
        result["reason"] = (
            f"{used + estimated_hours:.1f}h/{cap}h for {slug}, "
            f"{account + estimated_hours:.1f}h/{GPU_GUARD_WEEKLY_HOURS}h account-wide")
    return result


def reserve_gpu(slug: str, *, run_id: str, estimated_hours: float) -> None:
    ledger.append_jsonl(
        ledger.gpu_ledger_path(),
        {"event": GPU_RESERVE, "competition": slug, "run_id": run_id,
         "estimated_hours": estimated_hours},
    )


def record_gpu_actual(slug: str, *, run_id: str, hours_upper: float,
                      hours_lower: float | None = None,
                      basis: str = "wallclock_push_to_observed_terminal") -> None:
    ledger.append_jsonl(
        ledger.gpu_ledger_path(),
        {"event": GPU_ACTUAL, "competition": slug, "run_id": run_id,
         "hours_upper": round(hours_upper, 3),
         "hours_lower": None if hours_lower is None else round(hours_lower, 3),
         "basis": basis},
    )


def record_gpu_manual(hours_upper: float, *, competition: str = "*", note: str = "") -> None:
    """Let a human account for interactive Kaggle usage the loop cannot see."""
    ledger.append_jsonl(
        ledger.gpu_ledger_path(),
        {"event": GPU_MANUAL, "competition": competition, "actor": "human",
         "hours_upper": hours_upper, "note": note},
    )


def write_mission_template(slug: str, *, metric: str = "", direction: str = "higher_better",
                           deadline: str = "") -> Path:
    """Seed MISSION.md from the defaults. Never overwrites an existing file."""
    path = ledger.mission_path(slug)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Mission — {slug}",
        "",
        f"metric: {metric}",
        f"direction: {direction}",
        f"deadline: {deadline}",
        "target_lb:",
        f"submit_threshold_pct: {MISSION_DEFAULTS['submit_threshold_pct']}",
        f"max_lanes: {MISSION_DEFAULTS['max_lanes']}",
        f"gpu_weekly_hours: {MISSION_DEFAULTS['gpu_weekly_hours']}",
        f"max_gpu_hours_per_run: {MISSION_DEFAULTS['max_gpu_hours_per_run']}",
        f"stop_before_hours: {MISSION_DEFAULTS['stop_before_hours']}",
        f"max_idle_ticks: {MISSION_DEFAULTS['max_idle_ticks']}",
        f"max_consecutive_kernel_errors: {MISSION_DEFAULTS['max_consecutive_kernel_errors']}",
        f"human_submission_reserve: {MISSION_DEFAULTS['human_submission_reserve']}",
        f"agent_submission_cap: {MISSION_DEFAULTS['agent_submission_cap']}",
        "",
        "## Out of bounds",
        "- Never submit a run whose CV is worse than the current champion.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
