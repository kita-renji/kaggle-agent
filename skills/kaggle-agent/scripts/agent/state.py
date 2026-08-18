# SPDX-License-Identifier: MIT
"""Sense the workspace, reconcile reality, and choose the one phase to run.

This module is the loop's decision code. It answers a single question per tick
— *what is the one thing to do now?* — from disk state plus a handful of
bounded Kaggle calls, and renders ``STATE.md`` so a fresh context can pick the
work up without replaying any history.

The ladder is deliberately ordered and first-match-wins. Anything requiring
taste (which hypothesis, how to read a log, whether a CV gain is real) is left
to the markdown phase docs; anything requiring consistency across a context
boundary is decided here.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone

from agent import budget, ledger, runs

PHASES = ("STOP", "BOOTSTRAP", "HARVEST", "SUBMIT", "LAUNCH", "REFILL", "WAIT")

MIN_READY_HYPOTHESES = 3
BOOTSTRAP_ARTIFACTS = ("brief.md",)

# Wakeup pacing. ScheduleWakeup clamps to [60, 3600]; these stay inside it.
WAKE_WORK_QUEUED = 60
WAKE_AWAITING_EVAL = 600
WAKE_MAX = 3600
WAKE_IDLE_BASE = 600

# The separator is optional and accepts any dash the author happens to type:
# backlog.md is hand-edited, and a heading must never be silently skipped
# because someone used "--" instead of an em dash.
_BACKLOG_HEADING = re.compile(
    r"^##\s+(?P<id>H-\d+)\s*(?:[‐-―\-:]+\s*)?(?P<title>.*?)\s*"
    r"(?P<tags>(?:\[[^\]]*\]\s*)*)$"
)
_TAG = re.compile(r"\[(?P<key>[a-z_]+)\s*:\s*(?P<value>[^\]]*)\]", re.IGNORECASE)
_RETIRED_HEADING = re.compile(r"^##\s+Retired\b", re.IGNORECASE)


# --------------------------------------------------------------------------
# Backlog
# --------------------------------------------------------------------------

def parse_backlog(text: str) -> list[dict]:
    """Extract hypothesis headings and their bracket tags.

    Everything below a ``## Retired`` heading is ignored: retired items are
    prose for the agent's memory, not queue entries.
    """
    items: list[dict] = []
    for line in text.splitlines():
        if _RETIRED_HEADING.match(line.strip()):
            break
        match = _BACKLOG_HEADING.match(line.strip())
        if not match:
            continue
        tags = {m.group("key").lower(): m.group("value").strip().lower()
                for m in _TAG.finditer(match.group("tags") or "")}
        items.append({
            "id": match.group("id"),
            "title": match.group("title").strip(),
            "priority": tags.get("priority", "med"),
            "cost": tags.get("cost", ""),
            "status": tags.get("status", "ready"),
            "needs_gpu": tags.get("cost", "").startswith("gpu"),
        })
    return items


def read_backlog(slug: str) -> list[dict]:
    path = ledger.backlog_path(slug)
    if not path.exists():
        return []
    return parse_backlog(path.read_text(encoding="utf-8"))


def ready_items(items: list[dict], *, gpu_allowed: bool = True) -> list[dict]:
    ready = [i for i in items if i["status"] == "ready"]
    if not gpu_allowed:
        ready = [i for i in ready if not i["needs_gpu"]]
    order = {"high": 0, "med": 1, "medium": 1, "low": 2}
    return sorted(ready, key=lambda i: order.get(i["priority"], 1))


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------

def is_better(candidate: float, incumbent: float | None, direction: str) -> bool:
    if candidate is None:
        return False
    if incumbent is None:
        return True
    return candidate < incumbent if direction == "lower_better" else candidate > incumbent


def relative_gain(candidate: float, incumbent: float, direction: str) -> float:
    """Percentage improvement of candidate over incumbent, sign-aware."""
    scale = abs(incumbent)
    delta = (incumbent - candidate) if direction == "lower_better" else (candidate - incumbent)
    if scale == 0:
        return float("inf") if delta > 0 else (0.0 if delta == 0 else float("-inf"))
    return 100.0 * delta / scale


def champion(rows: list[dict], direction: str) -> dict | None:
    """Best scored run the agent chose to keep."""
    best = None
    for row in rows:
        if not row.get("scored") or row.get("cv_score") is None:
            continue
        if row.get("verdict") == "reject":
            continue
        if best is None or is_better(row["cv_score"], best["cv_score"], direction):
            best = row
    return best


def submitted_run_ids(slug: str) -> dict[str, dict]:
    """Runs already submitted, keyed by run_id, with any known public score.

    Two sources are merged because they fail differently: ``budget.jsonl``
    knows about a reservation the instant it is made (even if the process then
    died), while ``data/submissions.jsonl`` is what Kaggle actually accepted
    and carries the score. A run present in either must never be re-submitted.
    """
    found: dict[str, dict] = {}

    for record in ledger.read_jsonl(ledger.budget_ledger_path(slug)):
        run_id = record.get("run_id")
        if not run_id:
            continue
        event = record.get("event")
        if event == budget.SUBMISSION_RESERVED:
            found.setdefault(run_id, {"run_id": run_id, "public_score": None, "eval_status": None})
        elif event == budget.SUBMISSION_RELEASED:
            found.pop(run_id, None)
        elif event == budget.SUBMISSION_RESULT and run_id in found:
            found[run_id].update(
                eval_status=record.get("eval_status"),
                public_score=record.get("public_score"),
            )

    from submission_log import read_records, submission_attempts

    for attempt in submission_attempts(read_records(), competition=slug):
        message = str(attempt.get("message") or "")
        run_id = message.split()[0] if message else ""
        if not run_id.startswith("r-"):
            continue
        entry = found.setdefault(run_id, {"run_id": run_id, "public_score": None, "eval_status": None})
        entry["accepted"] = attempt.get("accepted")
        if attempt.get("public_score") is not None:
            entry["public_score"] = attempt.get("public_score")
        if attempt.get("eval_status") is not None:
            entry["eval_status"] = attempt.get("eval_status")
    return found


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_leaderboard(rows: list[dict], submitted: dict[str, dict], direction: str) -> dict | None:
    best = None
    for row in rows:
        entry = submitted.get(row["run_id"])
        score = _as_float(entry.get("public_score")) if entry else None
        if score is None:
            continue
        if best is None or is_better(score, best["lb_score"], direction):
            best = {"run_id": row["run_id"], "lb_score": score,
                    "hypothesis_id": row.get("hypothesis_id")}
    return best


def submit_candidate(rows: list[dict], submitted: dict[str, dict], mission: dict) -> dict | None:
    """The best kept, unsubmitted run that clears the improvement threshold.

    The threshold is measured against the best CV *already submitted*, not the
    champion: resubmitting a marginal variant of something Kaggle has already
    scored spends a slot for almost no information.
    """
    direction = mission.get("direction", "higher_better")
    threshold = float(mission.get("submit_threshold_pct") or 0.0)

    best_submitted_cv = None
    for row in rows:
        if row["run_id"] in submitted and row.get("cv_score") is not None:
            if is_better(row["cv_score"], best_submitted_cv, direction):
                best_submitted_cv = row["cv_score"]

    candidate = None
    for row in rows:
        if not row.get("scored") or row.get("cv_score") is None:
            continue
        if row.get("verdict") != "keep" or row["run_id"] in submitted:
            continue
        if not has_submission_file(row):
            continue
        if best_submitted_cv is not None:
            if relative_gain(row["cv_score"], best_submitted_cv, direction) < threshold:
                continue
        if candidate is None or is_better(row["cv_score"], candidate["cv_score"], direction):
            candidate = row
    return candidate


def has_submission_file(row: dict) -> bool:
    return any(str(name).endswith("submission.csv") for name in (row.get("files") or []))


# --------------------------------------------------------------------------
# Reconcile
# --------------------------------------------------------------------------

def reconcile(slug: str, *, api=None, mission: dict | None = None) -> list[str]:
    """Repair state left inconsistent by a crashed tick. Returns what it did.

    Runs before any decision, so the ladder always sees the truth rather than
    whatever the last process managed to write before it died.
    """
    mission = mission if mission is not None else budget.read_mission(slug)
    actions: list[str] = []

    for row in runs.active_runs(slug):
        if row.get("finished"):
            continue  # downloaded already; HARVEST owns it now
        result = runs.poll(slug, row["run_id"], api=api, row=row)
        status = result.get("status")

        if result.get("terminal"):
            if status == "complete":
                runs.collect(slug, row["run_id"], api=api, status=status, row=row)
                actions.append(f"collected {row['run_id']} ({status})")
            else:
                failure = result.get("failure_message") or status
                runs.collect(slug, row["run_id"], api=api, status=status, row=row)
                actions.append(f"collected failed run {row['run_id']} ({failure})")
        elif runs.is_overdue(row, mission):
            runs.abandon(
                slug, row["run_id"],
                reason=(f"no terminal status after "
                        f"{(runs.elapsed_seconds(row) or 0) / 3600:.1f}h; kernel may be stuck"),
                status="timeout", row=row,
            )
            actions.append(f"abandoned overdue {row['run_id']}")
    return actions


def reconcile_submissions(slug: str, *, api=None) -> list[str]:
    """Fold in evaluation scores for submissions the loop stopped polling.

    ``agent_submit`` returns the moment Kaggle accepts a submission, so the
    score always arrives on some later tick — this is where it lands.
    """
    submitted = submitted_run_ids(slug)
    pending = {rid: e for rid, e in submitted.items()
               if e.get("public_score") is None and e.get("eval_status") in (None, "pending", "timeout")}
    if not pending or api is None:
        return []

    try:
        subs = api.competition_submissions(slug)
    except Exception:  # noqa: BLE001 — a transient API error just defers this
        return []

    actions: list[str] = []
    for submission in subs or []:
        description = str(getattr(submission, "description", "") or "")
        run_id = description.split()[0] if description else ""
        if run_id not in pending:
            continue
        status = getattr(getattr(submission, "status", None), "name", "").lower()
        if status not in {"complete", "error"}:
            continue
        score = getattr(submission, "public_score", None)
        budget.record_submission_result(
            slug, run_id=run_id, accepted=True, eval_status=status,
            public_score=None if score is None else str(score),
        )
        actions.append(f"scored submission {run_id} ({status}: {score})")
        pending.pop(run_id, None)
    return actions


# --------------------------------------------------------------------------
# Stop conditions
# --------------------------------------------------------------------------

def stop_reasons(slug: str, *, mission: dict, rows: list[dict], best_lb: dict | None,
                 idle_ticks: int, gpu_week: dict, backlog: list[dict],
                 now: datetime | None = None) -> list[str]:
    now = now or ledger.utc_now()
    reasons: list[str] = []

    if ledger.halt_path(slug).exists():
        reasons.append("HALT file present — human kill switch")

    deadline = budget.mission_deadline(mission)
    if deadline is not None:
        margin = float(mission.get("stop_before_hours") or 0.0)
        if (deadline - now).total_seconds() <= margin * 3600:
            reasons.append(f"within {margin}h of the deadline ({deadline:%Y-%m-%dT%H:%MZ})")

    target = mission.get("target_lb")
    if target is not None and best_lb is not None:
        if not is_better(float(target), best_lb["lb_score"], mission.get("direction", "higher_better")):
            reasons.append(f"target LB {target} reached (best {best_lb['lb_score']})")

    max_idle = int(mission.get("max_idle_ticks") or 0)
    if max_idle and idle_ticks > max_idle:
        reasons.append(f"{idle_ticks} idle ticks with nothing to do (limit {max_idle})")

    max_errors = int(mission.get("max_consecutive_kernel_errors") or 0)
    errors = runs.consecutive_kernel_errors(rows)
    if max_errors and errors >= max_errors:
        reasons.append(f"{errors} kernel runs failed in a row — diagnose before retrying")

    ready = ready_items(backlog)
    if not ready and gpu_week["account_remaining"] <= 0:
        reasons.append("GPU guard exhausted and no CPU work queued")

    return reasons


# --------------------------------------------------------------------------
# Tick state
# --------------------------------------------------------------------------

def fingerprint(*, rows: list[dict], slug: str, agent_used: int, gpu_week: dict,
                backlog: list[dict]) -> str:
    """Hash of everything a WAIT tick would look at.

    When this is unchanged, the tick has nothing new to reason about and must
    cost essentially nothing — that is the whole token-burn guard.
    """
    run_lines = [
        f"{r['run_id']}:{r.get('status')}:{r.get('observed_status')}:"
        f"{int(r.get('finished', False))}{int(r.get('scored', False))}{int(r.get('abandoned', False))}"
        for r in rows[-10:]
    ]
    submissions = ledger.read_jsonl(ledger.budget_ledger_path(slug))
    payload = "|".join([
        *run_lines,
        json.dumps(submissions[-1] if submissions else {}, sort_keys=True, default=str),
        ",".join(sorted(i["id"] for i in ready_items(backlog))),
        str(agent_used),
        str(round(gpu_week["account_hours"], 1)),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def tick_state(slug: str, *, offline: bool = False, api=None, now: datetime | None = None,
               quota: dict | None = None) -> dict:
    """Assemble everything the tick needs, and name the one phase to run."""
    now = now or ledger.utc_now()
    mission = budget.read_mission(slug)
    direction = mission.get("direction", "higher_better")

    if not offline and api is None:
        from submit_kernel import get_api

        api = get_api()

    reconciled: list[str] = []
    if not offline:
        reconciled += reconcile(slug, api=api, mission=mission)
        reconciled += reconcile_submissions(slug, api=api)

    rows = runs.load_runs(slug)
    active = [r for r in rows if r["active"] and not r["finished"]]
    harvest_queue = runs.awaiting_harvest(slug, rows=rows)
    backlog = read_backlog(slug)
    submitted = submitted_run_ids(slug)
    gpu = budget.gpu_week(now=now, competition=slug)

    if offline and quota is None:
        # Offline is the same epistemic state as "Kaggle would not tell us":
        # decide from the local ledger alone, which can only under-spend.
        # agent_submit re-checks against the live quota before it commits.
        quota = budget.unknown_quota(slug, source="offline")
    sub_budget = budget.submission_budget(slug, now=now, quota=quota, mission=mission)

    gpu_headroom = gpu["account_remaining"] > 0 and (
        float(mission.get("gpu_weekly_hours") or 0.0) - gpu.get("competition_hours", 0.0)) > 0
    ready = ready_items(backlog, gpu_allowed=gpu_headroom)
    champ = champion(rows, direction)
    best_lb = best_leaderboard(rows, submitted, direction)
    candidate = submit_candidate(rows, submitted, mission)

    prior = ledger.read_json(ledger.tick_state_path(slug), default={}) or {}
    print_ready = ready_items(backlog)
    mark = fingerprint(rows=rows, slug=slug, agent_used=sub_budget["agent_used"],
                       gpu_week=gpu, backlog=backlog)
    unchanged = mark == prior.get("fingerprint")
    idle_ticks = int(prior.get("idle_ticks", 0)) + 1 if unchanged else 0

    stops = stop_reasons(slug, mission=mission, rows=rows, best_lb=best_lb,
                         idle_ticks=idle_ticks, gpu_week=gpu, backlog=backlog, now=now)

    state = {
        "competition": slug,
        "generated_at": ledger.utc_stamp(now),
        "tick": int(prior.get("tick", 0)) + 1,
        "last_phase": prior.get("phase"),
        "fingerprint": mark,
        "fingerprint_unchanged": unchanged,
        "idle_ticks": idle_ticks,
        "offline": offline,
        "reconciled": reconciled,
        "mission": mission,
        "direction": direction,
        "deadline": None,
        "runs": rows,
        "active_runs": active,
        "harvest_queue": harvest_queue,
        "backlog": backlog,
        "ready": print_ready,
        "ready_launchable": ready,
        "champion": champ,
        "best_lb": best_lb,
        "submitted": submitted,
        "submit_candidate": candidate,
        "submission_budget": sub_budget,
        "gpu_week": gpu,
        "gpu_headroom": gpu_headroom,
        "stop_reasons": stops,
        "bootstrap_missing": bootstrap_gaps(slug),
    }

    deadline = budget.mission_deadline(mission)
    if deadline is not None:
        state["deadline"] = deadline.isoformat(timespec="seconds")
        state["hours_to_deadline"] = round((deadline - now).total_seconds() / 3600.0, 1)

    phase, action = next_phase(state)
    state["next_phase"] = phase
    state["next_action"] = action
    state["wake_seconds"] = wake_seconds(state)
    return state


def bootstrap_gaps(slug: str) -> list[str]:
    missing = []
    if not ledger.mission_path(slug).exists():
        missing.append("MISSION.md")
    research = ledger.research_dir(slug)
    for name in BOOTSTRAP_ARTIFACTS:
        if not (research / name).exists():
            missing.append(f"research/{name}")
    if not ledger.backlog_path(slug).exists():
        missing.append("backlog.md")
    return missing


def next_phase(state: dict) -> tuple[str, str]:
    """The ladder. First match wins; exactly one phase runs per tick."""
    if state["stop_reasons"]:
        return "STOP", "; ".join(state["stop_reasons"])

    if state["bootstrap_missing"]:
        return "BOOTSTRAP", (
            "missing " + ", ".join(state["bootstrap_missing"])
            + " — research the competition and seed the backlog")

    if state["harvest_queue"]:
        row = state["harvest_queue"][0]
        return "HARVEST", (
            f"{row['run_id']} finished ({row.get('status')}); read "
            f"{row.get('output_dir')}/metrics.json then the log, and record a verdict")

    candidate = state["submit_candidate"]
    if candidate is not None and state["submission_budget"]["allowed"]:
        return "SUBMIT", (
            f"{candidate['run_id']} (CV {candidate.get('cv_score')}) beats the best submitted "
            f"score; {state['submission_budget']['reason']}")

    lanes = int(state["mission"].get("max_lanes", 1))
    if len(state["active_runs"]) < lanes and state["ready_launchable"]:
        item = state["ready_launchable"][0]
        return "LAUNCH", f"build and push {item['id']} — {item['title']} ({item['cost'] or 'cost unknown'})"

    if len(state["ready"]) < MIN_READY_HYPOTHESES:
        return "REFILL", (
            f"only {len(state['ready'])} ready hypothesis(es); generate more from the "
            "research brief and what the last runs showed")

    if state["active_runs"]:
        row = state["active_runs"][0]
        elapsed = runs.elapsed_seconds(row)
        return "WAIT", (
            f"{row['run_id']} still running"
            + (f" ({elapsed / 3600:.1f}h elapsed)" if elapsed else "")
            + "; do not push another kernel")

    if state["ready"] and not state["ready_launchable"]:
        return "WAIT", "GPU budget is spent and every ready hypothesis needs GPU; waiting for the weekly reset"

    return "WAIT", "nothing actionable"


def wake_seconds(state: dict) -> int:
    phase = state["next_phase"] if "next_phase" in state else next_phase(state)[0]

    if phase == "STOP":
        return 0
    if phase in {"HARVEST", "SUBMIT", "LAUNCH", "REFILL"} or phase == "BOOTSTRAP":
        return WAKE_WORK_QUEUED

    pending_eval = any(
        e.get("public_score") is None and e.get("eval_status") in (None, "pending")
        for e in state["submitted"].values()
    )
    if pending_eval:
        return WAKE_AWAITING_EVAL

    if state["active_runs"]:
        row = state["active_runs"][0]
        expected_min = row.get("expected_runtime_min")
        elapsed = runs.elapsed_seconds(row) or 0.0
        if expected_min:
            remaining = expected_min * 60 - elapsed
            if remaining > 3600:
                return WAKE_MAX
            return max(WAKE_AWAITING_EVAL, min(WAKE_MAX, int(remaining / 2)))
        return WAKE_MAX

    idle = max(1, state["idle_ticks"])
    return min(WAKE_MAX, WAKE_IDLE_BASE * (2 ** (idle - 1)))


def write_tick_state(slug: str, state: dict) -> None:
    ledger.write_json(ledger.tick_state_path(slug), {
        "fingerprint": state["fingerprint"],
        "idle_ticks": state["idle_ticks"],
        "tick": state["tick"],
        "phase": state["next_phase"],
        "updated_at": state["generated_at"],
    })


# --------------------------------------------------------------------------
# STATE.md
# --------------------------------------------------------------------------

def _fmt(value, default="—"):
    return default if value is None else value


def render_state_md(state: dict) -> str:
    slug = state["competition"]
    mission = state["mission"]
    sub = state["submission_budget"]
    gpu = state["gpu_week"]
    lines = [
        "<!-- GENERATED by agent_state.py — do not edit; edit MISSION.md or backlog.md instead -->",
        f"# STATE — {slug}",
        "",
        f"generated: {state['generated_at']}",
        f"tick: {state['tick']}   last_phase: {_fmt(state['last_phase'])}   "
        f"next_phase: {state['next_phase']}",
        f"fingerprint: {state['fingerprint']}   idle_ticks: {state['idle_ticks']}"
        + ("   (offline sense)" if state["offline"] else ""),
        f"metric: {_fmt(mission.get('metric'))} ({state['direction'].replace('_', ' ')})",
    ]
    if state.get("deadline"):
        lines.append(f"deadline: {state['deadline']} ({state.get('hours_to_deadline')}h left)")

    if state["reconciled"]:
        lines += ["", "## Reconciled this tick"] + [f"- {a}" for a in state["reconciled"]]

    champ = state["champion"]
    best_lb = state["best_lb"]
    lines += [
        "",
        "## Scores",
        f"champion CV : {_fmt(champ and champ.get('cv_score'))}"
        + (f"  ({champ['run_id']}, {_fmt(champ.get('hypothesis_id'))})" if champ else ""),
        f"best LB     : {_fmt(best_lb and best_lb.get('lb_score'))}"
        + (f"  ({best_lb['run_id']})" if best_lb else ""),
        f"target LB   : {_fmt(mission.get('target_lb'))}",
        "",
        "## Runs",
        f"active: {', '.join(r['run_id'] for r in state['active_runs']) or 'none'}",
        f"awaiting harvest: {', '.join(r['run_id'] for r in state['harvest_queue']) or 'none'}",
    ]
    recent = state["runs"][-3:]
    if recent:
        lines.append("last 3: " + " | ".join(
            f"{r['run_id']} {r.get('status')}"
            + ("/scored" if r.get("scored") else "/abandoned" if r.get("abandoned") else "")
            for r in reversed(recent)))

    verdict = "YES" if sub["allowed"] else "NO"
    lines += [
        "",
        "## Budget",
        f"submissions ({sub['day']} UTC): agent {sub['agent_used']}/{sub['agent_cap']}, "
        f"human reserve {sub['human_reserve']}, kaggle used {_fmt(sub['kaggle_used'], 'unknown')}"
        f"/{_fmt(sub['limit'], '?')}",
        f"agent may submit: {verdict} — {sub['reason']}",
        f"GPU week (resets {gpu['resets_at']}, {gpu['hours_until_reset']}h): "
        f"{slug} {gpu.get('competition_hours', 0.0)}h/{mission.get('gpu_weekly_hours')}h cap, "
        f"account {gpu['account_hours']}h/{gpu['account_guard']}h guard "
        f"({gpu['account_nominal']}h nominal)",
        "  !! the GPU ledger is a LOWER BOUND — interactive Kaggle runs are invisible to it",
    ]

    ready = state["ready"]
    running = [i for i in state["backlog"] if i["status"] == "running"]
    blocked = [i for i in state["backlog"] if i["status"] == "blocked"]
    lines += [
        "",
        "## Backlog",
        "ready: " + (", ".join(f"{i['id']} ({i['priority']}, {i['cost'] or '?'})" for i in ready) or "none"),
        "running: " + (", ".join(i["id"] for i in running) or "none")
        + "   blocked: " + (", ".join(i["id"] for i in blocked) or "none"),
    ]
    if ready and not state["ready_launchable"]:
        lines.append("note: every ready item needs GPU, and the GPU budget is spent")

    lines += [
        "",
        "## Next action",
        f"{state['next_phase']}: {state['next_action']}",
        f"wake in: {state['wake_seconds']}s",
        "",
    ]
    return "\n".join(lines)


def write_state_md(slug: str, state: dict) -> None:
    path = ledger.state_md_path(slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_state_md(state), encoding="utf-8")


def journal_entry(state: dict, notes: list[str]) -> str:
    stamp = state["generated_at"]
    lines = [f"## {stamp} — tick {state['tick']} — {state['next_phase']}"]
    lines += [f"- {note}" for note in notes]
    lines.append(f"- Wakeup {state['wake_seconds']}s.")
    return "\n".join(lines) + "\n"
