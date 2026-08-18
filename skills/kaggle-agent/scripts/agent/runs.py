# SPDX-License-Identifier: MIT
"""Non-blocking remote-run lifecycle: launch, poll, collect, abandon.

``run_kernel.py`` and ``submit_kernel.py`` block for hours inside their polling
loops, which is right for a human at a terminal and wrong for a loop tick. This
module reuses their *functions* — ``push_kernel``, ``download_outputs`` — and
supplies control flow where every remote interaction is single-shot:

    launch()  push, write the ledger, return in seconds
    poll()    exactly one kernels_status call
    collect() exactly one output download, once terminal

A run's lifetime therefore spans many ticks, and every fact about it lives in
``runs.jsonl`` rather than in a conversation that may be compacted away.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from agent import budget, ledger

logger = logging.getLogger(__name__)

RUN_STARTED = "run_started"
RUN_OBSERVED = "run_observed"
RUN_FINISHED = "run_finished"
RUN_SCORED = "run_scored"
RUN_ABANDONED = "run_abandoned"

TERMINAL_KERNEL_STATUSES = {"complete", "error", "cancel_acknowledged"}
DEFAULT_LANE = "default"
METRICS_FILENAME = "metrics.json"
DEFAULT_MAX_OUTPUT_MB = 500
SECONDS_PER_HOUR = 3600.0

# Bucket observation times so a re-poll of an unchanged run does not spam the
# ledger with a near-identical row every tick.
_OBSERVE_BUCKET_SECONDS = 900


class LaunchError(RuntimeError):
    """A run could not be started. Carries a human-readable reason."""


# --------------------------------------------------------------------------
# Folding the event log into run rows
# --------------------------------------------------------------------------

def fold_runs(records: list[dict]) -> list[dict]:
    """Collapse the event stream into one row per run, oldest first.

    Same shape of fold as ``submission_log.submission_attempts``: later events
    layer onto the row opened by ``run_started``. Events for an unknown run_id
    are dropped rather than inventing a row, so a truncated or hand-edited
    ledger degrades quietly.
    """
    rows: dict[str, dict] = {}
    for record in records:
        run_id = record.get("run_id")
        if not run_id:
            continue
        event = record.get("event")

        if event == RUN_STARTED:
            row = {k: v for k, v in record.items() if k not in {"event", "logged_at"}}
            row.update(
                started_at=record.get("logged_at"),
                status="running",
                observed_status=None,
                observed_at=None,
                elapsed_s=None,
                finished=False,
                scored=False,
                abandoned=False,
                verdict=None,
                cv_score=None,
                lb_score=None,
                submitted=False,
            )
            rows[run_id] = row
            continue

        row = rows.get(run_id)
        if row is None:
            logger.warning("ledger event %r references unknown run %s", event, run_id)
            continue

        if event == RUN_OBSERVED:
            row["observed_status"] = record.get("status")
            row["observed_at"] = record.get("logged_at")
            row["elapsed_s"] = record.get("elapsed_s")
        elif event == RUN_FINISHED:
            row.update(
                finished=True,
                status=record.get("status") or "complete",
                finished_at=record.get("logged_at"),
                output_dir=record.get("output_dir"),
                files=record.get("files") or [],
                log_path=record.get("log_path"),
                gpu_hours_upper=record.get("gpu_hours_upper"),
                elapsed_s_upper=record.get("elapsed_s_upper"),
                pruned_files=record.get("pruned_files") or [],
            )
        elif event == RUN_SCORED:
            row.update(
                scored=True,
                cv_score=record.get("cv_score"),
                cv_metric=record.get("cv_metric"),
                cv_direction=record.get("cv_direction"),
                cv_folds=record.get("cv_folds"),
                cv_std=record.get("cv_std"),
                verdict=record.get("verdict"),
                rationale=record.get("rationale"),
                parsed_from=record.get("parsed_from"),
                scored_at=record.get("logged_at"),
            )
        elif event == RUN_ABANDONED:
            row.update(
                abandoned=True,
                status=record.get("status") or "abandoned",
                abandoned_reason=record.get("reason"),
                abandoned_at=record.get("logged_at"),
            )

    for row in rows.values():
        # A run is active until it reaches a terminal *analysis* event. Merely
        # finishing on Kaggle is not terminal: its output still needs harvesting.
        row["active"] = not (row["scored"] or row["abandoned"])
    return list(rows.values())


def load_runs(slug: str) -> list[dict]:
    return fold_runs(ledger.read_jsonl(ledger.runs_ledger_path(slug)))


def active_runs(slug: str, *, lane: str | None = None, rows: list[dict] | None = None) -> list[dict]:
    rows = rows if rows is not None else load_runs(slug)
    active = [r for r in rows if r["active"]]
    if lane is not None:
        active = [r for r in active if (r.get("lane") or DEFAULT_LANE) == lane]
    return active


def find_run(slug: str, run_id: str, *, rows: list[dict] | None = None) -> dict | None:
    rows = rows if rows is not None else load_runs(slug)
    for row in rows:
        if row.get("run_id") == run_id:
            return row
    return None


def awaiting_harvest(slug: str, *, rows: list[dict] | None = None) -> list[dict]:
    """Runs whose output is downloaded but whose result has not been judged."""
    rows = rows if rows is not None else load_runs(slug)
    return [r for r in rows if r["finished"] and not r["scored"] and not r["abandoned"]]


# --------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------

def config_hash(code_path: Path, config: dict | None) -> str:
    """Fingerprint the exact experiment: code bytes plus declared config.

    Hashing the code file — not just the config dict — is what protects a
    *fresh context* from re-running an experiment it no longer remembers. A
    config that looks new but produces byte-identical code is the same run.
    """
    digest = hashlib.sha256()
    try:
        digest.update(code_path.read_bytes())
    except OSError as exc:
        raise LaunchError(f"cannot read code file {code_path}: {exc}") from exc
    digest.update(json.dumps(config or {}, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()[:16]


def duplicate_run(slug: str, digest: str, *, rows: list[dict] | None = None) -> dict | None:
    """A prior run with the same fingerprint that already reached a verdict."""
    rows = rows if rows is not None else load_runs(slug)
    for row in rows:
        if row.get("config_hash") == digest and (row["scored"] or row["abandoned"]):
            return row
    return None


# --------------------------------------------------------------------------
# Launch
# --------------------------------------------------------------------------

def _read_metadata(kernel_folder: Path) -> dict:
    """Read kernel-metadata.json, converting the helper's sys.exit into an error.

    ``submit_kernel.read_kernel_metadata`` prints and calls ``sys.exit(1)``,
    which would kill the tick process mid-loop. Trap it at the boundary.
    """
    from submit_kernel import read_kernel_metadata

    try:
        return read_kernel_metadata(str(kernel_folder))
    except SystemExit as exc:
        raise LaunchError(
            f"invalid kernel folder {kernel_folder}: kernel-metadata.json is missing, "
            "lacks 'id'/'code_file', or points at a code file that does not exist"
        ) from exc


def _push(api, kernel_folder: Path) -> int:
    from submit_kernel import push_kernel

    try:
        return push_kernel(api, str(kernel_folder))
    except SystemExit as exc:
        raise LaunchError(f"Kaggle rejected the push of {kernel_folder}") from exc


def launch(
    kernel_folder: str | Path,
    *,
    slug: str,
    hypothesis_id: str | None = None,
    config: dict | None = None,
    lane: str = DEFAULT_LANE,
    estimated_hours: float | None = None,
    expected_runtime_min: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    api=None,
    mission: dict | None = None,
) -> dict:
    """Push a kernel and record the run. Returns without waiting for it.

    Every guard runs before the push, so a refusal costs nothing: lane cap,
    config-hash dedupe, GPU budget, then the kernel lock. The lock is acquired
    here and released only by ``collect`` or ``abandon`` — it spans the run's
    whole life so no other process can push the same kernel while it executes.
    """
    kernel_folder = Path(kernel_folder).resolve()
    mission = mission if mission is not None else budget.read_mission(slug)
    meta = _read_metadata(kernel_folder)

    kernel_slug = meta["id"]
    code_path = kernel_folder / meta["code_file"]
    gpu = bool(meta.get("enable_gpu", False))
    digest = config_hash(code_path, config)

    rows = load_runs(slug)
    max_lanes = int(mission.get("max_lanes", 1))
    running = active_runs(slug, rows=rows)
    if len(running) >= max_lanes:
        raise LaunchError(
            f"{len(running)} run(s) already active and max_lanes is {max_lanes}: "
            + ", ".join(f"{r['run_id']} ({r.get('kernel')})" for r in running)
        )
    if any((r.get("lane") or DEFAULT_LANE) == lane for r in running):
        raise LaunchError(f"lane {lane!r} is already occupied")

    if not force:
        prior = duplicate_run(slug, digest, rows=rows)
        if prior is not None:
            raise LaunchError(
                f"identical experiment already ran as {prior['run_id']} "
                f"(config_hash {digest}, verdict {prior.get('verdict') or prior.get('status')}); "
                "change the code or config, or pass --force"
            )

    estimate = float(estimated_hours if estimated_hours is not None
                     else (expected_runtime_min or 0) / 60.0)
    gpu_verdict = None
    if gpu:
        gpu_verdict = budget.gpu_allows(slug, estimate, mission=mission)
        if not gpu_verdict["allowed"]:
            raise LaunchError(f"GPU budget refuses this run: {gpu_verdict['reason']}")

    plan = {
        "competition": slug,
        "kernel": kernel_slug,
        "kernel_folder": str(kernel_folder),
        "code_file": meta["code_file"],
        "gpu": gpu,
        "internet": bool(meta.get("enable_internet", False)),
        "config_hash": digest,
        "lane": lane,
        "hypothesis_id": hypothesis_id,
        "estimated_hours": estimate,
        "gpu_budget": gpu_verdict,
        "dataset_sources": meta.get("dataset_sources", []),
        "competition_sources": meta.get("competition_sources", []),
        "kernel_sources": meta.get("kernel_sources", []),
    }

    if dry_run or os.environ.get("KAGGLE_AGENT_DRY_RUN") == "1":
        plan.update(dry_run=True, run_id=None, version=None)
        return plan

    run_id = ledger.new_run_id()
    lock_name = ledger.kernel_lock_name(kernel_slug)
    ledger.acquire_lock(lock_name, run_id=run_id, competition=slug)
    try:
        if api is None:
            from submit_kernel import get_api

            api = get_api()
        version = _push(api, kernel_folder)
    except Exception:
        ledger.release_lock(lock_name)
        raise

    ledger.append_jsonl(
        ledger.runs_ledger_path(slug),
        {
            "event": RUN_STARTED,
            "run_id": run_id,
            "competition": slug,
            "hypothesis_id": hypothesis_id,
            "lane": lane,
            "kernel": kernel_slug,
            "version": version,
            "kernel_folder": str(kernel_folder),
            "gpu": gpu,
            "internet": plan["internet"],
            "config": config or {},
            "config_hash": digest,
            "expected_runtime_min": expected_runtime_min,
            "session": ledger.session_id(),
        },
    )
    if gpu:
        budget.reserve_gpu(slug, run_id=run_id, estimated_hours=estimate)

    plan.update(run_id=run_id, version=version, dry_run=False)
    return plan


# --------------------------------------------------------------------------
# Poll
# --------------------------------------------------------------------------

def poll(slug: str, run_id: str, *, api=None, row: dict | None = None) -> dict:
    """One ``kernels_status`` call. Records an observation only when it moved."""
    row = row if row is not None else find_run(slug, run_id)
    if row is None:
        raise LaunchError(f"unknown run {run_id}")

    if api is None:
        from submit_kernel import get_api

        api = get_api()

    try:
        response = api.kernels_status(row["kernel"])
        status = response.status.name.lower()
        failure = getattr(response, "failure_message", None)
    except Exception as exc:  # noqa: BLE001 — a transient API error is not fatal
        logger.warning("kernels_status failed for %s: %s", row["kernel"], exc)
        return {"run_id": run_id, "status": "unknown", "terminal": False, "error": str(exc)}

    elapsed = elapsed_seconds(row)
    result = {
        "run_id": run_id,
        "kernel": row["kernel"],
        "status": status,
        "failure_message": failure,
        "terminal": status in TERMINAL_KERNEL_STATUSES,
        "elapsed_s": elapsed,
    }

    moved = status != row.get("observed_status")
    bucket_changed = (
        elapsed is not None
        and int(elapsed // _OBSERVE_BUCKET_SECONDS)
        != int((row.get("elapsed_s") or 0) // _OBSERVE_BUCKET_SECONDS)
    )
    if moved or bucket_changed:
        ledger.append_jsonl(
            ledger.runs_ledger_path(slug),
            {"event": RUN_OBSERVED, "run_id": run_id, "status": status,
             "elapsed_s": None if elapsed is None else round(elapsed)},
        )
        result["recorded"] = True
    return result


def elapsed_seconds(row: dict, *, now: datetime | None = None) -> float | None:
    started = _parse_stamp(row.get("started_at"))
    if started is None:
        return None
    return ((now or ledger.utc_now()) - started).total_seconds()


def _parse_stamp(raw) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def is_overdue(row: dict, mission: dict, *, now: datetime | None = None) -> bool:
    """True when a run has outlived any plausible runtime and looks stuck."""
    elapsed = elapsed_seconds(row, now=now)
    if elapsed is None:
        return False
    ceiling = (float(mission.get("max_gpu_hours_per_run") or 8.0) + 2.0) * SECONDS_PER_HOUR
    return elapsed > ceiling


# --------------------------------------------------------------------------
# Collect
# --------------------------------------------------------------------------

def _prune_large_files(out_dir: Path, max_mb: int) -> list[str]:
    """Drop oversized artefacts so the workspace cannot fill the disk."""
    pruned: list[str] = []
    limit = max_mb * 1024 * 1024
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size <= limit:
                continue
            size_mb = path.stat().st_size / (1024 * 1024)
            path.unlink()
            pruned.append(f"{path.relative_to(out_dir)} ({size_mb:.0f} MB)")
        except OSError:
            continue
    return pruned


def read_metrics(run_id: str) -> dict | None:
    """Read the kernel's ``metrics.json`` contract, if it honoured it."""
    return ledger.read_json(ledger.run_output_dir(run_id) / METRICS_FILENAME)


def collect(slug: str, run_id: str, *, api=None, status: str = "complete",
            max_mb: int = DEFAULT_MAX_OUTPUT_MB, row: dict | None = None) -> dict:
    """Download a terminal run's outputs once, then release its kernel lock."""
    row = row if row is not None else find_run(slug, run_id)
    if row is None:
        raise LaunchError(f"unknown run {run_id}")

    if api is None:
        from submit_kernel import get_api

        api = get_api()

    from run_kernel import download_outputs

    out_dir = ledger.run_output_dir(run_id)
    files = download_outputs(api, row["kernel"], str(out_dir))
    pruned = _prune_large_files(out_dir, max_mb)
    if pruned:
        files = [f for f in files if not any(p.startswith(f) for p in pruned)]

    kernel_name = str(row["kernel"]).split("/")[-1]
    log_path = out_dir / f"{kernel_name}.log"
    elapsed = elapsed_seconds(row)
    gpu_hours_upper = round((elapsed or 0.0) / SECONDS_PER_HOUR, 3) if row.get("gpu") else 0.0

    ledger.append_jsonl(
        ledger.runs_ledger_path(slug),
        {
            "event": RUN_FINISHED,
            "run_id": run_id,
            "status": status,
            "elapsed_s_upper": None if elapsed is None else round(elapsed),
            "gpu_hours_upper": gpu_hours_upper,
            "output_dir": str(out_dir),
            "files": files,
            "log_path": str(log_path) if log_path.exists() else None,
            "pruned_files": pruned,
        },
    )
    if row.get("gpu"):
        lower = (row.get("elapsed_s") or 0) / SECONDS_PER_HOUR
        budget.record_gpu_actual(slug, run_id=run_id, hours_upper=gpu_hours_upper,
                                 hours_lower=round(lower, 3))

    ledger.release_lock(ledger.kernel_lock_name(row["kernel"]))
    return {
        "run_id": run_id, "status": status, "output_dir": str(out_dir),
        "files": files, "pruned_files": pruned,
        "log_path": str(log_path) if log_path.exists() else None,
        "metrics": read_metrics(run_id),
    }


# --------------------------------------------------------------------------
# Terminal analysis events
# --------------------------------------------------------------------------

def score(slug: str, run_id: str, *, cv_score: float | None, cv_metric: str | None = None,
          cv_direction: str | None = None, cv_folds: list | None = None,
          cv_std: float | None = None, verdict: str = "keep", rationale: str = "",
          parsed_from: str = METRICS_FILENAME) -> None:
    """Close a run with a judged result. Written by the HARVEST phase."""
    ledger.append_jsonl(
        ledger.runs_ledger_path(slug),
        {"event": RUN_SCORED, "run_id": run_id, "cv_score": cv_score,
         "cv_metric": cv_metric, "cv_direction": cv_direction, "cv_folds": cv_folds,
         "cv_std": cv_std, "verdict": verdict, "rationale": rationale,
         "parsed_from": parsed_from},
    )


def abandon(slug: str, run_id: str, *, reason: str, status: str = "error",
            row: dict | None = None) -> None:
    """Close a run that produced nothing usable, and free its kernel lock."""
    row = row if row is not None else find_run(slug, run_id)
    ledger.append_jsonl(
        ledger.runs_ledger_path(slug),
        {"event": RUN_ABANDONED, "run_id": run_id, "status": status, "reason": reason},
    )
    if row and row.get("kernel"):
        ledger.release_lock(ledger.kernel_lock_name(row["kernel"]))


def consecutive_kernel_errors(rows: list[dict]) -> int:
    """How many runs in a row ended badly — the no-blind-retry signal."""
    count = 0
    for row in reversed(rows):
        if row["active"] and not row["finished"]:
            continue
        if row.get("abandoned") or row.get("status") in {"error", "cancel_acknowledged"}:
            count += 1
        else:
            break
    return count
