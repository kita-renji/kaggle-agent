#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Drive a remote Kaggle run without ever blocking the tick.

Unlike ``run_kernel.py``, which pushes and then polls for hours, each
subcommand here is a single-shot interaction: push and return, check once,
download once. The run's state lives in ``competitions/<slug>/runs.jsonl``, so
a later tick — in a completely fresh context — can pick it up.

Usage:
    python agent_run.py launch <kernel-folder> [--competition <slug>] [options]
    python agent_run.py poll <run-id> [--competition <slug>]
    python agent_run.py collect <run-id> [--competition <slug>] [--max-mb N]
    python agent_run.py abandon <run-id> --reason "..." [--competition <slug>]
    python agent_run.py list [--competition <slug>]

``--competition`` defaults to the current directory's name, or to <slug> when
run from anywhere inside ``competitions/<slug>/``.
"""

import argparse
import json
import sys
from pathlib import Path

# runtime, submit_kernel, and submission_quota live in the sibling base skill.
sys.path.append(str(Path(__file__).resolve().parents[2] / "nvidia-kaggle-skill" / "scripts"))

from runtime import load_project_env  # noqa: E402

load_project_env()

from agent import ledger, runs  # noqa: E402 — .env must load before Kaggle imports


def _emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2, default=str))


def cmd_launch(args) -> int:
    config = {}
    if args.config_json:
        try:
            config = json.loads(args.config_json)
        except json.JSONDecodeError as exc:
            print(f"--config-json is not valid JSON: {exc}", file=sys.stderr)
            return 2

    try:
        result = runs.launch(
            args.kernel_folder,
            slug=args.competition,
            hypothesis_id=args.hypothesis,
            config=config,
            lane=args.lane,
            estimated_hours=args.estimated_hours,
            expected_runtime_min=args.expected_runtime_min,
            force=args.force,
            dry_run=args.dry_run,
        )
    except runs.LaunchError as exc:
        print(f"Launch refused: {exc}", file=sys.stderr)
        return 1
    except ledger.LockTimeout as exc:
        print(f"Launch refused: {exc}", file=sys.stderr)
        return 1

    _emit(result)
    if result.get("dry_run"):
        print("\nDry run: nothing was pushed and no ledger row was written.", file=sys.stderr)
    return 0


def cmd_poll(args) -> int:
    try:
        _emit(runs.poll(args.competition, args.run_id))
    except runs.LaunchError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def cmd_collect(args) -> int:
    try:
        result = runs.collect(args.competition, args.run_id, status=args.status, max_mb=args.max_mb)
    except runs.LaunchError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    _emit(result)
    if result.get("metrics") is None:
        print(
            "\nNo metrics.json in the output. The kernel must write "
            "/kaggle/working/metrics.json — fall back to reading the log tail.",
            file=sys.stderr,
        )
    return 0


def cmd_abandon(args) -> int:
    runs.abandon(args.competition, args.run_id, reason=args.reason, status=args.status)
    _emit({"run_id": args.run_id, "abandoned": True, "reason": args.reason})
    return 0


def cmd_list(args) -> int:
    rows = runs.load_runs(args.competition)
    _emit({
        "competition": args.competition,
        "active": [r["run_id"] for r in rows if r["active"]],
        "awaiting_harvest": [r["run_id"] for r in runs.awaiting_harvest(args.competition, rows=rows)],
        "runs": [
            {k: r.get(k) for k in ("run_id", "hypothesis_id", "kernel", "version", "status",
                                   "gpu", "cv_score", "verdict", "config_hash", "started_at")}
            for r in rows[-args.limit:]
        ],
    })
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-blocking Kaggle run lifecycle.")
    sub = parser.add_subparsers(dest="command", required=True)

    launch = sub.add_parser("launch", help="Push a kernel and record the run; returns immediately")
    launch.add_argument("kernel_folder", help="Folder containing kernel-metadata.json")
    launch.add_argument("--competition", help="Competition slug (default: the current directory's name)")
    launch.add_argument("--hypothesis", help="Backlog hypothesis id, e.g. H-005")
    launch.add_argument("--config-json", help="Experiment config as a JSON object")
    launch.add_argument("--lane", default=runs.DEFAULT_LANE, help="Lane name (multi-agent seam)")
    launch.add_argument("--estimated-hours", type=float, help="Estimated GPU hours, for the budget guard")
    launch.add_argument("--expected-runtime-min", type=int, help="Expected wall-clock minutes, for wakeup pacing")
    launch.add_argument("--force", action="store_true", help="Override the identical-experiment guard")
    launch.add_argument("--dry-run", action="store_true", help="Resolve and check everything, write nothing")
    launch.set_defaults(func=cmd_launch)

    poll = sub.add_parser("poll", help="One kernels_status call")
    poll.add_argument("run_id")
    poll.add_argument("--competition", help="Competition slug (default: the current directory's name)")
    poll.set_defaults(func=cmd_poll)

    collect = sub.add_parser("collect", help="Download a terminal run's outputs once")
    collect.add_argument("run_id")
    collect.add_argument("--competition", help="Competition slug (default: the current directory's name)")
    collect.add_argument("--status", default="complete", help="Terminal Kaggle status to record")
    collect.add_argument("--max-mb", type=int, default=runs.DEFAULT_MAX_OUTPUT_MB,
                         help="Prune downloaded files larger than this")
    collect.set_defaults(func=cmd_collect)

    abandon = sub.add_parser("abandon", help="Close a run that produced nothing usable")
    abandon.add_argument("run_id")
    abandon.add_argument("--reason", required=True)
    abandon.add_argument("--status", default="error")
    abandon.add_argument("--competition", help="Competition slug (default: the current directory's name)")
    abandon.set_defaults(func=cmd_abandon)

    listing = sub.add_parser("list", help="Show the run ledger")
    listing.add_argument("--competition", help="Competition slug (default: the current directory's name)")
    listing.add_argument("--limit", type=int, default=10)
    listing.set_defaults(func=cmd_list)

    args = parser.parse_args()
    try:
        args.competition = ledger.resolve_slug(getattr(args, "competition", None))
    except ledger.SlugError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
