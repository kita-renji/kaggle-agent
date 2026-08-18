#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Submit a scored run to its competition, under the agent's budget.

Two things make this different from ``submit_kernel.py``:

* **It never polls evaluation.** The call returns as soon as Kaggle accepts the
  submission; the public score is folded in by a later tick's reconcile. A tick
  that waited for evaluation would block the whole loop.
* **It spends from a budget.** At most ``agent_submission_cap`` submissions per
  UTC day, always leaving ``human_submission_reserve`` slots for the operator.
  The reservation is written before the API call, so a crash costs the agent a
  slot rather than risking a double-spend.

Usage:
    python agent_submit.py <competition> --run-id <id> [--file submission.csv] [--dry-run]
"""

import argparse
import json
import os
import sys
from pathlib import Path

# runtime, submit_kernel, and submission_quota live in the sibling base skill.
sys.path.append(str(Path(__file__).resolve().parents[2] / "nvidia-kaggle-skill" / "scripts"))

from runtime import load_project_env  # noqa: E402

load_project_env()

from agent import budget, ledger, runs, state  # noqa: E402 — .env first


DEFAULT_SUBMISSION_FILE = "submission.csv"


def _short_slug(kernel: str) -> str:
    return str(kernel).split("/")[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Budget-gated, non-blocking competition submission.")
    parser.add_argument("competition", nargs="?",
                        help="Competition slug (default: the current directory's name)")
    parser.add_argument("--run-id", required=True, help="Run to submit, from runs.jsonl")
    parser.add_argument("--file", default=DEFAULT_SUBMISSION_FILE,
                        help="Filename the kernel wrote into /kaggle/working (not a local path)")
    parser.add_argument("--message", help="Override the submission message (must start with the run id)")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate the budget and stop")
    args = parser.parse_args()

    try:
        slug = ledger.resolve_slug(args.competition)
    except ledger.SlugError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    dry_run = args.dry_run or os.environ.get("KAGGLE_AGENT_DRY_RUN") == "1"

    row = runs.find_run(slug, args.run_id)
    if row is None:
        print(f"Unknown run {args.run_id} in {slug}", file=sys.stderr)
        sys.exit(1)
    if not row.get("scored"):
        print(f"{args.run_id} has no recorded verdict yet — HARVEST it first.", file=sys.stderr)
        sys.exit(1)

    already = state.submitted_run_ids(slug)
    if args.run_id in already:
        print(f"{args.run_id} was already submitted ({already[args.run_id]}). Not spending another slot.",
              file=sys.stderr)
        sys.exit(1)

    mission = budget.read_mission(slug)
    verdict = budget.submission_budget(slug, mission=mission)
    message = args.message or f"{args.run_id} {row.get('hypothesis_id') or 'H-?'} {_short_slug(row['kernel'])}"

    if not verdict["allowed"]:
        print(json.dumps({"submitted": False, "reason": verdict["reason"], "budget": verdict},
                         indent=2, default=str))
        sys.exit(1)

    if dry_run:
        print(json.dumps({
            "submitted": False, "dry_run": True, "would_submit": {
                "run_id": args.run_id, "kernel": row["kernel"], "version": row.get("version"),
                "competition": slug, "file": args.file, "message": message,
            },
            "budget": verdict,
        }, indent=2, default=str))
        return

    # Reserve first: a crash between here and the API call costs one agent slot,
    # which is the safe direction to fail.
    budget.reserve_submission(slug, run_id=args.run_id, message=message,
                              hypothesis_id=row.get("hypothesis_id"), quota_snapshot=verdict)

    from submit_kernel import get_api, submit_to_competition

    api = get_api()
    accepted = submit_to_competition(api, row["kernel"], slug, args.file,
                                     row.get("version"), message)

    if not accepted:
        budget.release_submission(slug, run_id=args.run_id,
                                  reason=f"Kaggle rejected the submission of {args.file}")
        print(json.dumps({"submitted": False, "reason": "Kaggle rejected the submission"},
                         indent=2), file=sys.stderr)
        sys.exit(1)

    budget.record_submission_result(slug, run_id=args.run_id, accepted=True, eval_status="pending")
    print(json.dumps({
        "submitted": True, "run_id": args.run_id, "competition": slug,
        "kernel": row["kernel"], "version": row.get("version"), "message": message,
        "note": "Evaluation is not polled here — the score lands on a later tick's reconcile.",
        "budget_after": {"agent_used": verdict["agent_used"] + 1, "agent_cap": verdict["agent_cap"]},
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
