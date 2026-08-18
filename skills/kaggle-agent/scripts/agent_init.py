#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Scaffold a competition workspace for the autonomous loop.

Creates ``competitions/<slug>/`` with a MISSION.md, empty ledgers, and seed
markdown. The metric, its optimisation direction, and the deadline are inferred
from Kaggle rather than typed by hand — direction in particular is the one
field that silently corrupts every later comparison if it is wrong.

Idempotent: never overwrites an existing file, so it is safe to re-run against
a live workspace and safe for the loop to call on every start.

Usage:
    python agent_init.py [competition]        # slug defaults to the directory name
    python agent_init.py titanic --metric accuracy --direction higher_better
    python agent_init.py titanic --no-fetch   # skip inference, leave blanks for BOOTSTRAP
"""

import argparse
import sys
from pathlib import Path

# runtime, submit_kernel, and submission_quota live in the sibling base skill.
sys.path.append(str(Path(__file__).resolve().parents[2] / "nvidia-kaggle-skill" / "scripts"))

from runtime import load_project_env  # noqa: E402

load_project_env()

from agent import budget, competition, ledger  # noqa: E402 — .env first


BACKLOG_SEED = """# Backlog - {slug}

<!-- One `## H-NNN - title [priority: high|med|low] [cost: cpu-15m|gpu-2h] [status: ready|running|done|rejected|blocked]`
     per hypothesis. Everything under `## Retired` is history, not queue. -->

## Retired
"""

JOURNAL_SEED = """# Journal - {slug}

One block per tick: what the phase did, what it showed, what was decided and why.
This is where a fresh context recovers intent that the JSONL ledgers cannot carry.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a competition workspace for the loop.")
    parser.add_argument("competition", nargs="?",
                        help="Competition slug or URL (default: the current directory's name)")
    parser.add_argument("--metric", help="Override the inferred evaluation metric")
    parser.add_argument("--direction", choices=["higher_better", "lower_better"],
                        help="Override the inferred optimisation direction")
    parser.add_argument("--deadline", help="Override the inferred UTC ISO deadline")
    parser.add_argument("--gpu-weekly-hours", type=float,
                        help="This competition's slice of the ~30h/week account GPU quota")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Do not call Kaggle; leave unknown fields for BOOTSTRAP to fill")
    args = parser.parse_args()

    try:
        slug = ledger.resolve_slug(args.competition)
    except ledger.SlugError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.no_fetch:
        resolved = {
            "metric": args.metric or "", "direction": args.direction,
            "deadline": args.deadline or "", "title": "",
            "confident": bool(args.metric and args.direction),
            "sources": {"metric": "argument" if args.metric else None,
                        "direction": "argument" if args.direction else None},
        }
    else:
        resolved = competition.infer(slug, metric=args.metric, direction=args.direction)
        if args.deadline:
            resolved["deadline"] = args.deadline

    workspace = ledger.competition_dir(slug)
    root = ledger.project_root()
    created: list[str] = []

    for directory in (workspace, ledger.research_dir(slug), ledger.kernels_dir(slug)):
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(str(directory.relative_to(root)))

    mission = ledger.mission_path(slug)
    mission_existed = mission.exists()
    budget.write_mission_template(
        slug,
        metric=resolved.get("metric") or "",
        direction=resolved.get("direction") or "higher_better",
        deadline=resolved.get("deadline") or "",
    )
    if not mission_existed:
        created.append(str(mission.relative_to(root)))
        if args.gpu_weekly_hours is not None:
            mission.write_text(
                mission.read_text(encoding="utf-8").replace(
                    f"gpu_weekly_hours: {budget.MISSION_DEFAULTS['gpu_weekly_hours']}",
                    f"gpu_weekly_hours: {args.gpu_weekly_hours}"),
                encoding="utf-8",
            )

    for path, content in (
        (ledger.backlog_path(slug), BACKLOG_SEED.format(slug=slug)),
        (ledger.journal_path(slug), JOURNAL_SEED.format(slug=slug)),
    ):
        if not path.exists():
            path.write_text(content, encoding="utf-8")
            created.append(str(path.relative_to(root)))

    print(f"Competition: {slug}" + (f"  ({resolved['title']})" if resolved.get("title") else ""))
    print(f"Workspace:   {workspace}")
    if created:
        print("Created:     " + ", ".join(created))
    else:
        print("Created:     nothing — the workspace already existed")

    if mission_existed:
        print("\nMISSION.md already exists and was left untouched.")
    else:
        sources = resolved.get("sources", {})
        print("\nMISSION.md:")
        print(f"  metric:    {resolved.get('metric') or '(unknown)'}"
              f"   [{sources.get('metric') or 'not determined'}]")
        print(f"  direction: {resolved.get('direction') or '(unknown)'}"
              f"   [{sources.get('direction') or 'not determined'}]")
        if resolved.get("deadline"):
            print(f"  deadline:  {resolved['deadline']}")

        if not resolved.get("confident"):
            print(
                "\n  Could not pin down every field. BOOTSTRAP will read the competition's "
                "Evaluation page on the first tick and fill in the rest."
            )
        elif sources.get("direction") not in {"argument", "evaluation page wording"}:
            print(
                f"\n  direction was inferred from the metric's name. Worth a glance: it decides "
                f"which run wins.\n  Fix with:  --direction lower_better"
            )

    if not (root / ".env").exists():
        print("\nWarning: no .env at the project root. Set KAGGLE_API_TOKEN before the first live run.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
