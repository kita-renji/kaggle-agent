#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Sense the loop's state for one competition and name the next phase.

This is the single "what now?" call at the top of every tick. It reconciles
whatever a crashed prior tick left behind, rewrites STATE.md, and prints the
one directive to act on.

Usage:
    python agent_state.py <competition> [--as-json] [--write-state] [--offline]
"""

import argparse
import json
import sys
from pathlib import Path

# runtime, submit_kernel, and submission_quota live in the sibling base skill.
sys.path.append(str(Path(__file__).resolve().parents[2] / "nvidia-kaggle-skill" / "scripts"))

from runtime import load_project_env  # noqa: E402

load_project_env()

from agent import ledger, state  # noqa: E402 — .env must load before Kaggle imports


def main() -> None:
    parser = argparse.ArgumentParser(description="Report loop state and the next phase for a competition.")
    parser.add_argument("competition", nargs="?",
                        help="Competition slug (default: the current directory's name)")
    parser.add_argument("--as-json", action="store_true", help="Print the full tick state as JSON")
    parser.add_argument("--write-state", action="store_true", help="Rewrite competitions/<slug>/STATE.md")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip every Kaggle call: no reconcile, no quota check. For tests and idle fast paths.",
    )
    args = parser.parse_args()

    try:
        slug = ledger.resolve_slug(args.competition)
    except ledger.SlugError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    if not ledger.competition_dir(slug).exists() and not args.offline:
        print(
            f"No workspace at {ledger.competition_dir(slug)}.\n"
            f"Create one with: python agent_init.py {slug}",
            file=sys.stderr,
        )

    try:
        tick = state.tick_state(slug, offline=args.offline)
    except Exception as exc:  # noqa: BLE001 — a sense failure must be legible, not a traceback
        print(f"Error sensing {slug}: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.write_state:
        state.write_state_md(slug, tick)
        state.write_tick_state(slug, tick)

    if args.as_json:
        # Runs carry full config dicts; keep the payload to what a tick decides on.
        payload = {k: v for k, v in tick.items() if k not in {"runs", "submitted"}}
        payload["recent_runs"] = tick["runs"][-5:]
        payload["submitted_run_ids"] = sorted(tick["submitted"])
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(state.render_state_md(tick))


if __name__ == "__main__":
    main()
