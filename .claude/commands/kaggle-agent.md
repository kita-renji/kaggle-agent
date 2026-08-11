---
description: Start, stop, or check the autonomous Kaggle loop for a competition
argument-hint: start|stop|status [competition-slug]
---

Autonomous Kaggle loop control. Arguments: $ARGUMENTS

Parse them as `<action> [competition-slug]`, where action is `start`, `stop`, or `status`.
If no action is given, assume `start`. If no slug is given, leave it out of the commands below —
every script defaults to the current directory's name, or to `<slug>` when run from inside
`competitions/<slug>/`.

## start

1. Scaffold the workspace. Idempotent — it never overwrites anything, and it infers the metric,
   optimisation direction, and deadline from Kaggle:

   ```bash
   python skills/nvidia-kaggle-skill/scripts/agent_init.py [slug]
   ```

   Show the user the metric and direction it resolved, and where each came from. If direction was
   inferred from the metric's *name* rather than stated outright on the Evaluation page, say so in
   one line — it decides which run wins, so a wrong guess is worth catching now. Do not block on
   it; the loop can start regardless and BOOTSTRAP will refine MISSION.md.

2. Hand off to the loop skill, which will repeat one tick at a time and pace itself:

   Invoke the `loop` skill with argument: `/kaggle-tick <slug>`

That is the whole start path. Do not run a tick inline first — the loop's own first iteration does
it, and doing it twice would double-count the tick.

## stop

```bash
touch competitions/<slug>/HALT
```

Then tell the user the loop stops at its next tick, and that deleting the file lets it resume.
If the loop is running in this session, it is also fine to end it immediately with
`ScheduleWakeup{stop: true}`.

## status

```bash
python skills/nvidia-kaggle-skill/scripts/agent_state.py [slug] --offline
```

Print the rendered STATE.md. `--offline` keeps it free: no Kaggle calls, no quota check, no
reconcile. Add `--write-state` only if the user wants STATE.md refreshed on disk.

## Notes

- Never train locally and never download the competition dataset — all compute runs on Kaggle.
- One loop per competition. Concurrent loops share one Kaggle account, so keep the sum of
  `gpu_weekly_hours` across their MISSION.md files under 26.
- If `KAGGLE_API_TOKEN` is unset and there is no `.env` at the project root, say so before starting:
  every live path will fail without it.
