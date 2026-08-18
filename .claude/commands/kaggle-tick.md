---
description: Run one tick of the autonomous Kaggle loop for a competition
argument-hint: <competition-slug>
---

Run exactly ONE tick of the autonomous Kaggle loop for competition: $ARGUMENTS

1. Sense:

   ```bash
   python skills/kaggle-agent/scripts/agent_state.py $ARGUMENTS --as-json --write-state
   ```

2. **Idle fast path.** If `fingerprint_unchanged` is true and `next_phase` is `WAIT`: read no other
   file, write nothing, say at most two sentences, and call `ScheduleWakeup` with `wake_seconds`,
   `noop: true`, and prompt `/kaggle-tick $ARGUMENTS`. Stop there. Do not re-derive anything.

3. Otherwise read `skills/kaggle-agent/agent-loop.md` and
   `competitions/$ARGUMENTS/STATE.md`, then execute exactly the one phase named in `next_phase`.
   Do not start a second phase, however tempting — the next tick will get to it.

4. Rewrite `STATE.md` (rerun `agent_state.py --write-state`), append one journal block to
   `competitions/$ARGUMENTS/journal.md`, and call `ScheduleWakeup` with `wake_seconds`,
   `noop: false`, and prompt `/kaggle-tick $ARGUMENTS`.

If `next_phase` is `STOP`: write the handoff journal block, then call `ScheduleWakeup` with
`stop: true` instead.

Never train locally and never download the competition dataset — all compute runs on Kaggle.
