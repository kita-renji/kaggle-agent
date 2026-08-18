# Autonomous Loop — the tick contract

One `/loop` session per competition. Each tick does **one** thing and then sleeps. All durable
state is on disk, so a tick that starts with a completely fresh context loses nothing.

```
/kaggle-agent start [slug]     scaffold if needed, then run the loop
/kaggle-agent stop  [slug]     create the HALT file
/kaggle-agent status [slug]    print STATE.md, no Kaggle calls
```

`/kaggle-agent start` is the only command you need — it scaffolds the workspace (inferring the
metric, direction, and deadline from Kaggle) and hands off to `/loop /kaggle-tick <slug>`. The
slug is optional everywhere: run from inside `competitions/<slug>/` and it is taken from the
directory. Stop by creating `competitions/<slug>/HALT`, or let a stop condition end it.

## The tick

```
1. SENSE     python skills/kaggle-agent/scripts/agent_state.py <slug> --as-json --write-state
2. Read the ONE phase named by next_phase. Do that phase. Nothing else.
3. Rewrite STATE.md, append one journal block.
4. ScheduleWakeup with wake_seconds, prompt "/kaggle-tick <slug>".
```

`agent_state.py` reconciles before it decides: it polls each active run once, downloads any that
finished, abandons anything overdue, and folds in evaluation scores for submissions it stopped
polling. Whatever a crashed prior tick left behind is repaired before the ladder runs.

**The idle fast path.** If `fingerprint_unchanged` is true and `next_phase` is `WAIT`, read no
other file, write no journal entry, emit at most two sentences, and schedule the wakeup. Do not
re-derive anything, do not re-read the backlog, do not "just check" the run. This is what keeps an
overnight loop from burning tokens on nothing.

## The phase ladder

First match wins. Exactly one phase runs per tick.

| Phase | Fires when | Do this |
|---|---|---|
| **STOP** | any stop condition | Write a handoff journal block, then `ScheduleWakeup{stop: true}` |
| **BOOTSTRAP** | MISSION.md, research/brief.md, or backlog.md missing | Research the competition, seed the mission and backlog |
| **HARVEST** | a run finished but has no verdict | Read its result, record `run_scored` or `run_abandoned` |
| **SUBMIT** | a kept run beats the best submitted CV and budget allows | `agent_submit.py`, then return — never poll evaluation |
| **LAUNCH** | a lane is free and a ready hypothesis fits the budget | Write the kernel, push it with `agent_run.py launch` |
| **REFILL** | fewer than 3 ready hypotheses | Generate more, grounded in evidence |
| **WAIT** | a run is in flight, or nothing is actionable | Nothing. Sleep. |

### BOOTSTRAP

Follow the base skill's `skills/nvidia-kaggle-skill/research-brief.md` for how to gather and cite.
Produce, in this order, checkpointing each file as you write it so a mid-phase interruption is not
a total loss:

1. `research/overview.md` and `research/dataset.md` — from `fetch_competition_info.py` and
   `fetch_dataset_info.py` (in `skills/nvidia-kaggle-skill/scripts/`).
2. `research/brief.md` — the strategy brief. Metric and its direction, submission format, data
   shape, what has actually scored well and where you read that, and a baseline → strong → top
   score ladder with each rung linked.
3. `research/sources.jsonl` — one row per source consulted: `{kind, ref, url, note}`. This is what
   later REFILL phases mine, so it earns its keep even when the brief is already written.
4. `MISSION.md` — `agent_init.py` already inferred `metric`, `direction`, and `deadline` from
   Kaggle. Verify them against the Evaluation page you just read and correct any that are wrong or
   blank — **`direction` above all**, since it decides which run wins. Then set `target_lb` from
   the score ladder in the brief, and this competition's `gpu_weekly_hours` (keep the sum across
   active competitions under 26h).
5. `backlog.md` — 5–8 seed hypotheses per `agent-hypotheses.md`.

BOOTSTRAP may span two or three ticks. Wakeup stays at 60s while it is incomplete.

### HARVEST

Read `agent-analysis.md`. In short: read `<output_dir>/metrics.json` first, the log tail second.
Then append exactly one terminal event:

```bash
python skills/kaggle-agent/scripts/agent_run.py collect <slug> <run-id>   # if not already collected
```

Record the verdict from Python, so the numbers in the ledger are the numbers you read:

```python
from agent import runs
runs.score("<slug>", "<run-id>", cv_score=0.8412, cv_metric="accuracy",
           cv_direction="higher_better", cv_folds=[...], cv_std=0.0061,
           verdict="keep", rationale="beats H-003 (0.8371); fold spread tight")
```

`verdict` is `keep` or `reject`. A rejected run is never submitted and never becomes champion, but
it still counts as evidence — say what it ruled out. If the run errored, use
`runs.abandon(slug, run_id, reason="...")` with the actual failure from the log, not a guess.

Then update the hypothesis's `[status: ...]` in `backlog.md` to `done` or `rejected`, and move it
under `## Retired` with its result.

### SUBMIT

```bash
python skills/kaggle-agent/scripts/agent_submit.py <slug> --run-id <run-id> --file submission.csv
```

It re-checks the budget against the live quota, reserves a slot, submits, and returns. **It does
not wait for the score** — that arrives on a later tick's reconcile. Do not follow it with a poll.

Add `--dry-run` to see the decision without spending anything.

### LAUNCH

Read `remote-run.md` for kernel authoring, compute limits, checkpointing, and kernel chaining.
Build the kernel folder under `competitions/<slug>/kernels/<kernel-slug>/`, starting from
`templates/kernel/`. Two rules the template already encodes and you must keep:

- Write `/kaggle/working/metrics.json` — `{cv_score, metric, direction, folds, config, notes}`.
  This is how HARVEST reads a number instead of guessing one.
- Checkpoint into `/kaggle/working/` as you go, so a 9h timeout still leaves resumable weights.

Sanity-check locally — `python -m py_compile`, an import check, a tiny-sample dry run. **Never
train locally and never download the competition data.** Then:

```bash
python skills/kaggle-agent/scripts/agent_run.py launch competitions/<slug>/kernels/<name> \
    --competition <slug> --hypothesis H-005 --config-json '{"model":"lgbm","folds":5,"lr":0.03}' \
    --expected-runtime-min 40 [--estimated-hours 2.5]
```

`--config-json` must mirror the kernel's `CONFIG` dict — the ledger is only as honest as this
argument. `--expected-runtime-min` drives wakeup pacing; `--estimated-hours` drives the GPU guard.

A refusal is information, not an obstacle. `identical experiment already ran` means you rebuilt
something the ledger remembers; change the approach rather than passing `--force`.

### REFILL

Read `agent-hypotheses.md`. Read the last ~10 journal blocks, the scored runs, and
`research/brief.md`. Write 3–5 new hypotheses. If they would just be variations of what has
already failed, fetch one specific writeup or discussion thread and ground them in that instead.

### WAIT

If the fingerprint changed, one line in the journal is enough. If it did not, write nothing — see
the idle fast path above.

## Wakeup pacing

`agent_state.py` computes `wake_seconds`; use it. It ranges from 60s when work is queued, through
600s while an evaluation is pending, to 3600s during a long run, with exponential backoff on
consecutive idle ticks. `ScheduleWakeup` clamps to [60, 3600], and every value stays inside that.

Pass `noop: true` when the tick changed nothing, `noop: false` when it wrote something.

## Safety rails

These are enforced in Python, not left to judgement. Do not work around them; if one blocks you,
say so in the journal and take a different action.

- **Submissions.** At most 3/day per competition, always leaving 2 for the human. If the daily
  limit is 2 or fewer, the agent submits nothing at all and escalates. A reservation is written
  before the API call, so a crash costs a slot rather than risking a double-spend.
- **GPU.** Both this competition's `gpu_weekly_hours` slice and the 26h account-wide guard must
  allow the run. The estimate is reserved at push time, so concurrent competition loops see the
  commitment immediately. The ledger cannot see interactive Kaggle usage, so it is a lower bound —
  `record_gpu_manual()` is how a human corrects it.
- **One run at a time.** `max_lanes` defaults to 1. A kernel lock, held for the run's whole life,
  makes a duplicate push fail fast even from another session.
- **No blind retries.** After `max_consecutive_kernel_errors` failures the loop stops. Read the
  log and fix the cause.

## Stop conditions

`HALT` file · past `deadline − stop_before_hours` · `target_lb` reached · idle-tick overflow ·
consecutive kernel errors · no CV improvement with an empty backlog after a REFILL · GPU exhausted
with only GPU work left · missing or invalid `KAGGLE_API_TOKEN`.

On STOP, the journal block is a handoff: what was learned, what is queued, what the human should
decide. Then `ScheduleWakeup{stop: true}`.

## Files

| Path | Written by | Purpose |
|---|---|---|
| `MISSION.md` | human | Goal contract. The agent reads it and never writes it. |
| `STATE.md` | `agent_state.py` | Regenerated every tick. Read first. Never hand-edit. |
| `backlog.md` | agent | Hypothesis queue. |
| `journal.md` | agent | One block per tick — where intent survives compaction. |
| `runs.jsonl` | `agent_run.py` | Experiment ledger, append-only. |
| `budget.jsonl` | `agent_submit.py` | Submission reserve/commit log. |
| `research/` | BOOTSTRAP | Brief, overview, dataset, sources. |
| `kernels/<name>/` | LAUNCH | Kernel code and metadata. |
| `data/agent/gpu_usage.jsonl` | shared | Account-wide GPU accounting. |
| `data/agent/runs/<run-id>/` | `collect` | Downloaded outputs and the run log. |
