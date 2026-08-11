# Reading a run

Used by the HARVEST phase of `agent-loop.md`. The job is to turn one finished Kaggle run into one
honest ledger row and one decision.

## Order of reading

1. **`<output_dir>/metrics.json`** — the contract. `{cv_score, metric, direction, folds, config,
   notes}`. If it is there, this is the number.
2. **The log tail** — only for what `metrics.json` cannot carry: warnings, fold-level instability,
   silent fallbacks, how long stages actually took.
3. **The full log** — only when something is wrong.

If `metrics.json` is missing, the kernel broke its contract. Recover the number from the log if you
can and set `parsed_from="log"` so the row records that it is less trustworthy; then fix the kernel
template usage in the next LAUNCH. Do not silently treat a scraped number as equivalent.

## Deciding the verdict

**`keep`** — the result is real and the approach is worth building on. It becomes eligible for
submission and can become champion.

**`reject`** — the result is worse than the champion, or it is not trustworthy. Still record the
CV: a rejected run is evidence, and its number is what stops the idea coming back.

**`abandon`** (via `runs.abandon`) — the run produced no usable number at all: crashed, OOM'd, timed
out, wrote no outputs. Record the actual failure line from the log, not a paraphrase. This is what
makes the next attempt different from the last one.

## What to check before trusting a number

A CV that is too good is the most expensive thing in a competition, because it sends every
subsequent decision in the wrong direction.

- **Fold spread.** A high mean with one wild fold is not a high mean. If `cv_std` is large relative
  to the gap you are claiming, you have measured noise. Say so.
- **Leakage.** Did any feature see the target, or the full dataset, before the split? Target
  encoding, scalers fitted on train+test, group members split across folds, time-ordered data
  shuffled — these are the usual causes of a sudden implausible jump.
- **Split matches the task.** Grouped data needs grouped folds; time series needs time-ordered
  splits. If the public kernels use a specific scheme and you do not, that is a difference worth
  naming.
- **Sudden large jumps.** A gain much bigger than anything in the writeups is more likely a bug than
  a breakthrough. Check before you celebrate it in the journal.
- **Did it actually run what you think?** Compare the config in `metrics.json` against the
  `--config-json` recorded in the ledger. A silent fallback (fewer folds, a smaller model, a
  swallowed exception) shows up here.

## CV against LB

Once a run has both, the relationship matters more than either number.

- **Both improve** — the signal is real. Keep going in that direction.
- **CV improves, LB does not** — you are overfitting the validation split, or the split does not
  match the test distribution. Stop tuning; fix validation. Note it in the journal explicitly,
  because it invalidates the comparisons behind every queued hypothesis.
- **LB improves, CV does not** — usually luck on a small public split. Do not chase it. Trust CV.
- **Large constant offset, same ordering** — normal. Distribution shift between train and test. The
  ordering is what matters.

Record the divergence in the journal the first tick you see it. It should change the backlog, not
just the commentary.

## Writing the row

```python
from agent import runs
runs.score(
    "<slug>", "<run-id>",
    cv_score=0.0721, cv_metric="wMAE", cv_direction="lower_better",
    cv_folds=[0.0705, 0.0733, 0.0718, 0.0729, 0.0720], cv_std=0.0011,
    verdict="keep",
    rationale="beats H-003 (0.0812) by 11%; fold spread tight; no leakage signal in the log",
)
```

`rationale` is read by a future context that has none of your reasoning. One sentence on what the
number means relative to the champion, plus anything that would change how the next run is built.

Then update `backlog.md`: set the hypothesis's `[status: ...]` to `done` or `rejected` and move it
under `## Retired` with its result.

## Journal block

```markdown
## 2026-08-11T10:52Z — tick 37 — HARVEST
- Collected r-…a3f2 (complete, ≤3h18m GPU). CV wMAE 0.0721 vs champion 0.0812 — best so far.
- Fold spread 0.0705–0.0733 (std 0.0011); no leakage signal in the log.
- Verdict keep. H-004 done, promoted to champion.
- Next: SUBMIT — agent has 0/3 spent today.
```

What it needs to carry: what ran, what it scored against what, whether you trust it and why, what
changed in the backlog, and what happens next. Not a transcript.
