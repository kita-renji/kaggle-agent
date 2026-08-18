# Forming hypotheses

Used by the BOOTSTRAP and REFILL phases of `agent-loop.md`. The backlog is the loop's supply of
work; when it is thin or lazy, every downstream tick is wasted on a run that could not have taught
anything.

## What a hypothesis must have

An entry that cannot be falsified is a wish, not a hypothesis. Every one needs five things:

```markdown
## H-005 — Concat ECFP4 fingerprints into the GINE readout  [priority: high] [cost: gpu-2h] [status: ready]
**Why:** [Public GNN+FP kernel](https://www.kaggle.com/code/x/y) reports −0.004 wMAE from ECFP4
concat; our H-004 is graph-features-only.
**Change:** 2048-bit ECFP4 (r=2) concatenated into the readout MLP. Everything else fixed vs H-004.
**Success:** CV wMAE < 0.0715 (H-004 = 0.0721).
**Falsify if:** CV ≥ 0.0721 after 5 folds, or fold std > 0.002.
```

- **Why** — grounded in a real source you actually fetched, or in a specific result in `runs.jsonl`.
  A link you did not open is not evidence. If the reason is "seems worth trying", say that plainly
  and mark it low priority rather than dressing it up.
- **Change** — one variable, against a named parent run. Two changes at once means a result you
  cannot attribute.
- **Success** — a number, compared against the current champion's number.
- **Falsify if** — the condition under which you will mark it `rejected` and move on. Without this
  you will keep nursing a dead idea.
- **Tags** — `priority` (high/med/low), `cost` (`cpu-15m`, `gpu-2h` — this is what the GPU guard
  filters on, so an honest estimate matters), `status`.

## Ordering

Cheap and discriminating beats expensive and vague. A 15-minute CPU run that rules out a whole
family of approaches is worth more than a 6-hour GPU run that yields one more decimal place.

Roughly, in order:

1. **Validation first.** If CV and LB disagree, nothing else you measure is trustworthy. Fix the
   split before optimising anything on top of it.
2. **Baseline.** A working end-to-end run that produces `submission.csv` and a CV number. Until one
   exists, there is no champion to measure against.
3. **Features and data.** Usually the largest gains, usually cheap.
4. **Model and architecture.** Larger gains than tuning, larger cost.
5. **Tuning.** Real but small. Do not start here.
6. **Ensembling.** Last, once there are diverse strong models to blend.

## Where to look when the backlog runs dry

In roughly descending value per token spent:

- `research/brief.md` and `research/sources.jsonl` — what you already gathered but have not tried.
- Your own `runs.jsonl` — a rejected hypothesis often suggests a better-posed neighbour. Why did it
  fail? Was the idea wrong, or the implementation?
- The discussion cache — `discussion_query.py <slug> --min-votes 20`.
- Top public kernels — `fetch_top_kernel_scores.py <slug> --sort descending`, then `kernel_read.py`
  on the two or three that actually score well, not the ones with the most votes.
- Solution writeups from *similar past competitions* — `fetch_leaderboard_writeups.py`. Often the
  richest source once the obvious local ideas are exhausted.

Fetch one specific thing and ground three hypotheses in it. Do not bulk-ingest.

## Honesty rules

These exist because the failure mode is quiet and expensive: a backlog full of plausible-sounding
entries that produce runs teaching nothing.

- **Votes are not scores.** A popular notebook is popular. If you are ranking by votes because
  scores were unavailable, say so in the `**Why:**` line.
- **A score in a title is a claim, not a measurement.** Treat it as unverified unless you fetched
  the kernel's actual score.
- **Never invent a number.** Not a baseline, not an expected gain, not a competitor's score. If you
  do not know it, write that you do not know it.
- **Retire honestly.** When a hypothesis is falsified, move it under `## Retired` with what it
  actually scored and one line on what that ruled out. That line is what stops you or a future
  context re-proposing it in three ticks' time.
- **Do not pad the queue.** Three grounded hypotheses beat eight filler ones. If you genuinely
  cannot find three, say so in the journal — running out of ideas is a legitimate stop condition,
  and pretending otherwise wastes GPU hours.

## Retiring

```markdown
## Retired
- H-004 — GINE 5-fold baseline → CV 0.0721, LB 0.0740 (r-20260811T0731Z-a3f2) — **kept**, champion
- H-003 — LightGBM on RDKit descriptors → CV 0.0812 (r-…11ab) — **superseded** by H-004
- H-002 — 10× LR warmup → CV 0.0819 vs 0.0812 control — **rejected**; LR schedule is not the
  bottleneck, stop tuning it
```
