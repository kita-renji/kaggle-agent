# kaggle-agent (fork)

Fork of the NVIDIA Kaggle Plugin, customized for an environment **without local GPUs or a cluster**.

## Hard constraint: remote-only compute

- Never train models, run heavy inference, or process large datasets on the local machine.
- Never download full competition datasets locally. Local disk is for code, small samples, logs, and fetched kernel outputs only.
- All heavy compute runs on Kaggle: push code as a kernel and execute it there.

## Primary workflow

The skill lives in `skills/nvidia-kaggle-skill/`. For running code, use the **Remote Run** workflow (`skills/nvidia-kaggle-skill/remote-run.md`):

1. Develop code locally in a kernel folder (`kernel-metadata.json` + code file).
2. `python scripts/run_kernel.py <kernel-folder>` — pushes, runs on Kaggle, downloads outputs and run log.
3. Read the log, fix locally, push again.
4. Submit with `scripts/submit_kernel.py` (see `submission.md`) only after a clean run, and check quota first (`submission_quota.py`).

Ship multi-file code to kernels as a Kaggle dataset (`scripts/upload_dataset.py`) attached via `dataset_sources`, or chain kernels via `kernel_sources`.

## Autonomous loop

One `/loop` session per competition, self-paced. One command drives it:

```
/kaggle-agent start [slug]     scaffold if needed, then loop until a stop condition
/kaggle-agent stop  [slug]     kill switch (writes competitions/<slug>/HALT)
/kaggle-agent status [slug]    print STATE.md, no Kaggle calls
```

The slug is optional: run from inside `competitions/<slug>/` and it comes from the directory name.
`start` infers the metric, optimisation direction, and deadline from Kaggle — no hand-editing
needed before the first tick, though BOOTSTRAP verifies them against the Evaluation page.

- Contract: `skills/nvidia-kaggle-skill/agent-loop.md`. Each tick senses with `agent_state.py`,
  runs the one phase it names, writes state, and sleeps. Nothing blocks on a Kaggle run.
- State: `competitions/<slug>/` (mission, backlog, journal, `runs.jsonl`, `budget.jsonl`, research,
  kernel code) plus shared `data/agent/`. Kernel outputs and `HALT` are gitignored; the rest is
  tracked so decisions stay diffable.
- Budgets are enforced in Python, not by judgement: at most 3 submissions/day per competition with
  2 reserved for you, and a per-competition weekly GPU slice under a 26h account-wide guard.
  Concurrent loops share one Kaggle account, so keep the sum of `gpu_weekly_hours` under 26.
- The GPU ledger cannot see interactive Kaggle usage — it is a lower bound. Correct it with
  `agent.budget.record_gpu_manual(hours)`.

## Locally allowed

- Editing code, writing kernel folders, metadata.
- Research scripts (competition info, writeups, discussions, kernel search) — API calls only, light on disk.
- Quick CPU sanity checks: syntax, imports, tiny-sample dry runs.
