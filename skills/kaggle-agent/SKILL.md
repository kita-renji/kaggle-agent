---
name: kaggle-agent
description: "Use for running Kaggle code on remote compute (Remote Run) and for the autonomous competition loop: launch kernels, harvest results, manage budgets, and submit. Builds on nvidia-kaggle-skill for research and submission plumbing."
license: MIT
permissions:
  - "shell: run the bundled scripts/, the scripts under ../nvidia-kaggle-skill/scripts/, the Kaggle CLI (kaggle), and basic file utilities"
  - "network: HTTPS to Kaggle APIs (kaggle.com, api.kaggle.com) and PyPI"
  - "env: read environment variables, including KAGGLE_API_TOKEN, and load a project .env file"
  - "file_read: read the project .env, competition workspaces, and user-specified paths"
  - "file_write: write competition state under competitions/<slug>/, shared ledgers under data/agent/, and kernel outputs"
metadata:
  short-description: "Remote Kaggle execution and autonomous competition loop"
  author: "kaggle-agent fork"
  tags:
    - kaggle
    - competition
    - autonomous
    - remote-execution
---

# Kaggle Agent Skill

## Purpose

Use this skill when code must execute on Kaggle compute (training, GPU inference, anything too heavy for the local machine) and when a competition should be worked autonomously: research, run experiments, judge results, submit, and iterate across many ticks.

This skill layers on top of `nvidia-kaggle-skill` and calls into its scripts (`submit_kernel.py`, `submission_quota.py`, `runtime.py`). Use `nvidia-kaggle-skill` directly for research workflows — competition details, writeups, discussion and kernel research, dataset upload. Both skills must be installed together.

This environment has no local GPU or cluster. All training, heavy inference, and large data processing must run on Kaggle compute via the Remote Run workflow — never locally. Do not download full competition datasets locally; only quick CPU sanity checks (syntax, imports, tiny-sample dry runs) run on the local machine.

## Inputs

| Input | Required | Description |
|---|---|---|
| Competition slug or kernel folder | Depends on task | Primary target for the requested action. |
| `KAGGLE_API_TOKEN` | Required for every workflow | KGAT token string for Kaggle API, CLI, and SDK calls. |
| `../nvidia-kaggle-skill/scripts/` | Required | The base skill's scripts are imported at runtime. |

## Prerequisites

- Set `KAGGLE_API_TOKEN` before any workflow.
- Keep `skills/nvidia-kaggle-skill/` present at its sibling path — this skill's scripts import from it.

## Workflows

### Remote Run

Use this when code needs to execute on Kaggle compute: training, GPU inference, or any run too heavy for the local machine. Develop locally, push as a kernel, run remotely, and fetch the log and outputs back. Read `./remote-run.md`.

```bash
PYTHONUNBUFFERED=1 python ./scripts/run_kernel.py <kernel-folder> [--output DIR] [--log-tail N] [--no-download]
```

### Autonomous Loop

Use this when the user wants the agent to work a competition unattended. One `/loop` session per competition. Read `./agent-loop.md` for the tick contract, `./agent-hypotheses.md` for how to form hypotheses, and `./agent-analysis.md` for how to read a finished run.

```bash
python ./scripts/agent_init.py [slug]                        # scaffold; infers metric + direction
python ./scripts/agent_state.py [slug] --as-json --write-state   # sense + next phase
python ./scripts/agent_run.py launch <kernel-folder> [--competition <slug>] [--dry-run]
python ./scripts/agent_submit.py [slug] --run-id <id> [--dry-run]
```

The slug is optional in every script: it defaults to the current directory's name, or to `<slug>` when run from anywhere inside `competitions/<slug>/`.

The user drives the loop with `/kaggle-agent start [slug]`, which scaffolds and then hands off to `/loop /kaggle-tick <slug>`; `/kaggle-agent stop` writes the HALT file. Submission and GPU budgets are enforced in Python — at most 3 submissions/day per competition with 2 reserved for the human, and a per-competition weekly GPU slice under a 26h account-wide guard shared by every concurrent loop.

### Research and submission plumbing

Delegate to `nvidia-kaggle-skill`: competition details, writeups, discussion and kernel research, kernel setup, dataset upload, and manual submission (`submission.md`). When pulling kernels for reading in this environment, pass `skip-competition` and skip large dataset/model downloads — reproduction runs happen on Kaggle via Remote Run, not locally.

## Outputs

- Remote run downloads kernel output files and the run log into `<kernel-folder>/output/` (or `--output DIR`).
- The autonomous loop keeps per-competition state in `competitions/<slug>/` (`MISSION.md`, `STATE.md`, `backlog.md`, `journal.md`, `runs.jsonl`, `budget.jsonl`, `research/`, `kernels/`) and shared state in `data/agent/` (`gpu_usage.jsonl`, downloaded run outputs).

## Troubleshooting

| Symptom | Cause | Action |
|---|---|---|
| `ModuleNotFoundError: submit_kernel` (or `runtime`, `submission_quota`) | `skills/nvidia-kaggle-skill/scripts/` missing or moved. | Restore the base skill at its sibling path; this skill imports from it. |
| `KAGGLE_API_TOKEN` missing or invalid | Workflow started without valid Kaggle credentials. | Set `KAGGLE_API_TOKEN` and rerun the exact command. |
| Submission blocked by budget | Daily cap reached or human reserve would be spent. | Wait for the 00:00 UTC reset or let the human decide; never bypass the Python budget guard. |
| Run stuck in `launched` | Kaggle queue delay or the kernel died without a log. | `agent_run.py poll` stays non-blocking; after the timeout the loop abandons and re-queues the hypothesis. |
