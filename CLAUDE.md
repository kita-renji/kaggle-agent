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

## Locally allowed

- Editing code, writing kernel folders, metadata.
- Research scripts (competition info, writeups, discussions, kernel search) — API calls only, light on disk.
- Quick CPU sanity checks: syntax, imports, tiny-sample dry runs.
