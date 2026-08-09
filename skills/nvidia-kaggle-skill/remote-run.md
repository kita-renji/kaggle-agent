# Remote Run

Use this workflow to develop code locally and execute it on Kaggle compute. This is the primary execution path when no local GPU is available: all training, heavy inference, and data processing run inside Kaggle kernels, never locally.

## Inputs

| Input | Required | Description |
|---|---|---|
| Local kernel folder | Yes | Contains `kernel-metadata.json` and the code file it references. |
| `KAGGLE_API_TOKEN` | Yes | Required to push kernels and download outputs. |

## Core Loop

1. Edit code locally in the kernel folder.
2. Push and run on Kaggle, then fetch the log and outputs:

```bash
PYTHONUNBUFFERED=1 python ./scripts/run_kernel.py <kernel-folder>
python ./scripts/run_kernel.py <kernel-folder> --output ./results --log-tail 100
python ./scripts/run_kernel.py <kernel-folder> --no-download
```

3. Read the downloaded log (`<output-dir>/<kernel-slug>.log`), diagnose, edit, repeat.

The script pushes the kernel, polls until the run finishes, then downloads all output files plus the run log into `<kernel-folder>/output/` (or `--output DIR`) and prints the log tail. It never submits to a competition — use `submit_kernel.py` (see [submission.md](submission.md)) once a run produces a valid submission file.

Run long kernels in the background with unbuffered output. If the poll times out, the kernel keeps running on Kaggle; fetch results later with `kaggle kernels output <owner>/<slug> -p <dir>`.

## Creating A New Kernel Folder

A kernel folder needs `kernel-metadata.json` plus the code file. Template:

```json
{
  "id": "<kaggle-username>/<kernel-slug>",
  "title": "<Kernel Title>",
  "code_file": "train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": ["<owner>/<dataset-slug>"],
  "competition_sources": ["<competition-slug>"],
  "kernel_sources": [],
  "model_sources": []
}
```

Rules:

- `id` must start with the authenticated Kaggle username. The slug must be lowercase alphanumeric with dashes, and the first push creates the kernel.
- `kernel_type` is `script` for `.py` files or `notebook` for `.ipynb`. Prefer `script` for agent-driven iteration — plain Python diffs cleanly and avoids notebook JSON churn.
- `title` must be at least five characters and should match the slug to avoid Kaggle renaming the kernel URL.
- Attach every input the code reads: competitions under `competition_sources`, datasets under `dataset_sources`, other kernels' outputs under `kernel_sources`, Kaggle models under `model_sources`.
- Inside the kernel, inputs mount read-only under `/kaggle/input/<source-name>/`, and everything written to `/kaggle/working/` becomes the kernel's downloadable output (max 20 GB).
- Code competitions usually require `enable_internet: false` at submission time. Keep a training kernel with internet on and a separate offline inference/submission kernel if needed.

## Kaggle Compute Constraints

Design runs around these limits instead of discovering them mid-run:

| Resource | Limit |
|---|---|
| GPU (T4 x2 or P100) | ~30 h/week quota, 9 h max per session |
| TPU | 20 h/week, 9 h max per session |
| CPU-only session | 12 h max |
| RAM | ~13 GB (CPU) / ~16 GB (GPU) |
| `/kaggle/working` output | 20 GB |
| Scratch outside working dir | ~20 GB, discarded after run |

- Checkpoint training to `/kaggle/working` so a timed-out run still yields resumable weights.
- Split long jobs into stages: a training kernel saves weights, then either a follow-up kernel lists it in `kernel_sources` to consume `/kaggle/input/<training-kernel-slug>/`, or upload the downloaded weights as a dataset (see [SKILL.md](SKILL.md#upload-dataset)).
- Only quick CPU sanity checks (syntax, imports, tiny-sample dry runs) belong on the local machine.

## Shipping Local Code To Kernels

Kernels are a single code file. For multi-module projects, pick one:

- Package the project as a Kaggle dataset (`upload_dataset.py <project-folder>`), attach it via `dataset_sources`, then `sys.path.insert(0, "/kaggle/input/<dataset-slug>")` inside the kernel.
- Or inline/concatenate modules into the single `code_file` for small projects.

Re-upload the dataset (new version) whenever the library code changes; kernels pick up the latest dataset version on the next push.

## Workflow-Specific Troubleshooting

See [SKILL.md](SKILL.md#troubleshooting) for credential, rate-limit, and access failures.

| Symptom | Action |
|---|---|
| Push rejected: title/slug mismatch or invalid slug | Fix `title`/`id` in `kernel-metadata.json` to satisfy slug rules, then push again. |
| Push rejected: source not found | Verify each `dataset_sources`/`kernel_sources` ref exists and is spelled `owner/slug`; competition rules must be accepted for `competition_sources`. |
| Kernel status `error` | Read the downloaded `<kernel-slug>.log` for the traceback before editing; do not blind-retry. |
| GPU quota exhausted | Wait for the weekly reset, switch `enable_gpu` off for CPU-feasible work, or reduce run frequency by batching experiments per push. |
| Output download empty | The kernel may have written outside `/kaggle/working`; only files under it are preserved. |
