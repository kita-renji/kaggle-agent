# SPDX-License-Identifier: MIT
# Kernel template for the autonomous loop.
#
# Two contracts this file exists to enforce. Break either and the loop degrades
# from "reads a number" to "guesses from a log":
#
#   1. Write /kaggle/working/metrics.json before exiting. The HARVEST phase
#      reads it directly; log scraping is the fallback, not the path.
#   2. Checkpoint into /kaggle/working/ as you go. Only that directory survives,
#      and a 9h session timeout should still leave resumable weights behind.
#
# Everything below the CONFIG block is an example — replace it with the real
# experiment. Keep the metrics.json write and the checkpoint discipline.

import json
import os
import time

import numpy as np
import pandas as pd

WORKING = "/kaggle/working"
INPUT = "/kaggle/input"

# --------------------------------------------------------------------------
# CONFIG — mirror this dict in the launch command's --config-json so the
# experiment ledger records exactly what ran.
# --------------------------------------------------------------------------
CONFIG = {
    "model": "baseline",
    "folds": 5,
    "seed": 42,
}

METRIC = "accuracy"
DIRECTION = "higher_better"


def checkpoint(name: str, payload) -> str:
    """Persist intermediate state where a timed-out run can still be resumed."""
    path = os.path.join(WORKING, name)
    if hasattr(payload, "to_csv"):
        payload.to_csv(path, index=False)
    else:
        with open(path, "w") as handle:
            json.dump(payload, handle)
    return path


def write_metrics(cv_score, folds, notes="", extra=None):
    """The CV contract. Called exactly once, at the end of a successful run."""
    metrics = {
        "cv_score": None if cv_score is None else float(cv_score),
        "metric": METRIC,
        "direction": DIRECTION,
        "folds": [float(f) for f in (folds or [])],
        "config": CONFIG,
        "notes": notes,
    }
    if extra:
        metrics.update(extra)
    with open(os.path.join(WORKING, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"CV {METRIC}: {cv_score}")
    return metrics


def main():
    started = time.time()
    rng = np.random.default_rng(CONFIG["seed"])

    # ---- Load ------------------------------------------------------------
    # train = pd.read_csv(f"{INPUT}/<competition-slug>/train.csv")
    # test = pd.read_csv(f"{INPUT}/<competition-slug>/test.csv")

    # ---- Cross-validate --------------------------------------------------
    # Score every fold, checkpointing after each so a timeout is not a total loss.
    fold_scores = []
    for fold in range(CONFIG["folds"]):
        score = float(rng.uniform(0.80, 0.86))  # replace with a real fold score
        fold_scores.append(score)
        checkpoint("fold_scores.json", fold_scores)
        print(f"fold {fold}: {score:.5f}", flush=True)

    cv_score = float(np.mean(fold_scores))

    # ---- Predict ---------------------------------------------------------
    # Write submission.csv only when this run is meant to be submittable.
    submission = pd.DataFrame({"id": [1], "target": [0]})
    checkpoint("submission.csv", submission)

    # ---- Report ----------------------------------------------------------
    write_metrics(
        cv_score,
        fold_scores,
        notes=f"fold std {np.std(fold_scores):.5f}",
        extra={"runtime_seconds": round(time.time() - started, 1)},
    )


if __name__ == "__main__":
    main()
