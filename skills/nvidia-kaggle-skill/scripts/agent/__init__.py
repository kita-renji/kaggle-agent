# SPDX-License-Identifier: MIT
"""Autonomous Kaggle loop: deterministic mechanics for the /kaggle-tick loop.

The judgement lives in markdown (``agent-loop.md``, ``agent-hypotheses.md``,
``agent-analysis.md``); this package owns only what must not be re-derived by a
language model each tick — file layout, locking, ledger folds, budget
arithmetic, and phase selection.
"""
