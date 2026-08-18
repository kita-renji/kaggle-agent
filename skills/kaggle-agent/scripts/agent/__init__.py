# SPDX-License-Identifier: MIT
"""Autonomous Kaggle loop: deterministic mechanics for the /kaggle-tick loop.

The judgement lives in markdown (``agent-loop.md``, ``agent-hypotheses.md``,
``agent-analysis.md``); this package owns only what must not be re-derived by a
language model each tick — file layout, locking, ledger folds, budget
arithmetic, and phase selection.
"""

import sys
from pathlib import Path

# This package imports runtime, submit_kernel, and submission_quota from the
# sibling nvidia-kaggle-skill; make its scripts importable no matter which
# entry point loaded the package.
_BASE_SKILL_SCRIPTS = Path(__file__).resolve().parents[3] / "nvidia-kaggle-skill" / "scripts"
if str(_BASE_SKILL_SCRIPTS) not in sys.path:
    sys.path.append(str(_BASE_SKILL_SCRIPTS))
