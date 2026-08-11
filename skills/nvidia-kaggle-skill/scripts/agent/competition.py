# SPDX-License-Identifier: MIT
"""Work out a competition's metric, optimisation direction, and deadline.

``direction`` is the one MISSION.md field that changes behaviour rather than
display: it decides which run is champion, whether a candidate clears the
submit threshold, and whether ``target_lb`` counts as reached. Get it backwards
and the loop confidently promotes its worst run. So it is inferred here, from
Kaggle's own evaluation text, rather than left to a human to type correctly at
2am — and every inference reports how it was reached so a wrong guess is
visible in STATE.md instead of silent.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

HIGHER = "higher_better"
LOWER = "lower_better"

# Phrases Kaggle actually uses on evaluation pages. Checked before the name
# table, because an explicit statement beats a guess from a metric's name.
_EXPLICIT_LOWER = (
    "lower is better", "lower values are better", "smaller is better",
    "the lower the better", "minimize", "minimise", "minimizing", "minimising",
)
_EXPLICIT_HIGHER = (
    "higher is better", "higher values are better", "larger is better",
    "the higher the better", "maximize", "maximise", "maximizing", "maximising",
)

# Matched as whole words against the metric name, longest first, so that
# "root mean squared error" does not fall through to a bare "error" rule and
# "balanced accuracy" resolves the same way as "accuracy".
_LOWER_METRICS = (
    "root mean squared logarithmic error", "root mean squared error",
    "mean squared logarithmic error", "mean absolute percentage error",
    "symmetric mean absolute percentage error", "weighted mean absolute error",
    "mean absolute error", "mean squared error", "mean columnwise rmse",
    "continuous ranked probability score", "multi class log loss",
    "multiclass log loss", "binary log loss", "quantile loss", "pinball loss",
    "hamming loss", "brier score", "perplexity", "levenshtein", "edit distance",
    "wasserstein", "chamfer", "hausdorff", "rmsle", "rmspe", "rmse", "msle",
    "smape", "mape", "wmae", "mcrmse", "crps", "logloss", "log loss", "mae",
    "mse", "mad", "error rate", "error", "loss", "deviation", "distance",
    "regret", "cost", "penalty",
)
_HIGHER_METRICS = (
    "quadratic weighted kappa", "cohen kappa", "mean average precision",
    "average precision", "area under the roc curve", "area under curve",
    "balanced accuracy", "categorization accuracy", "top-1 accuracy",
    "macro f1", "micro f1", "weighted f1", "f1 score", "f-score", "fbeta",
    "matthews correlation", "spearman", "pearson", "correlation",
    "dice coefficient", "jaccard", "intersection over union", "iou",
    "normalized discounted cumulative gain", "ndcg", "mrr", "recall@",
    "precision@", "map@", "auroc", "auprc", "roc auc", "auc", "gini",
    "accuracy", "precision", "recall", "kappa", "dice", "r2", "r^2",
    "score", "similarity", "agreement", "win rate",
)


def direction_from_text(text: str) -> str | None:
    """Direction stated outright in the evaluation prose, if it is."""
    lowered = (text or "").lower()
    first_lower = min((lowered.find(p) for p in _EXPLICIT_LOWER if p in lowered), default=-1)
    first_higher = min((lowered.find(p) for p in _EXPLICIT_HIGHER if p in lowered), default=-1)
    if first_lower < 0 and first_higher < 0:
        return None
    if first_lower < 0:
        return HIGHER
    if first_higher < 0:
        return LOWER
    # Both appear; trust whichever Kaggle says first.
    return LOWER if first_lower < first_higher else HIGHER


def direction_from_metric(metric: str) -> str | None:
    """Direction implied by a metric's name."""
    name = re.sub(r"[^a-z0-9@^ ]+", " ", (metric or "").lower())
    name = re.sub(r"\s+", " ", name).strip()
    if not name:
        return None
    for phrase in sorted(_LOWER_METRICS, key=len, reverse=True):
        if phrase in name:
            return LOWER
    for phrase in sorted(_HIGHER_METRICS, key=len, reverse=True):
        if phrase in name:
            return HIGHER
    return None


def _fetch_info(slug: str):
    """Competition metadata from the Kaggle API, or None when unavailable."""
    from kernels.kaggle_client import KaggleKernelClient

    with KaggleKernelClient() as client:
        return client.get_competition_info(slug)


def _evaluation_text(slug: str) -> str:
    """The competition's Evaluation page, when it can be fetched."""
    try:
        from runtime import competition_pages

        pages = competition_pages(slug)
    except Exception as exc:  # noqa: BLE001 — inference must never break a tick
        logger.warning("could not fetch competition pages for %s: %s", slug, exc)
        return ""
    for key in ("evaluation", "overview/evaluation", "description"):
        if pages.get(key):
            return pages[key]
    return " ".join(v for v in pages.values() if isinstance(v, str))[:20000]


def infer(slug: str, *, metric: str | None = None, direction: str | None = None) -> dict:
    """Resolve metric, direction, deadline and title for a competition.

    Explicit arguments always win. Anything left unset is inferred, and
    ``sources`` records how — an inference you cannot audit is worse than none.
    """
    result = {
        "competition": slug,
        "metric": metric or "",
        "direction": direction,
        "deadline": "",
        "title": "",
        "url": f"https://www.kaggle.com/competitions/{slug}",
        "sources": {"metric": "argument" if metric else None,
                    "direction": "argument" if direction else None},
        "confident": bool(metric and direction),
    }

    info = None
    if not metric or not direction:
        try:
            info = _fetch_info(slug)
        except Exception as exc:  # noqa: BLE001 — see infer() docstring
            logger.warning("could not fetch competition info for %s: %s", slug, exc)

    if info is not None:
        result["title"] = info.title or ""
        result["url"] = info.url or result["url"]
        if info.deadline:
            result["deadline"] = str(info.deadline)
        if not result["metric"] and info.evaluation_metric:
            result["metric"] = info.evaluation_metric.strip()
            result["sources"]["metric"] = "kaggle api"

    if result["direction"] is None and result["metric"]:
        guess = direction_from_metric(result["metric"])
        if guess:
            result["direction"] = guess
            result["sources"]["direction"] = f"metric name ({result['metric']!r})"

    if result["direction"] is None or not result["metric"]:
        try:
            text = _evaluation_text(slug)
        except Exception as exc:  # noqa: BLE001 — inference must never break a tick
            logger.warning("could not read the evaluation page for %s: %s", slug, exc)
            text = ""
        if text:
            if result["direction"] is None:
                stated = direction_from_text(text)
                if stated:
                    result["direction"] = stated
                    result["sources"]["direction"] = "evaluation page wording"
            if not result["metric"]:
                guessed = metric_from_text(text)
                if guessed:
                    result["metric"] = guessed
                    result["sources"]["metric"] = "evaluation page text"
                    if result["direction"] is None:
                        implied = direction_from_metric(guessed)
                        if implied:
                            result["direction"] = implied
                            result["sources"]["direction"] = f"metric name ({guessed!r})"

    result["confident"] = bool(result["metric"] and result["direction"])
    return result


_METRIC_SENTENCE = re.compile(
    r"(?:submissions?|entries|solutions?|predictions?|results?)\s+(?:are|is|will be)\s+"
    r"(?:being\s+)?(?:evaluated|scored|judged|measured)\s+"
    r"(?:on|by|using|with|against)\s+(?:the\s+)?(?P<metric>[^.;:\n]{3,80})",
    re.IGNORECASE,
)


def metric_from_text(text: str) -> str | None:
    """Pull the metric out of a "submissions are evaluated on ..." sentence."""
    match = _METRIC_SENTENCE.search(text or "")
    if not match:
        return None
    metric = re.sub(r"\s+", " ", match.group("metric")).strip()
    # Trim trailing subordinate clauses: "RMSE between the predicted and ..."
    metric = re.split(r"\s+(?:between|across|over|for each|where|which)\b", metric,
                      maxsplit=1, flags=re.IGNORECASE)[0]
    # Strip markdown and punctuation last, so emphasis around the whole phrase
    # ("**balanced accuracy**") is removed after the clause has been cut off.
    metric = metric.strip(" \t*_`\"'()[].,;:")
    return metric[:80] or None
