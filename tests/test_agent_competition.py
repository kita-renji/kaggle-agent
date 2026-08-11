# SPDX-License-Identifier: MIT
"""Tests for metric/direction inference and slug resolution.

``direction`` is the field that silently corrupts every comparison when wrong,
so it gets the most coverage here.
"""

import pytest

from agent import competition, ledger


# --------------------------------------------------------------------------
# Direction from the metric's name
# --------------------------------------------------------------------------

@pytest.mark.parametrize("metric,expected", [
    ("accuracy", competition.HIGHER),
    ("Categorization Accuracy", competition.HIGHER),
    ("AUC", competition.HIGHER),
    ("area under the ROC curve", competition.HIGHER),
    ("macro F1", competition.HIGHER),
    ("Quadratic Weighted Kappa", competition.HIGHER),
    ("mean average precision @ 5", competition.HIGHER),
    ("Dice coefficient", competition.HIGHER),
    ("R2", competition.HIGHER),
    ("RMSE", competition.LOWER),
    ("Root Mean Squared Logarithmic Error", competition.LOWER),
    ("mean absolute error", competition.LOWER),
    ("wMAE", competition.LOWER),
    ("multi class log loss", competition.LOWER),
    ("LogLoss", competition.LOWER),
    ("SMAPE", competition.LOWER),
    ("Continuous Ranked Probability Score", competition.LOWER),
    ("Levenshtein distance", competition.LOWER),
])
def test_direction_from_metric_name(metric, expected):
    assert competition.direction_from_metric(metric) == expected


def test_longer_metric_names_win_over_substrings():
    """"root mean squared error" must not resolve via a bare "score"/"error" rule."""
    assert competition.direction_from_metric("root mean squared error") == competition.LOWER
    # "balanced accuracy" contains neither a lower-metric phrase nor a bare fallback trap.
    assert competition.direction_from_metric("balanced accuracy") == competition.HIGHER


def test_unknown_metric_yields_no_guess():
    assert competition.direction_from_metric("bespoke leaderboard points") is None
    assert competition.direction_from_metric("") is None
    assert competition.direction_from_metric(None) is None


# --------------------------------------------------------------------------
# Direction stated outright
# --------------------------------------------------------------------------

def test_direction_from_explicit_wording():
    assert competition.direction_from_text("Submissions are scored on wMAE. Lower is better.") \
        == competition.LOWER
    assert competition.direction_from_text("The metric is custom; higher is better.") \
        == competition.HIGHER
    assert competition.direction_from_text("Goal: minimize the reconstruction error.") \
        == competition.LOWER
    assert competition.direction_from_text("No statement here at all.") is None


def test_first_explicit_statement_wins():
    text = "Lower is better for this metric, though higher is better on the secondary chart."
    assert competition.direction_from_text(text) == competition.LOWER


# --------------------------------------------------------------------------
# Metric from prose
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Submissions are evaluated on the mean columnwise RMSE between the predicted and actual values.",
     "mean columnwise RMSE"),
    ("Entries are scored by quadratic weighted kappa.", "quadratic weighted kappa"),
    ("Predictions are judged using **balanced accuracy** across all classes.",
     "balanced accuracy"),
])
def test_metric_from_text(text, expected):
    assert competition.metric_from_text(text) == expected


def test_metric_from_text_returns_none_without_the_pattern():
    assert competition.metric_from_text("This competition is about polymers.") is None
    assert competition.metric_from_text("") is None


# --------------------------------------------------------------------------
# infer(): precedence and degradation
# --------------------------------------------------------------------------

def test_explicit_arguments_short_circuit_every_lookup(agent_workspace, monkeypatch):
    def fail(*_a, **_k):
        raise AssertionError("must not hit the network when both fields are given")

    monkeypatch.setattr(competition, "_evaluation_text", fail)
    result = competition.infer("titanic", metric="accuracy", direction="higher_better")

    assert result["metric"] == "accuracy"
    assert result["direction"] == "higher_better"
    assert result["confident"] is True
    assert result["sources"] == {"metric": "argument", "direction": "argument"}


def test_infer_uses_the_api_metric_then_its_name(agent_workspace, monkeypatch):
    monkeypatch.setattr(competition, "_evaluation_text", lambda _slug: "")
    monkeypatch.setattr(competition, "_fetch_info", lambda _slug: _Info(
        title="Polymer Prediction", evaluation_metric="wMAE",
        deadline="2026-09-15T23:59:00Z"))

    result = competition.infer("nopp")
    assert result["metric"] == "wMAE"
    assert result["direction"] == competition.LOWER
    assert result["deadline"] == "2026-09-15T23:59:00Z"
    assert result["sources"]["metric"] == "kaggle api"
    assert "metric name" in result["sources"]["direction"]
    assert result["confident"] is True


def test_evaluation_wording_beats_an_unrecognised_metric_name(agent_workspace, monkeypatch):
    monkeypatch.setattr(competition, "_fetch_info", lambda _slug: _Info(
        evaluation_metric="Bespoke Leaderboard Points"))
    monkeypatch.setattr(competition, "_evaluation_text",
                        lambda _slug: "Scores are ranked; lower is better.")

    result = competition.infer("weird")
    assert result["metric"] == "Bespoke Leaderboard Points"
    assert result["direction"] == competition.LOWER
    assert result["sources"]["direction"] == "evaluation page wording"


def test_infer_falls_back_to_prose_for_the_metric(agent_workspace, monkeypatch):
    monkeypatch.setattr(competition, "_fetch_info", lambda _slug: _Info())
    monkeypatch.setattr(
        competition, "_evaluation_text",
        lambda _slug: "Submissions are evaluated on the mean absolute error between predictions.")

    result = competition.infer("some-comp")
    assert result["metric"] == "mean absolute error"
    assert result["direction"] == competition.LOWER
    assert result["sources"]["metric"] == "evaluation page text"


def test_infer_degrades_when_kaggle_is_unreachable(agent_workspace, monkeypatch):
    def boom(_slug):
        raise RuntimeError("kaggle is down")

    monkeypatch.setattr(competition, "_fetch_info", boom)
    monkeypatch.setattr(competition, "_evaluation_text", boom)

    result = competition.infer("titanic")
    assert result["metric"] == ""
    assert result["direction"] is None
    assert result["confident"] is False


class _Info:
    def __init__(self, title="", evaluation_metric="", deadline=None, url=""):
        self.title = title
        self.evaluation_metric = evaluation_metric
        self.deadline = deadline
        self.url = url


# --------------------------------------------------------------------------
# Slug resolution
# --------------------------------------------------------------------------

def test_explicit_slug_wins(agent_workspace):
    assert ledger.resolve_slug("titanic") == "titanic"


def test_explicit_url_is_reduced_to_a_slug(agent_workspace):
    assert ledger.resolve_slug("https://www.kaggle.com/competitions/birdclef-2025") == "birdclef-2025"


def test_slug_comes_from_the_competition_workspace_directory(agent_workspace):
    workspace = ledger.competition_dir("birdclef-2025")
    (workspace / "kernels" / "demo").mkdir(parents=True)

    assert ledger.resolve_slug(cwd=workspace) == "birdclef-2025"
    # Also from anywhere beneath it.
    assert ledger.resolve_slug(cwd=workspace / "kernels" / "demo") == "birdclef-2025"


def test_slug_falls_back_to_the_directory_name(agent_workspace, tmp_path):
    sibling = tmp_path.parent / "arc-prize-2025"
    sibling.mkdir(exist_ok=True)
    assert ledger.resolve_slug(cwd=sibling) == "arc-prize-2025"


def test_project_root_is_refused_rather_than_guessed(agent_workspace):
    with pytest.raises(ledger.SlugError, match="project root"):
        ledger.resolve_slug(cwd=ledger.project_root())


def test_competitions_dir_itself_is_refused(agent_workspace):
    competitions = ledger.competitions_root()
    competitions.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ledger.SlugError, match="competitions/ directory itself"):
        ledger.resolve_slug(cwd=competitions)
