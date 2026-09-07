"""Tests for autoprogramming.metric — approval lifecycle, hashing, pair scoring."""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pytest

from autoprogramming import metric
from autoprogramming.errors import (
    MetricChangedError,
    MetricNotApprovedError,
    SchemaError,
)
from autoprogramming.schema import Schema


class Loud(str):
    """The input, uppercased, with an exclamation mark."""


def shout(text: str) -> Loud:
    """Uppercase the text."""


class Answer(str):
    """Direct answer to the question, one sentence."""


class Confidence(float):
    """Calibrated probability that the answer is correct, 0.0-1.0."""


def qa(question: str) -> tuple[Answer, Confidence]:
    """Answer a factual question with a calibrated confidence."""


SHOUT_SCHEMA = Schema.from_function(shout)
QA_SCHEMA = Schema.from_function(qa)

EXACT = 'def metric(predicted, expected):\n    return 1.0 if predicted == expected else 0.0\n'
HALVES = 'def metric(predicted, expected):\n    return 0.5\n'


class FakeWorkspace:
    """Duck-typed stand-in for workspace.Workspace: just the paths metric.py uses."""

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.metric_py = root / "metric.py"
        self.metric_approval = root / "metric_approval.json"
        self.scores_json = root / "scores.json"
        self.budget_json = root / "budget.json"


@pytest.fixture(autouse=True)
def _no_auto_approve(monkeypatch):
    monkeypatch.delenv("AP_AUTO_APPROVE_METRIC", raising=False)


@pytest.fixture
def ws(tmp_path):
    return FakeWorkspace(tmp_path / "shout_ap")


def _seed_scores(ws, sha):
    scores = {
        "metric_sha": sha,
        "candidates": {"candidate_0": {"val": {"rows": {"row_0": 1.0}, "mean": 1.0}}},
        "val_scored": ["candidate_0"],
        "flags": {},
    }
    ws.scores_json.write_text(json.dumps(scores))
    return scores


# ---------------------------------------------------------------- metric_sha


def test_metric_sha_none_when_absent(ws):
    assert metric.metric_sha(ws) is None


def test_metric_sha_tracks_content(ws):
    metric.write_metric(ws, EXACT)
    first = metric.metric_sha(ws)
    assert isinstance(first, str) and len(first) == 64
    metric.write_metric(ws, HALVES)
    assert metric.metric_sha(ws) != first
    metric.write_metric(ws, EXACT)
    assert metric.metric_sha(ws) == first


# ------------------------------------------------------------------ approval


def test_approval_lifecycle_write_approve_sha_match(ws):
    metric.write_metric(ws, EXACT)
    assert not metric.is_approved(ws)
    with pytest.raises(MetricNotApprovedError) as exc:
        metric.ensure_approved(ws)
    assert "approve" in str(exc.value)

    metric.approve(ws, "lilly")
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    first_line = ws.metric_py.read_text().splitlines()[0]
    assert first_line == f"# metric.py — approved by lilly on {today}"
    assert metric.is_approved(ws)
    assert metric.ensure_approved(ws) is None

    record = json.loads(ws.metric_approval.read_text())
    assert record["sha"] == metric.metric_sha(ws)
    assert record["approved_by"] == "lilly"
    assert record["weights"] is None
    assert record["approved_at"]


def test_approve_requires_metric_file(ws):
    with pytest.raises(MetricNotApprovedError):
        metric.approve(ws, "lilly")


def test_reapprove_replaces_header_instead_of_stacking(ws):
    metric.write_metric(ws, EXACT)
    metric.approve(ws, "alice")
    metric.approve(ws, "bob")
    lines = ws.metric_py.read_text().splitlines()
    headers = [l for l in lines if l.startswith("# metric.py")]
    assert len(headers) == 1
    assert "bob" in headers[0]
    assert metric.is_approved(ws)


def test_ensure_approved_refuses_missing_metric(ws):
    with pytest.raises(MetricNotApprovedError) as exc:
        metric.ensure_approved(ws)
    assert "metric.py" in str(exc.value)


# ------------------------------------------------------- changed metric rules


def test_changed_metric_archives_scores_and_voids_approval(ws):
    metric.write_metric(ws, EXACT)
    metric.approve(ws, "lilly")
    _seed_scores(ws, metric.metric_sha(ws))

    with pytest.warns(UserWarning, match="never comparable"):
        metric.write_metric(ws, HALVES)

    archived = ws.root / "scores.archive" / "0.json"
    assert archived.exists()
    assert "candidate_0" in json.loads(archived.read_text())["candidates"]

    fresh = json.loads(ws.scores_json.read_text())
    assert fresh["candidates"] == {}
    assert fresh["val_scored"] == []
    assert fresh["metric_sha"] == metric.metric_sha(ws)

    assert not metric.is_approved(ws)
    with pytest.raises(MetricChangedError) as exc:
        metric.ensure_approved(ws)
    assert "re-approve" in str(exc.value).lower() or "approve" in str(exc.value)


def test_changed_metric_without_scores_archives_nothing(ws):
    metric.write_metric(ws, EXACT)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metric.write_metric(ws, HALVES)
    assert not (ws.root / "scores.archive").exists()


def test_rewriting_identical_metric_keeps_approval_and_scores(ws):
    metric.write_metric(ws, EXACT)
    metric.approve(ws, "lilly")
    _seed_scores(ws, metric.metric_sha(ws))
    same = ws.metric_py.read_text()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metric.write_metric(ws, same)
    assert metric.is_approved(ws)
    assert "candidate_0" in json.loads(ws.scores_json.read_text())["candidates"]


def test_second_archive_gets_next_index(ws):
    metric.write_metric(ws, EXACT)
    _seed_scores(ws, metric.metric_sha(ws))
    with pytest.warns(UserWarning):
        metric.write_metric(ws, HALVES)
    _seed_scores(ws, metric.metric_sha(ws))
    with pytest.warns(UserWarning):
        metric.write_metric(ws, EXACT)
    assert (ws.root / "scores.archive" / "0.json").exists()
    assert (ws.root / "scores.archive" / "1.json").exists()


# ------------------------------------------------------------- auto-approval


def test_auto_approve_only_inside_ensure_approved(ws, monkeypatch):
    monkeypatch.setenv("AP_AUTO_APPROVE_METRIC", "1")
    metric.write_metric(ws, EXACT)
    assert not metric.is_approved(ws)

    metric.ensure_approved(ws)
    assert metric.is_approved(ws)
    record = json.loads(ws.metric_approval.read_text())
    assert record["approved_by"] == "auto (AP_AUTO_APPROVE_METRIC)"


@pytest.mark.parametrize("value", ["true", "YES", " 1 "])
def test_auto_approve_truthy_values(ws, monkeypatch, value):
    monkeypatch.setenv("AP_AUTO_APPROVE_METRIC", value)
    metric.write_metric(ws, EXACT)
    metric.ensure_approved(ws)
    assert metric.is_approved(ws)


@pytest.mark.parametrize("value", ["0", "", "no", "false"])
def test_auto_approve_falsy_values_still_refuse(ws, monkeypatch, value):
    monkeypatch.setenv("AP_AUTO_APPROVE_METRIC", value)
    metric.write_metric(ws, EXACT)
    with pytest.raises(MetricNotApprovedError):
        metric.ensure_approved(ws)


def test_auto_approve_cannot_conjure_a_metric(ws, monkeypatch):
    monkeypatch.setenv("AP_AUTO_APPROVE_METRIC", "1")
    with pytest.raises(MetricNotApprovedError):
        metric.ensure_approved(ws)


def test_auto_approve_never_blesses_a_changed_metric(ws, monkeypatch):
    metric.write_metric(ws, EXACT)
    metric.approve(ws, "lilly")
    ws.metric_py.write_text(HALVES)
    monkeypatch.setenv("AP_AUTO_APPROVE_METRIC", "1")
    with pytest.raises(MetricChangedError):
        metric.ensure_approved(ws)


# ---------------------------------------------------------------- load/demo


def test_load_metric_returns_callable_and_caches(ws):
    metric.write_metric(ws, EXACT)
    fn = metric.load_metric(ws)
    assert fn("a", "a") == 1.0
    assert fn("a", "b") == 0.0
    assert metric.load_metric(ws) is fn

    metric.write_metric(ws, HALVES)
    fn2 = metric.load_metric(ws)
    assert fn2 is not fn
    assert fn2("a", "b") == 0.5


def test_load_metric_missing_file(ws):
    with pytest.raises(MetricNotApprovedError):
        metric.load_metric(ws)


def test_load_metric_requires_metric_function(ws):
    metric.write_metric(ws, "def score(p, e):\n    return 1.0\n")
    with pytest.raises(SchemaError, match="metric"):
        metric.load_metric(ws)


def test_load_metric_import_error(ws):
    metric.write_metric(ws, "1 / 0\n")
    with pytest.raises(SchemaError):
        metric.load_metric(ws)


def test_demonstrate_uses_current_unapproved_metric(ws):
    metric.write_metric(ws, EXACT)
    assert not metric.is_approved(ws)
    demo = metric.demonstrate(ws, [("HI!", "HI!"), ("HI!", "NO!")])
    assert demo == [
        {"predicted": "HI!", "expected": "HI!", "scores": {"quality": 1.0}, "score": 1.0},
        {"predicted": "HI!", "expected": "NO!", "scores": {"quality": 0.0}, "score": 0.0},
    ]


# ---------------------------------------------------------------- score_pair


def test_score_pair_single_output(ws):
    seen = []

    def fn(p, e):
        seen.append((p, e))
        return 0.25

    score, per_field = metric.score_pair(fn, SHOUT_SCHEMA, {"Loud": "HI!"}, {"Loud": "HO!"}, None)
    assert score == 0.25
    assert per_field is None
    assert seen == [("HI!", "HO!")]


def test_score_pair_single_output_non_numeric(ws):
    def fn(p, e):
        return {"oops": 1.0}

    with pytest.raises(SchemaError):
        metric.score_pair(fn, SHOUT_SCHEMA, {"Loud": "x"}, {"Loud": "x"}, None)


def test_score_pair_multi_output_equal_weights():
    def fn(p, e):
        return {"Answer": 1.0, "Confidence": 0.5}

    predicted = {"Answer": "Paris", "Confidence": 0.5}
    expected = {"Answer": "Paris", "Confidence": 0.9}
    score, per_field = metric.score_pair(fn, QA_SCHEMA, predicted, expected, None)
    assert score == pytest.approx(0.75)
    assert per_field == {"Answer": 1.0, "Confidence": 0.5}


def test_score_pair_multi_output_weighted():
    def fn(p, e):
        return {"Answer": 1.0, "Confidence": 0.5}

    score, _ = metric.score_pair(
        fn, QA_SCHEMA, {"Answer": "a", "Confidence": 0.1}, {"Answer": "a", "Confidence": 0.1},
        {"Answer": 3.0, "Confidence": 1.0},
    )
    assert score == pytest.approx((3.0 * 1.0 + 1.0 * 0.5) / 4.0)


def test_score_pair_multi_output_missing_key():
    def fn(p, e):
        return {"Answer": 1.0}

    with pytest.raises(SchemaError, match="Confidence"):
        metric.score_pair(fn, QA_SCHEMA, {"Answer": "a", "Confidence": 0.1},
                          {"Answer": "a", "Confidence": 0.1}, None)


def test_score_pair_multi_output_non_dict():
    def fn(p, e):
        return 0.9

    with pytest.raises(SchemaError):
        metric.score_pair(fn, QA_SCHEMA, {"Answer": "a", "Confidence": 0.1},
                          {"Answer": "a", "Confidence": 0.1}, None)


def test_score_pair_zero_weights_refused():
    def fn(p, e):
        return {"Answer": 1.0, "Confidence": 0.5}

    with pytest.raises(SchemaError, match="weight"):
        metric.score_pair(fn, QA_SCHEMA, {"Answer": "a", "Confidence": 0.1},
                          {"Answer": "a", "Confidence": 0.1},
                          {"Answer": 0.0, "Confidence": 0.0})


def test_reproposing_identical_raw_code_after_approval_keeps_everything(ws):
    """A fresh session re-proposing the very same metric must not wipe scores.

    approve() stamps a header line into metric.py, so the raw proposed code
    and the stamped on-disk file differ by exactly that line — which is
    sign-off metadata, not a metric change.
    """
    metric.write_metric(ws, EXACT)
    metric.approve(ws, "lilly")
    sha = metric.metric_sha(ws)
    _seed_scores(ws, sha)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        metric.write_metric(ws, EXACT)

    assert metric.metric_sha(ws) == sha
    assert metric.is_approved(ws)
    stored = json.loads(ws.scores_json.read_text())
    assert "candidate_0" in stored["candidates"]
    assert stored["val_scored"] == ["candidate_0"]
    assert not (ws.root / "scores.archive").exists()
    assert ws.metric_py.read_text().startswith("# metric.py — approved by lilly")


def test_reapproval_does_not_change_the_metric_sha(ws):
    metric.write_metric(ws, EXACT)
    metric.approve(ws, "lilly")
    sha = metric.metric_sha(ws)
    _seed_scores(ws, sha)

    metric.approve(ws, "someone else")

    assert metric.metric_sha(ws) == sha
    assert metric.is_approved(ws)
    record = json.loads(ws.metric_approval.read_text())
    assert record["approved_by"] == "someone else"
    assert ws.metric_py.read_text().startswith("# metric.py — approved by someone else")


# ------------------------------------------------------- multi-metric objectives


MULTI = (
    "def exact(predicted, expected):\n"
    "    return 1.0 if predicted == expected else 0.0\n"
    "\n\n"
    "def graded(predicted, expected):\n"
    "    return 0.5\n"
    "\n\n"
    "METRICS = {'exact': exact, 'graded': graded}\n"
)


def test_quality_metrics_single_defaults_to_quality(ws):
    metric.write_metric(ws, EXACT)
    metrics = metric.quality_metrics(ws)
    assert set(metrics) == {"quality"}
    assert metrics["quality"]("a", "a") == 1.0
    assert metrics["quality"]("a", "b") == 0.0


def test_quality_metrics_multi_returns_named_dict(ws):
    metric.write_metric(ws, MULTI)
    metrics = metric.quality_metrics(ws)
    assert list(metrics) == ["exact", "graded"]
    assert metrics["exact"]("a", "a") == 1.0
    assert metrics["graded"]("a", "b") == 0.5


def test_quality_metrics_dict_wins_over_single_metric(ws):
    metric.write_metric(ws, EXACT + "\nMETRICS = {'renamed': metric}\n")
    assert list(metric.quality_metrics(ws)) == ["renamed"]


def test_quality_metrics_empty_name_rejected(ws):
    metric.write_metric(ws, "def m(p, e):\n    return 1.0\nMETRICS = {'': m}\n")
    with pytest.raises(SchemaError, match="name"):
        metric.quality_metrics(ws)


def test_quality_metrics_non_callable_rejected(ws):
    metric.write_metric(ws, "METRICS = {'x': 5}\n")
    with pytest.raises(SchemaError, match="callable"):
        metric.quality_metrics(ws)


def test_quality_metrics_empty_dict_rejected(ws):
    metric.write_metric(ws, "METRICS = {}\n")
    with pytest.raises(SchemaError, match="non-empty"):
        metric.quality_metrics(ws)


@pytest.mark.parametrize("reserved", ["cost_dollars", "latency_s"])
def test_quality_metrics_collision_with_cost_objective(ws, reserved):
    metric.write_metric(ws, f"def m(p, e):\n    return 1.0\nMETRICS = {{{reserved!r}: m}}\n")
    with pytest.raises(SchemaError, match="cost objective"):
        metric.quality_metrics(ws)


def test_quality_metrics_neither_form_defined(ws):
    metric.write_metric(ws, "def score(p, e):\n    return 1.0\n")
    with pytest.raises(SchemaError, match="metric"):
        metric.quality_metrics(ws)


def test_primary_name_single_metric(ws):
    metric.write_metric(ws, EXACT)
    assert metric.primary_name(ws) == "quality"


def test_primary_name_defaults_to_first_metric_key(ws):
    metric.write_metric(ws, MULTI)
    assert metric.primary_name(ws) == "exact"


def test_primary_name_uses_approved_primary(ws):
    metric.write_metric(ws, MULTI)
    metric.approve(ws, "tester", primary="graded")
    assert metric.primary_name(ws) == "graded"


def test_primary_name_falls_back_when_approval_primary_invalid(ws):
    metric.write_metric(ws, MULTI)
    metric.approve(ws, "tester", primary="graded")
    record = json.loads(ws.metric_approval.read_text())
    record["primary"] = "ghost"  # names a metric that isn't defined
    ws.metric_approval.write_text(json.dumps(record))
    assert metric.primary_name(ws) == "exact"


def test_approve_primary_must_name_a_defined_metric(ws):
    metric.write_metric(ws, MULTI)
    with pytest.raises(SchemaError, match="primary"):
        metric.approve(ws, "tester", primary="nonexistent")
    assert not ws.metric_approval.exists()


def test_approve_records_primary_and_none(ws):
    metric.write_metric(ws, MULTI)
    metric.approve(ws, "tester", primary="graded")
    assert json.loads(ws.metric_approval.read_text())["primary"] == "graded"
    metric.write_metric(ws, EXACT)
    metric.approve(ws, "tester")
    assert json.loads(ws.metric_approval.read_text())["primary"] is None


def test_changing_primary_does_not_change_metric_sha(ws):
    metric.write_metric(ws, MULTI)
    metric.approve(ws, "tester", primary="exact")
    sha = metric.metric_sha(ws)
    metric.approve(ws, "tester", primary="graded")
    assert metric.metric_sha(ws) == sha  # primary is config, not code


def test_load_metric_returns_primary_callable(ws):
    metric.write_metric(ws, MULTI)
    metric.approve(ws, "tester", primary="graded")
    fn = metric.load_metric(ws)
    assert fn("a", "b") == 0.5  # the 'graded' primary
    metric.approve(ws, "tester", primary="exact")
    assert metric.load_metric(ws)("a", "b") == 0.0  # now the 'exact' primary


def test_demonstrate_multi_metric_scores_all_with_primary_mirror(ws):
    metric.write_metric(ws, MULTI)
    metric.approve(ws, "tester", primary="graded")
    demo = metric.demonstrate(ws, [("a", "a"), ("a", "b")])
    assert demo == [
        {"predicted": "a", "expected": "a",
         "scores": {"exact": 1.0, "graded": 0.5}, "score": 0.5},
        {"predicted": "a", "expected": "b",
         "scores": {"exact": 0.0, "graded": 0.5}, "score": 0.5},
    ]
