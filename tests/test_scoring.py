"""Tests for autoprogramming.scoring — bootstrap stats, evaluate order, compare.

Sibling modules that other groups own (guards, data, candidates, runner) are
stubbed into sys.modules per-test, so this file passes standalone.
"""

from __future__ import annotations

import json
import sys
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoprogramming import metric, scoring
from autoprogramming.budget import Budget, BudgetLedger
from autoprogramming.errors import (
    BudgetExceededError,
    DataDisciplineError,
    MemorizationWarning,
    MetricNotApprovedError,
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
MULTI_METRIC = (
    "def metric(predicted, expected):\n"
    "    answer = 1.0 if predicted['Answer'] == expected['Answer'] else 0.0\n"
    "    conf = 1.0 - abs(float(predicted['Confidence']) - float(expected['Confidence']))\n"
    "    return {'Answer': answer, 'Confidence': conf}\n"
)
CAND_SOURCE = 'def predict(text):\n    return text.upper() + "!"\n'

SPLITS = {
    "train": [
        {"text": "hi", "Loud": "HI!"},
        {"text": "yo", "Loud": "YO!"},
        {"text": "bad", "Loud": "WRONG"},
    ],
    "val": [
        {"text": "ok", "Loud": "OK!"},
        {"text": "go", "Loud": "GO!"},
    ],
}


class FakeWorkspace:
    """Duck-typed stand-in for workspace.Workspace: paths plus a schema."""

    def __init__(self, root: Path, schema: Schema):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self.schema = schema
        self.metric_py = root / "metric.py"
        self.metric_approval = root / "metric_approval.json"
        self.scores_json = root / "scores.json"
        self.budget_json = root / "budget.json"


def good_run(inputs):
    return SimpleNamespace(
        ok=True, outputs={"Loud": inputs["text"].upper() + "!"}, error=None,
        cost_dollars=None,
    )


def install_fakes(monkeypatch, *, splits=SPLITS, candidate=None, run_fn=good_run,
                  memo_flags=None, calls=None):
    """Stub guards/data/candidates/runner in sys.modules for one test."""
    guards = types.ModuleType("autoprogramming.guards")

    def assert_eval_allowed(split, per_instance):
        if split == "test":
            raise DataDisciplineError("test belongs to finalize()")
        if split not in ("train", "val"):
            raise DataDisciplineError(f"unknown split {split!r}")
        if split == "val" and per_instance:
            raise DataDisciplineError("per-row val scores are refused")

    def register_val_candidate(ws, name):
        s = scoring.load_scores(ws)
        if name not in s["val_scored"]:
            s["val_scored"].append(name)
            scoring.save_scores(ws, s)

    guards.assert_eval_allowed = assert_eval_allowed
    guards.register_val_candidate = register_val_candidate
    guards.memorization_check = (
        lambda source, train_mean, val_mean, train_rows, schema: list(memo_flags or [])
    )

    data = types.ModuleType("autoprogramming.data")
    data.load_split = lambda ws, split: [dict(r) for r in splits[split]]

    cands = types.ModuleType("autoprogramming.candidates")
    cands.load_candidate = lambda ws, name: candidate

    runner = types.ModuleType("autoprogramming.runner")

    def run_candidate(ws, cand, inputs, timeout=120.0):
        if calls is not None:
            calls.append(dict(inputs))
        return run_fn(inputs)

    runner.run_candidate = run_candidate

    for name, mod in (("guards", guards), ("data", data),
                      ("candidates", cands), ("runner", runner)):
        monkeypatch.setitem(sys.modules, f"autoprogramming.{name}", mod)


def make_candidate(deterministic=True, name="candidate_0"):
    return SimpleNamespace(name=name, source=CAND_SOURCE, deterministic=deterministic)


@pytest.fixture(autouse=True)
def _no_auto_approve(monkeypatch):
    monkeypatch.delenv("AP_AUTO_APPROVE_METRIC", raising=False)


@pytest.fixture
def ws(tmp_path):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    metric.write_metric(workspace, EXACT)
    metric.approve(workspace, "tester")
    BudgetLedger.start(workspace.budget_json, Budget(eval_calls=1000))
    return workspace


# ------------------------------------------------------------- bootstrap_ci


def test_bootstrap_ci_empty():
    assert scoring.bootstrap_ci([]) == (0.0, 0.0)


def test_bootstrap_ci_single_value():
    assert scoring.bootstrap_ci([0.7]) == (0.7, 0.7)


def test_bootstrap_ci_constant_list_is_degenerate():
    assert scoring.bootstrap_ci([0.5] * 10) == (0.5, 0.5)


def test_bootstrap_ci_seeded_and_bounded():
    vals = [0.1, 0.4, 0.35, 0.9, 0.6, 0.2]
    lo1, hi1 = scoring.bootstrap_ci(vals, seed=1)
    lo2, hi2 = scoring.bootstrap_ci(vals, seed=1)
    assert (lo1, hi1) == (lo2, hi2)
    assert scoring.bootstrap_ci(vals, seed=2) != (lo1, hi1)
    mean = sum(vals) / len(vals)
    assert min(vals) <= lo1 <= mean <= hi1 <= max(vals)
    assert lo1 < hi1


def test_bootstrap_ci_two_point_hand_check():
    lo, hi = scoring.bootstrap_ci([0.0, 1.0])
    assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------- paired_bootstrap_diff


def test_paired_diff_constant_shift_is_exact():
    assert scoring.paired_bootstrap_diff([1.0, 2.0, 3.0], [1.5, 2.5, 3.5]) == (0.5, 0.5)


def test_paired_diff_identical_is_zero():
    assert scoring.paired_bootstrap_diff([0.3, 0.6], [0.3, 0.6]) == (0.0, 0.0)


def test_paired_diff_length_mismatch():
    with pytest.raises(ValueError):
        scoring.paired_bootstrap_diff([1.0], [1.0, 2.0])


# ---------------------------------------------------------- scores.json I/O


def test_load_scores_skeleton_without_file(tmp_path):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    assert scoring.load_scores(workspace) == {
        "metric_sha": None, "objectives": {}, "primary": None,
        "candidates": {}, "val_scored": [], "flags": {},
    }


def test_save_and_load_roundtrip(ws):
    scores = scoring.load_scores(ws)
    scores["candidates"]["candidate_0"] = {"val": {"rows": {"row_0": 1.0}, "mean": 1.0}}
    scoring.save_scores(ws, scores)
    assert scoring.load_scores(ws) == scores


def test_load_scores_archives_when_metric_changed_underneath(ws):
    scores = scoring.load_scores(ws)
    scores["candidates"]["candidate_0"] = {"val": {"rows": {"row_0": 1.0}, "mean": 1.0}}
    scoring.save_scores(ws, scores)

    ws.metric_py.write_text("def metric(p, e):\n    return 0.5\n")
    with pytest.warns(UserWarning, match="never comparable"):
        fresh = scoring.load_scores(ws)

    assert fresh["candidates"] == {}
    assert fresh["metric_sha"] == metric.metric_sha(ws)
    archived = ws.root / "scores.archive" / "0.json"
    assert "candidate_0" in json.loads(archived.read_text())["candidates"]


# ------------------------------------------------------------------ evaluate


def test_evaluate_train_per_instance(ws, monkeypatch):
    install_fakes(monkeypatch, candidate=make_candidate())
    rep = scoring.evaluate(ws, "candidate_0", split="train", per_instance=True)

    assert rep.candidate == "candidate_0"
    assert rep.split == "train"
    assert rep.per_row == {"row_0": 1.0, "row_1": 1.0, "row_2": 0.0}
    assert rep.mean == pytest.approx(2 / 3)
    assert rep.std == pytest.approx((2 / 9) ** 0.5)
    assert rep.n_rows == 3
    assert rep.n_repeats == 1
    assert rep.repeat_variance == 0.0
    assert rep.errors == []
    assert 0.0 <= rep.ci95[0] <= rep.mean <= rep.ci95[1] <= 1.0

    stored = json.loads(ws.scores_json.read_text())
    assert stored["candidates"]["candidate_0"]["train"]["rows"] == rep.per_row


def test_evaluate_train_without_per_instance_hides_per_row(ws, monkeypatch):
    install_fakes(monkeypatch, candidate=make_candidate())
    rep = scoring.evaluate(ws, "candidate_0", split="train")
    assert rep.per_row is None
    stored = json.loads(ws.scores_json.read_text())
    assert stored["candidates"]["candidate_0"]["train"]["rows"] == {
        "row_0": 1.0, "row_1": 1.0, "row_2": 0.0,
    }


def test_evaluate_val_returns_aggregate_but_persists_rows(ws, monkeypatch):
    install_fakes(monkeypatch, candidate=make_candidate())
    rep = scoring.evaluate(ws, "candidate_0", split="val")

    assert rep.per_row is None
    assert rep.mean == 1.0
    stored = json.loads(ws.scores_json.read_text())
    assert stored["candidates"]["candidate_0"]["val"]["rows"] == {"row_0": 1.0, "row_1": 1.0}
    assert stored["val_scored"] == ["candidate_0"]


def test_guards_run_before_metric_approval(tmp_path, monkeypatch):
    workspace = FakeWorkspace(tmp_path / "bare_ap", SHOUT_SCHEMA)
    install_fakes(monkeypatch, candidate=make_candidate())
    with pytest.raises(DataDisciplineError):
        scoring.evaluate(workspace, "candidate_0", split="test")
    with pytest.raises(DataDisciplineError):
        scoring.evaluate(workspace, "candidate_0", split="val", per_instance=True)
    with pytest.raises(DataDisciplineError):
        scoring.evaluate(workspace, "candidate_0", split="bogus")


def test_metric_approval_checked_before_any_run(tmp_path, monkeypatch):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    metric.write_metric(workspace, EXACT)
    calls = []
    install_fakes(monkeypatch, candidate=make_candidate(), calls=calls)
    with pytest.raises(MetricNotApprovedError):
        scoring.evaluate(workspace, "candidate_0", split="val")
    assert calls == []
    assert "candidate_0" not in scoring.load_scores(workspace)["candidates"]


def test_auto_approve_env_flows_through_evaluate(tmp_path, monkeypatch):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    metric.write_metric(workspace, EXACT)
    BudgetLedger.start(workspace.budget_json, Budget(eval_calls=1000))
    monkeypatch.setenv("AP_AUTO_APPROVE_METRIC", "1")
    install_fakes(monkeypatch, candidate=make_candidate())

    rep = scoring.evaluate(workspace, "candidate_0", split="val")
    assert rep.mean == 1.0
    record = json.loads(workspace.metric_approval.read_text())
    assert record["approved_by"] == "auto (AP_AUTO_APPROVE_METRIC)"


def test_repeats_deterministic_one_stochastic_three(ws, monkeypatch):
    calls = []
    install_fakes(monkeypatch, candidate=make_candidate(deterministic=True), calls=calls)
    scoring.evaluate(ws, "candidate_0", split="train")
    assert len(calls) == 3

    calls.clear()
    install_fakes(monkeypatch, candidate=make_candidate(deterministic=False), calls=calls)
    rep = scoring.evaluate(ws, "candidate_0", split="val")
    assert len(calls) == 6
    assert rep.n_repeats == scoring.DEFAULT_REPEATS == 3


def test_explicit_n_repeats_overrides(ws, monkeypatch):
    calls = []
    install_fakes(monkeypatch, candidate=make_candidate(deterministic=True), calls=calls)
    rep = scoring.evaluate(ws, "candidate_0", split="val", n_repeats=2)
    assert len(calls) == 4
    assert rep.n_repeats == 2


def test_budget_checked_before_each_row_and_abort_persists_nothing(tmp_path, monkeypatch):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    metric.write_metric(workspace, EXACT)
    metric.approve(workspace, "tester")
    BudgetLedger.start(workspace.budget_json, Budget(eval_calls=2))

    def costly(inputs):
        run = good_run(inputs)
        run.cost_dollars = 0.5
        return run

    install_fakes(monkeypatch, candidate=make_candidate(), run_fn=costly)
    with pytest.raises(BudgetExceededError):
        scoring.evaluate(workspace, "candidate_0", split="train")

    assert "candidate_0" not in scoring.load_scores(workspace)["candidates"]
    ledger = BudgetLedger(workspace.budget_json)
    assert ledger.spent["eval_calls"] == 2
    assert ledger.spent["dollars"] == pytest.approx(1.0)


def test_failed_run_scores_zero_and_records_error(ws, monkeypatch):
    def sometimes_fail(inputs):
        if inputs["text"] == "yo":
            return SimpleNamespace(
                ok=False, outputs=None, cost_dollars=None,
                error="Traceback (most recent call last):\n  ...\nValueError: boom",
            )
        return good_run(inputs)

    install_fakes(monkeypatch, candidate=make_candidate(), run_fn=sometimes_fail)
    rep = scoring.evaluate(ws, "candidate_0", split="train", per_instance=True)
    assert rep.per_row == {"row_0": 1.0, "row_1": 0.0, "row_2": 0.0}
    assert rep.errors == ["row_1 repeat 1: ValueError: boom"]


def test_repeat_variance_across_stochastic_repeats(tmp_path, monkeypatch):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    metric.write_metric(workspace, EXACT)
    metric.approve(workspace, "tester")
    BudgetLedger.start(workspace.budget_json, Budget(eval_calls=1000))

    state = {"n": 0}

    def flaky(inputs):
        state["n"] += 1
        out = inputs["text"].upper() + "!" if state["n"] == 1 else "XX"
        return SimpleNamespace(ok=True, outputs={"Loud": out}, error=None, cost_dollars=None)

    splits = {"train": [{"text": "hi", "Loud": "HI!"}], "val": []}
    install_fakes(monkeypatch, splits=splits, run_fn=flaky,
                  candidate=make_candidate(deterministic=False))
    rep = scoring.evaluate(workspace, "candidate_0", split="train")

    assert rep.n_repeats == 3
    assert rep.mean == pytest.approx(1 / 3)
    assert rep.repeat_variance == pytest.approx(2 / 9)
    assert rep.ci95 == (pytest.approx(1 / 3), pytest.approx(1 / 3))


def test_memorization_flags_stored_and_warned(ws, monkeypatch):
    flags = ["memorizer: train 1.00 vs val 0.20 — verbatim train outputs in source"]
    install_fakes(monkeypatch, candidate=make_candidate(), memo_flags=flags)

    with warnings.catch_warnings():
        warnings.simplefilter("error", MemorizationWarning)
        train_rep = scoring.evaluate(ws, "candidate_0", split="train")
    assert train_rep.flags == []

    with pytest.warns(MemorizationWarning):
        val_rep = scoring.evaluate(ws, "candidate_0", split="val")
    assert val_rep.flags == flags
    stored = json.loads(ws.scores_json.read_text())
    assert stored["flags"]["candidate_0"] == flags
    assert str(val_rep).splitlines()[1] == flags[0]


def test_multi_output_per_field_and_weights(tmp_path, monkeypatch):
    workspace = FakeWorkspace(tmp_path / "qa_ap", QA_SCHEMA)
    metric.write_metric(workspace, MULTI_METRIC)
    metric.approve(workspace, "tester", weights={"Answer": 3.0, "Confidence": 1.0})
    BudgetLedger.start(workspace.budget_json, Budget(eval_calls=1000))

    splits = {
        "train": [],
        "val": [{"question": "capital of France?", "Answer": "Paris", "Confidence": "0.9"}],
    }

    def qa_run(inputs):
        return SimpleNamespace(
            ok=True, outputs={"Answer": "Paris", "Confidence": 0.5},
            error=None, cost_dollars=None,
        )

    cand = SimpleNamespace(name="candidate_0", source="def predict(question): ...",
                           deterministic=True)
    install_fakes(monkeypatch, splits=splits, candidate=cand, run_fn=qa_run)
    rep = scoring.evaluate(workspace, "candidate_0", split="val")

    assert rep.mean == pytest.approx(0.9)
    assert rep.per_field["Answer"] == pytest.approx(1.0)
    assert rep.per_field["Confidence"] == pytest.approx(0.6)
    stored = json.loads(workspace.scores_json.read_text())
    assert stored["candidates"]["candidate_0"]["val"]["per_field"]["Confidence"] == pytest.approx(0.6)


def test_evaluate_rejects_zero_repeats(ws, monkeypatch):
    install_fakes(monkeypatch, candidate=make_candidate())
    with pytest.raises(ValueError):
        scoring.evaluate(ws, "candidate_0", split="val", n_repeats=0)


# ------------------------------------------------------------------- compare


def _store_rows(workspace, name, split, rows):
    scores = scoring.load_scores(workspace)
    scores["candidates"].setdefault(name, {})[split] = {
        "rows": rows,
        "mean": sum(rows.values()) / len(rows),
        "std": 0.0, "ci95": [0.0, 1.0], "n_repeats": 1,
        "repeat_variance": 0.0, "per_field": None,
    }
    scoring.save_scores(workspace, scores)


def test_compare_improved_on_constant_shift(tmp_path):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    _store_rows(workspace, "candidate_0", "val", {"row_0": 0.5, "row_1": 0.25, "row_2": 0.75})
    _store_rows(workspace, "candidate_1", "val", {"row_0": 1.0, "row_1": 0.75, "row_2": 1.25})

    rep = scoring.compare(workspace, "candidate_0", "candidate_1")
    assert rep.a == "candidate_0" and rep.b == "candidate_1"
    assert rep.diff_mean == pytest.approx(0.5)
    assert rep.ci95 == (0.5, 0.5)
    assert rep.improved
    assert "improved" in str(rep)


def test_compare_not_improved_when_ci_includes_zero(tmp_path):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    _store_rows(workspace, "candidate_0", "val", {"row_0": 0.5, "row_1": 0.5, "row_2": 0.5, "row_3": 0.5})
    _store_rows(workspace, "candidate_1", "val", {"row_0": 0.9, "row_1": 0.1, "row_2": 0.9, "row_3": 0.1})

    rep = scoring.compare(workspace, "candidate_0", "candidate_1")
    assert not rep.improved
    assert rep.ci95[0] <= 0.0
    assert "not improved" in str(rep)


def test_compare_pairs_by_row_id_intersection(tmp_path):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    _store_rows(workspace, "candidate_0", "val", {"row_0": 0.0, "row_1": 1.0})
    _store_rows(workspace, "candidate_1", "val", {"row_1": 1.2, "row_0": 0.3, "row_2": 9.9})

    rep = scoring.compare(workspace, "candidate_0", "candidate_1")
    assert rep.diff_mean == pytest.approx(0.25)
    assert rep.improved


def test_compare_requires_stored_scores(tmp_path):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    _store_rows(workspace, "candidate_0", "val", {"row_0": 0.5})
    with pytest.raises(DataDisciplineError, match="eval"):
        scoring.compare(workspace, "candidate_0", "candidate_1")
    with pytest.raises(DataDisciplineError):
        scoring.compare(workspace, "candidate_0", "candidate_0", split="train")


# ------------------------------------------------------------------ reports


def test_eval_report_str_matches_readme_shape():
    rep = scoring.EvalReport(
        candidate="candidate_1", split="val", mean=0.914, std=0.031,
        ci95=(0.860, 0.950), n_rows=10, n_repeats=3, repeat_variance=0.001,
        per_row=None, per_field=None, errors=[], flags=[],
    )
    assert str(rep) == (
        "candidate_1 on val: 0.914 ± 0.031 (n=3 repeats), 95% CI [0.860, 0.950]"
    )


def test_compare_report_str_names_both_candidates():
    rep = scoring.CompareReport(
        a="candidate_0", b="candidate_1", split="val",
        diff_mean=0.08, ci95=(0.02, 0.14), improved=True,
    )
    text = str(rep)
    assert "candidate_1" in text and "candidate_0" in text
    assert "+0.080" in text and "CI" in text


def test_metric_edit_and_reapproval_keeps_val_registration(ws, monkeypatch):
    """The MetricChangedError recovery flow (edit + re-approve) must not
    silently drop a candidate from val_scored on its next val eval.

    Uses the REAL guards module so registration goes through the same
    scores.json writes production uses; only data/candidates/runner are faked.
    """
    import importlib

    real_guards = importlib.import_module("autoprogramming.guards")
    install_fakes(monkeypatch, candidate=make_candidate())
    monkeypatch.setitem(sys.modules, "autoprogramming.guards", real_guards)
    (ws.root / "data").mkdir(exist_ok=True)
    ws.split_json = ws.root / "data" / "split.json"
    ws.split_json.write_text(json.dumps({
        "seed": 0,
        "ratios": [0.6, 0.2, 0.2],
        "counts": {"train": 3, "val": 50, "test": 2},
        "data_sha": "0" * 64,
        "bootstrap": False,
    }))

    scoring.evaluate(ws, "candidate_0", split="val")
    stored = json.loads(ws.scores_json.read_text())
    assert stored["val_scored"] == ["candidate_0"]

    # The prescribed recovery flow: edit metric.py in place, re-approve it.
    ws.metric_py.write_text(
        ws.metric_py.read_text() + "# tightened after review\n"
    )
    metric.approve(ws, "tester")

    # The candidate's outputs were cached by the first eval, so the metric
    # change re-scores from cache instead of wiping — no "never comparable"
    # archive warning fires (only stale/absent caches warn).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = scoring.evaluate(ws, "candidate_0", split="val")
    assert not any("never comparable" in str(w.message) for w in caught)
    assert report.mean == 1.0

    stored = json.loads(ws.scores_json.read_text())
    assert stored["val_scored"] == ["candidate_0"]
    assert stored["metric_sha"] == metric.metric_sha(ws)
    assert "candidate_0" in stored["candidates"]
    # Recovered from cache, so nothing was archived (archive is the last resort
    # for candidates whose code changed or was never cached).
    assert not (ws.root / "scores.archive" / "0.json").exists()


# ------------------------------------------------------- multi-objective scoring


def test_aggregate_reports_cold_start_separately_from_warm_latency():
    cache = {
        "row_0": [{
            "ok": True, "outputs": {"Loud": "A!"}, "cost_dollars": 0.0,
            "latency_s": 1.5, "cold_start": True,
        }],
        "row_1": [{
            "ok": True, "outputs": {"Loud": "B!"}, "cost_dollars": 0.0,
            "latency_s": 0.1, "cold_start": False,
        }],
    }
    expected = {"row_0": {"Loud": "A!"}, "row_1": {"Loud": "B!"}}
    sub = scoring._aggregate_from_cache(
        cache,
        {"quality": lambda p, e: float(p == e)},
        "quality",
        SHOUT_SCHEMA,
        None,
        expected,
        1,
    )
    assert sub["cold_start_s"] == pytest.approx(1.5)
    assert sub["objectives"]["latency_s"]["mean"] == pytest.approx(0.1)


MULTI_QUALITY = (
    "def exact(predicted, expected):\n"
    "    return 1.0 if predicted == expected else 0.0\n"
    "def half(predicted, expected):\n"
    "    return 0.5\n"
    "METRICS = {'exact': exact, 'half': half}\n"
)


def priced_run(cost, latency):
    """A run_fn that reports a fixed cost and latency (deterministic objectives)."""
    def _run(inputs):
        return SimpleNamespace(
            ok=True, outputs={"Loud": inputs["text"].upper() + "!"},
            error=None, cost_dollars=cost, duration_s=latency,
        )
    return _run


def _multi_ws(tmp_path, name="shout_ap", primary="exact"):
    workspace = FakeWorkspace(tmp_path / name, SHOUT_SCHEMA)
    metric.write_metric(workspace, MULTI_QUALITY)
    metric.approve(workspace, "tester", primary=primary)
    BudgetLedger.start(workspace.budget_json, Budget(eval_calls=1000))
    return workspace


def test_evaluate_scores_all_objectives_and_mirrors_primary(tmp_path, monkeypatch):
    workspace = _multi_ws(tmp_path)
    install_fakes(monkeypatch, candidate=make_candidate(), run_fn=priced_run(0.25, 0.1))
    rep = scoring.evaluate(workspace, "candidate_0", split="val")

    # primary (exact) mirrored to the top level, exactly today's shape
    assert rep.primary == "exact"
    assert rep.mean == 1.0
    # every objective present, including the primary and the two cost objectives
    assert set(rep.objectives) == {"exact", "half", "cost_dollars", "latency_s"}
    assert rep.objectives["exact"]["mean"] == 1.0
    assert rep.objectives["half"]["mean"] == 0.5
    assert rep.objectives["cost_dollars"]["mean"] == pytest.approx(0.25)
    assert rep.objectives["latency_s"]["mean"] == pytest.approx(0.1)

    stored = json.loads(workspace.scores_json.read_text())
    assert stored["primary"] == "exact"
    assert stored["objectives"] == {
        "exact": "max", "half": "max", "cost_dollars": "min", "latency_s": "min",
    }
    val = stored["candidates"]["candidate_0"]["val"]
    assert val["mean"] == 1.0  # primary mirror at the top level
    assert set(val["objectives"]) == {"exact", "half", "cost_dollars", "latency_s"}
    assert val["objectives"]["cost_dollars"]["mean"] == pytest.approx(0.25)

    text = str(rep)
    assert "[primary: exact]" in text
    assert "half:" in text and "(max)" in text
    assert "cost_dollars:" in text and "(min)" in text

    # the outputs cache is written, carrying per-repeat cost + latency
    cache = scoring.cache_path(workspace, "candidate_0", "val")
    assert cache.exists()
    cached = json.loads(cache.read_text())
    assert cached["rows"]["row_0"][0]["cost_dollars"] == pytest.approx(0.25)
    assert cached["rows"]["row_0"][0]["latency_s"] == pytest.approx(0.1)


def test_unknown_cost_is_conservative_and_strict_json_safe(tmp_path, monkeypatch):
    workspace = _multi_ws(tmp_path)
    install_fakes(monkeypatch, candidate=make_candidate(), run_fn=priced_run(None, 0.1))
    rep = scoring.evaluate(workspace, "candidate_0", split="val")
    assert rep.objectives["cost_dollars"]["mean"] == scoring.UNKNOWN_COST
    assert "unknown" in str(rep)
    # scores.json is consumable by strict JSON implementations (no Infinity/NaN).
    json.dumps(json.loads(workspace.scores_json.read_text()), allow_nan=False)


def test_rescore_from_cache_charges_no_budget(tmp_path, monkeypatch):
    workspace = _multi_ws(tmp_path)
    install_fakes(monkeypatch, candidate=make_candidate(), run_fn=priced_run(0.25, 0.1))
    scoring.evaluate(workspace, "candidate_0", split="val")

    spent = BudgetLedger(workspace.budget_json).spent["eval_calls"]
    assert scoring.rescore(workspace, "candidate_0", "val") is True
    assert BudgetLedger(workspace.budget_json).spent["eval_calls"] == spent  # no runs

    # a candidate that was never evaluated has no cache -> caller must re-run
    assert scoring.rescore(workspace, "candidate_9", "val") is False


def test_metric_edit_rescore_from_cache_without_archiving(tmp_path, monkeypatch):
    workspace = _multi_ws(tmp_path)
    install_fakes(monkeypatch, candidate=make_candidate(), run_fn=priced_run(0.25, 0.1))
    scoring.evaluate(workspace, "candidate_0", split="val")
    assert json.loads(workspace.scores_json.read_text())[
        "candidates"]["candidate_0"]["val"]["mean"] == 1.0

    # Edit the metric in place: now everything scores 0.5. The candidate's code
    # is unchanged, so its cached outputs re-score for free — no archive/warn.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metric.write_metric(workspace, "def metric(p, e):\n    return 0.5\n")
    assert not any("never comparable" in str(w.message) for w in caught)
    assert not (workspace.root / "scores.archive").exists()

    scores = scoring.load_scores(workspace)
    val = scores["candidates"]["candidate_0"]["val"]
    assert val["mean"] == 0.5  # re-scored under the new metric from cache
    assert val["objectives"]["quality"]["mean"] == 0.5
    # cost/latency survived untouched (they are metric-independent)
    assert val["objectives"]["cost_dollars"]["mean"] == pytest.approx(0.25)
    assert val["objectives"]["latency_s"]["mean"] == pytest.approx(0.1)
    assert scores["primary"] == "quality"


def test_primary_switch_remirrors_top_level_without_reruns(tmp_path, monkeypatch):
    # Switching the primary metric is config, not code: the top-level mirror
    # must follow to the new primary's stored numbers, or finalize() would
    # compare an old-primary val mean against a new-primary test mean and
    # fabricate an overfit demotion (and it must not re-run or wipe anything).
    workspace = _multi_ws(tmp_path, primary="exact")
    install_fakes(monkeypatch, candidate=make_candidate(), run_fn=priced_run(0.25, 0.1))
    scoring.evaluate(workspace, "candidate_0", split="val")
    assert json.loads(workspace.scores_json.read_text())[
        "candidates"]["candidate_0"]["val"]["mean"] == 1.0

    metric.approve(workspace, "tester", primary="half")   # re-approve, new primary
    spent = BudgetLedger(workspace.budget_json).spent["eval_calls"]
    scores = scoring.load_scores(workspace)               # reconcile_config runs here

    val = scores["candidates"]["candidate_0"]["val"]
    assert scores["primary"] == "half"
    assert val["mean"] == 0.5                             # mirror now follows 'half'
    assert val["objectives"]["exact"]["mean"] == 1.0      # both objectives retained
    assert val["objectives"]["half"]["mean"] == 0.5
    assert BudgetLedger(workspace.budget_json).spent["eval_calls"] == spent
    assert not (workspace.root / "scores.archive").exists()  # config change never wipes


def test_compare_on_cost_objective_respects_min_goal(tmp_path, monkeypatch):
    # A more-expensive challenger must NOT read as "improved" on a min objective.
    workspace = _multi_ws(tmp_path)
    install_fakes(monkeypatch, candidate=make_candidate(name="candidate_0"),
                  run_fn=priced_run(0.10, 0.1))
    scoring.evaluate(workspace, "candidate_0", split="val")
    install_fakes(monkeypatch, candidate=make_candidate(name="candidate_1"),
                  run_fn=priced_run(0.20, 0.1))
    scoring.evaluate(workspace, "candidate_1", split="val")

    worse = scoring.compare(workspace, "candidate_0", "candidate_1", objective="cost_dollars")
    assert worse.goal == "min"
    assert worse.diff_mean > 0            # candidate_1 - candidate_0 = +0.10 (costlier)
    assert worse.improved is False        # costlier is not an improvement
    better = scoring.compare(workspace, "candidate_1", "candidate_0", objective="cost_dollars")
    assert better.diff_mean < 0
    assert better.improved is True        # the cheaper challenger wins
    assert "lower is better" in str(better)


def test_per_field_counts_failed_repeats_as_zero(tmp_path, monkeypatch):
    # Multi-output: a failed run scores 0 in the headline mean; the per-field
    # breakdown must count it as 0 too, not average over successes only.
    workspace = FakeWorkspace(tmp_path / "qa_ap", QA_SCHEMA)
    metric.write_metric(workspace, MULTI_METRIC)
    metric.approve(workspace, "tester")
    BudgetLedger.start(workspace.budget_json, Budget(eval_calls=1000))

    splits = {"val": [
        {"question": "q1", "Answer": "a1", "Confidence": "1.0"},
        {"question": "q2", "Answer": "a2", "Confidence": "1.0"},
    ]}

    def run_fn(inputs):
        if inputs["question"] == "q2":
            return SimpleNamespace(ok=False, outputs=None, error="boom",
                                   cost_dollars=None, duration_s=0.0)
        return SimpleNamespace(ok=True, outputs={"Answer": "a1", "Confidence": 1.0},
                               error=None, cost_dollars=None, duration_s=0.0)

    install_fakes(monkeypatch, splits=splits, candidate=make_candidate(), run_fn=run_fn)
    rep = scoring.evaluate(workspace, "candidate_0", split="val")
    assert rep.mean == pytest.approx(0.5)          # one perfect row, one failed (0)
    assert rep.per_field["Answer"] == pytest.approx(0.5)
    assert rep.per_field["Confidence"] == pytest.approx(0.5)


# ------------------------------------------------------- pareto_nondominated


def test_pareto_nondominated_quality_cost_frontier():
    points = {
        "a": {"quality": 0.9, "cost_dollars": 0.20},
        "b": {"quality": 0.8, "cost_dollars": 0.01},
        "c": {"quality": 0.7, "cost_dollars": 0.30},  # dominated by both a and b
    }
    goals = {"quality": "max", "cost_dollars": "min"}
    assert scoring.pareto_nondominated(points, goals) == ["a", "b"]  # insertion order


def test_pareto_nondominated_cost_tie_broken_by_latency():
    points = {
        "slow": {"cost_dollars": 0.1, "latency_s": 0.50},
        "fast": {"cost_dollars": 0.1, "latency_s": 0.20},  # same cost, faster
    }
    goals = {"cost_dollars": "min", "latency_s": "min"}
    assert scoring.pareto_nondominated(points, goals) == ["fast"]


def test_pareto_nondominated_full_tie_keeps_all():
    points = {"a": {"quality": 0.5}, "b": {"quality": 0.5}}
    assert scoring.pareto_nondominated(points, {"quality": "max"}) == ["a", "b"]


# --------------------------------------------------------------- tradeoffs


def _obj(mean):
    return {"mean": mean, "std": 0.0, "ci95": [mean, mean], "per_field": None}


def test_tradeoffs_frontier_over_stored_objectives(tmp_path):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    metric.write_metric(workspace, EXACT)
    metric.approve(workspace, "tester")
    scores = scoring.load_scores(workspace)
    scores["objectives"] = {"quality": "max", "cost_dollars": "min", "latency_s": "min"}
    scores["primary"] = "quality"
    scores["candidates"] = {
        "candidate_0": {"val": {"objectives": {
            "quality": _obj(0.9), "cost_dollars": _obj(0.20), "latency_s": _obj(0.1)}}},
        "candidate_1": {"val": {"objectives": {
            "quality": _obj(0.8), "cost_dollars": _obj(0.01), "latency_s": _obj(0.1)}}},
        "candidate_2": {"val": {"objectives": {
            "quality": _obj(0.5), "cost_dollars": _obj(0.50), "latency_s": _obj(0.3)}}},
    }
    scoring.save_scores(workspace, scores)

    rep = scoring.tradeoffs(workspace)
    assert isinstance(rep, scoring.TradeoffReport)
    assert set(rep.nondominated) == {"candidate_0", "candidate_1"}
    # rows sorted by primary (quality) descending
    assert [r["candidate"] for r in rep.rows] == [
        "candidate_0", "candidate_1", "candidate_2"]
    assert rep.rows[0]["dominated"] is False
    assert rep.rows[-1]["dominated"] is True
    text = str(rep)
    assert "quality / cost tradeoffs on val" in text
    assert "frontier" in text


def test_tradeoffs_empty_when_no_objectives(tmp_path):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    metric.write_metric(workspace, EXACT)
    metric.approve(workspace, "tester")
    rep = scoring.tradeoffs(workspace)
    assert rep.rows == []
    assert rep.nondominated == []
    assert "eval some" in str(rep)


# ------------------------------------------------------- compare(objective=...)


def test_compare_on_named_objective(tmp_path):
    workspace = FakeWorkspace(tmp_path / "shout_ap", SHOUT_SCHEMA)
    metric.write_metric(workspace, EXACT)
    metric.approve(workspace, "tester")
    scores = scoring.load_scores(workspace)

    def sub(quality_rows, cost_rows):
        return {
            "rows": quality_rows,
            "mean": sum(quality_rows.values()) / len(quality_rows),
            "std": 0.0, "ci95": [0.0, 1.0], "n_repeats": 1,
            "repeat_variance": 0.0, "per_field": None,
            "objectives": {
                "quality": {"rows": quality_rows},
                "cost_dollars": {"rows": cost_rows},
            },
        }

    scores["candidates"] = {
        "candidate_0": {"val": sub({"row_0": 1.0, "row_1": 1.0},
                                   {"row_0": 0.5, "row_1": 0.5})},
        "candidate_1": {"val": sub({"row_0": 1.0, "row_1": 1.0},
                                   {"row_0": 0.2, "row_1": 0.2})},
    }
    scoring.save_scores(workspace, scores)

    # the default path (objective=None) is the primary metric, unchanged
    prim = scoring.compare(workspace, "candidate_0", "candidate_1")
    assert prim.objective is None
    assert prim.diff_mean == pytest.approx(0.0)

    # comparing on cost_dollars pairs the stored per-row cost scores
    cost = scoring.compare(workspace, "candidate_0", "candidate_1", objective="cost_dollars")
    assert cost.objective == "cost_dollars"
    assert cost.diff_mean == pytest.approx(-0.3)  # b(0.2) - a(0.5) per row
    assert "[cost_dollars]" in str(cost)

    # the refusal names the objective when a candidate lacks its per-row scores
    with pytest.raises(DataDisciplineError, match="cost_dollars"):
        scoring.compare(workspace, "candidate_0", "candidate_9", objective="cost_dollars")


# ------------------------------------------------------------- determinism


def test_evaluate_objectives_deterministic_across_runs(tmp_path, monkeypatch):
    def run_once(name):
        workspace = _multi_ws(tmp_path, name=name)
        install_fakes(monkeypatch, candidate=make_candidate(), run_fn=priced_run(0.25, 0.1))
        rep = scoring.evaluate(workspace, "candidate_0", split="val")
        return rep, scoring.tradeoffs(workspace)

    rep_a, tr_a = run_once("run_a")
    rep_b, tr_b = run_once("run_b")
    assert rep_a.objectives == rep_b.objectives
    assert str(rep_a) == str(rep_b)
    assert tr_a.nondominated == tr_b.nondominated
    assert [r["objectives"] for r in tr_a.rows] == [r["objectives"] for r in tr_b.rows]
