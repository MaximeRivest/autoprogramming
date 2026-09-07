"""End-to-end offline loop: define -> optimize (scripted backend) -> finalize ->
ship -> log -> review -> re-optimize from reviewed logs.

No network, no API keys, no uv: every candidate is stdlib-only (runner fast
path) and the metric is auto-approved via AP_AUTO_APPROVE_METRIC. The whole
pipeline runs once in a module-scoped fixture; the test functions are
assertions over its recorded state.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
import tomllib
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import autoprogramming as ap
from autoprogramming import (
    BudgetError,
    BudgetExceededError,
    DataDisciplineError,
    FinalizedError,
)
from autoprogramming import data as data_mod
from autoprogramming import logs as logs_mod
from autoprogramming.budget import BudgetLedger
from autoprogramming.harness import FinalReport

PKG = "shout_e2e_ap"

METRIC_SRC = (
    "def metric(predicted, expected):\n"
    "    return 1.0 if predicted == expected else 0.0\n"
)

_DETERMINISTIC_BLOCK = (
    "# /// script\n"
    "# [tool.ap]\n"
    "# deterministic = true\n"
    "# ///\n"
)
IDENTITY_SRC = _DETERMINISTIC_BLOCK + "def predict(text):\n    return text\n"
UPPER_SRC = _DETERMINISTIC_BLOCK + 'def predict(text):\n    return text.upper() + "!"\n'


class Loud(str):
    """The input, uppercased, with an exclamation mark."""


def shout(text: str) -> Loud:
    """Uppercase the text."""


def _rows(n: int) -> list[dict]:
    return [
        {"text": f"hello world {i}", "Loud": f"HELLO WORLD {i}!"}
        for i in range(n)
    ]


class ScriptedBackend:
    """A fake coding agent: metric sign-off, two candidates, evals, trace,
    frontier, compare — everything the README's loop does, scripted."""

    def __init__(self):
        self.context = None
        self.events = {}

    def run(self, harness, context):
        self.context = dict(context)
        ev = self.events
        ev["metric_ok"] = harness.propose_metric(
            METRIC_SRC,
            examples=[("HI!", "HI!"), ("hi", "HI!")],
            note="exact match",
        )
        c0 = harness.new_candidate(source=IDENTITY_SRC)
        c1 = harness.new_candidate(source=UPPER_SRC)
        ev["names"] = (c0.name, c1.name)
        ev["train0"] = harness.eval(c0.name, split="train", per_instance=True)
        ev["train1"] = harness.eval(c1.name, split="train", per_instance=True)
        ev["val0"] = harness.eval(c0.name)
        ev["val1"] = harness.eval(c1.name)
        ev["trace"] = harness.run(c1.name, split="train", row=0)
        ev["frontier"] = harness.frontier()
        ev["compare"] = harness.compare(c0.name, c1.name)
        ev["budget_seen"] = harness.budget
        ev["train_len"] = len(harness.data.train)
        ev["val_len"] = len(harness.data.val)


class NullBackend:
    """A backend that does nothing (workspace creation checks only)."""

    def run(self, harness, context):
        return None


class BudgetTrippingBackend:
    """Evals until the eval_calls budget trips mid-eval on train."""

    def __init__(self):
        self.tripped_message = None

    def run(self, harness, context):
        harness.propose_metric(METRIC_SRC, examples=[("A", "A")])
        cand = harness.new_candidate(source=UPPER_SRC)
        with pytest.raises(BudgetExceededError) as exc_info:
            harness.eval(cand.name, split="train", per_instance=True)
        self.tripped_message = str(exc_info.value)


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    """Run the full offline loop once; return everything the tests inspect."""
    mp = pytest.MonkeyPatch()
    mp.setenv("AP_AUTO_APPROVE_METRIC", "1")
    base = tmp_path_factory.mktemp("e2e")
    ws_root = base / PKG

    prog = ap.program(shout)
    rows = _rows(40)
    backend = ScriptedBackend()
    report = prog.optimize(
        rows,
        budget=ap.Budget(eval_calls=500, dollars=50),
        workspace=str(ws_root),
        seed=0,
        backend=backend,
    )
    ws = prog.workspace
    harness = ap.attach(str(ws_root))
    spent_after_finalize = BudgetLedger(ws.budget_json).spent

    # --- ship it: import the workspace as a plain package -------------------
    sys.path.insert(0, str(base))
    pkg = importlib.import_module(PKG)
    shipped_out = pkg.shout("please stop")

    pkg.shout.enable_logging()
    logged_out = pkg.shout(text="turn it up")
    log_file = ws_root / "logs" / (
        datetime.now(timezone.utc).strftime("%Y-%m-%d") + ".jsonl"
    )
    log_lines = [json.loads(l) for l in log_file.read_text().splitlines()]
    for i in range(7):
        pkg.shout(f"review entry {i}")
    pkg.shout.disable_logging()

    # --- review the traffic with a scripted input_fn ------------------------
    responses = iter(["c", "FIXED!", "r"] + ["a"] * 20)
    printed: list[str] = []
    review_counts = logs_mod.review_logs(
        ws, input_fn=lambda prompt: next(responses), print_fn=printed.append
    )
    reviewed_entries = logs_mod.read_reviewed(ws)
    reviewed_rows = logs_mod.logs_to_rows(reviewed_entries, prog.schema)

    # --- re-optimize from the reviewed logs into a fresh workspace ----------
    reviewed_root = base / "shout_reviewed_ap"
    reviewed_report = prog.optimize(
        "logs:reviewed",
        budget=ap.Budget(eval_calls=10),
        workspace=str(reviewed_root),
        backend=NullBackend(),
    )

    # --- budget accounting: trip eval_calls mid-eval -------------------------
    budget_root = base / "shout_budget_ap"
    trip_backend = BudgetTrippingBackend()
    budget_prog = ap.program(shout)
    budget_report = budget_prog.optimize(
        _rows(12),
        budget=ap.Budget(eval_calls=5),
        workspace=str(budget_root),
        backend=trip_backend,
    )

    ns = SimpleNamespace(
        base=base,
        ws=ws,
        ws_root=ws_root,
        prog=prog,
        report=report,
        backend=backend,
        harness=harness,
        spent=spent_after_finalize,
        pkg=pkg,
        shipped_out=shipped_out,
        logged_out=logged_out,
        log_lines=log_lines,
        review_counts=review_counts,
        reviewed_entries=reviewed_entries,
        reviewed_rows=reviewed_rows,
        reviewed_root=reviewed_root,
        reviewed_report=reviewed_report,
        budget_root=budget_root,
        trip_backend=trip_backend,
        budget_report=budget_report,
    )
    yield ns
    for name in [m for m in sys.modules if m == PKG or m.startswith(PKG + ".")]:
        del sys.modules[name]
    sys.path.remove(str(base))
    mp.undo()


# --------------------------------------------------------------- optimization


def test_backend_ran_with_optimize_context(e2e):
    assert e2e.backend.context == {"mode": "optimize"}
    assert e2e.backend.events["metric_ok"] is True
    assert e2e.backend.events["names"] == ("candidate_0", "candidate_1")


def test_split_sizes_and_agent_data_view(e2e):
    counts = json.loads(e2e.ws.split_json.read_text())["counts"]
    assert counts == {"train": 24, "val": 8, "test": 8}
    assert e2e.backend.events["train_len"] == 24
    assert e2e.backend.events["val_len"] == 8
    assert not hasattr(e2e.harness.data, "test")


def test_train_eval_is_per_row_val_is_aggregate_only(e2e):
    ev = e2e.backend.events
    train1 = ev["train1"]
    assert train1.split == "train"
    assert train1.per_row is not None and len(train1.per_row) == 24
    assert train1.mean == pytest.approx(1.0)
    assert ev["train0"].mean == pytest.approx(0.0)
    val1 = ev["val1"]
    assert val1.split == "val"
    assert val1.per_row is None
    assert val1.mean == pytest.approx(1.0)
    assert val1.n_repeats == 1  # [tool.ap] deterministic = true
    assert ev["val0"].mean == pytest.approx(0.0)


def test_trace_frontier_compare(e2e):
    ev = e2e.backend.events
    trace = ev["trace"]
    assert trace.result.ok
    assert trace.score == pytest.approx(1.0)
    assert "score: 1.000" in str(trace)
    frontier = ev["frontier"]
    assert frontier.nondominated == ["candidate_1"]
    assert frontier.missing == []
    assert len(frontier.rows) == 24
    cmp = ev["compare"]
    assert cmp.improved
    assert cmp.diff_mean == pytest.approx(1.0)


def test_scores_json_holds_per_row_val_scores(e2e):
    scores = json.loads(e2e.ws.scores_json.read_text())
    val_rows = scores["candidates"]["candidate_1"]["val"]["rows"]
    assert len(val_rows) == 8  # stored for the harness, never shown per-row
    assert scores["val_scored"] == ["candidate_0", "candidate_1"]
    assert scores["flags"] == {}


# ------------------------------------------------------------------- finalize


def test_final_report_mirrors_readme(e2e):
    report = e2e.report
    assert isinstance(report, FinalReport)
    assert report.activated == "candidate_1"
    assert report.val_reliability == "ok"
    text = str(report)
    assert "test scores (evaluated once):" in text
    assert "candidate_1: 1.00" in text
    assert "(val was 1.00 — healthy gap)" in text
    assert "activated: candidate_1" in text


def test_winner_activated_and_test_evaluated_once(e2e):
    active = e2e.ws.active
    assert active["active"] == "candidate_1"
    assert active["finalized"] is True
    assert active["test_score"] == pytest.approx(1.0)
    assert active["metric_sha"]
    # 24+24 train, 8+8 val, 1 trace, then exactly one 8-row test pass per
    # finalist (2 finalists, 1 repeat each — deterministic): 81 total.
    assert e2e.spent["eval_calls"] == 81
    assert e2e.spent["dollars"] == pytest.approx(0.0)
    report_on_disk = json.loads(e2e.ws.final_report.read_text())
    assert report_on_disk["activated"] == "candidate_1"


def test_generated_pyproject_parses_and_maps_package(e2e):
    doc = tomllib.loads(e2e.ws.pyproject.read_text())
    assert doc["project"]["name"] == PKG.replace("_", "-")
    assert doc["project"]["dependencies"] == []
    assert doc["project"]["requires-python"] == ">=3.11"
    assert doc["build-system"]["build-backend"] == "setuptools.build_meta"
    assert doc["tool"]["setuptools"]["packages"] == [PKG]
    assert doc["tool"]["setuptools"]["package-dir"] == {PKG: "."}
    package_data = doc["tool"]["setuptools"]["package-data"][PKG]
    assert "active.json" in package_data and "candidates/*.py" in package_data


# ------------------------------------------------------- the shipped package


def test_shipped_package_returns_typed_output(e2e):
    pkg = e2e.pkg
    assert e2e.shipped_out == "PLEASE STOP!"
    assert isinstance(e2e.shipped_out, pkg.schema.Loud)
    assert isinstance(e2e.shipped_out, str)
    assert pkg.schema.Loud.__doc__ == "The input, uppercased, with an exclamation mark."
    assert set(pkg.__all__) == {"shout", "Loud"}
    with pytest.raises(TypeError):
        pkg.shout()
    with pytest.raises(TypeError):
        pkg.shout("x", nope="y")


def test_shipped_logging_writes_readme_jsonl(e2e):
    assert e2e.logged_out == "TURN IT UP!"
    entry = e2e.log_lines[0]
    assert list(entry) == ["inputs", "outputs", "candidate", "n_repeat", "timestamp"]
    assert entry["inputs"] == {"text": "turn it up"}
    assert entry["outputs"] == {"Loud": "TURN IT UP!"}
    assert entry["candidate"] == "candidate_1"
    assert entry["n_repeat"] == 1
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", entry["timestamp"])


# ----------------------------------------------------------------- guard rails


def test_val_rows_are_not_readable(e2e):
    prg = e2e.harness
    assert len(prg.data.val) == 8
    with pytest.raises(DataDisciplineError):
        prg.data.val[0]
    with pytest.raises(DataDisciplineError):
        list(prg.data.val)


def test_eval_on_test_is_refused(e2e):
    with pytest.raises(DataDisciplineError, match="finalize"):
        e2e.harness.eval("candidate_1", split="test")


def test_trace_on_val_is_refused(e2e):
    with pytest.raises(DataDisciplineError, match="train"):
        e2e.harness.run("candidate_1", split="val")


def test_per_instance_val_is_refused(e2e):
    with pytest.raises(DataDisciplineError, match="aggregate"):
        e2e.harness.eval("candidate_1", per_instance=True)


def test_second_finalize_is_refused(e2e):
    with pytest.raises(FinalizedError, match="final_report.json"):
        e2e.harness.finalize()


def test_optimize_on_raw_logs_is_refused(e2e):
    with pytest.raises(DataDisciplineError) as exc_info:
        e2e.prog.optimize("logs", budget=ap.Budget(eval_calls=1))
    message = str(exc_info.value)
    assert "review_logs" in message
    assert "logs:reviewed" in message


def test_budget_requires_at_least_one_limit():
    with pytest.raises(BudgetError):
        ap.Budget()


# ------------------------------------------------------------ budget accounting


def test_budget_trips_mid_eval(e2e):
    assert e2e.trip_backend.tripped_message is not None
    assert "eval_calls" in e2e.trip_backend.tripped_message
    spent = BudgetLedger(e2e.budget_root / "budget.json").spent
    assert spent["eval_calls"] == 5  # charged up to the limit, then refused
    scores = json.loads((e2e.budget_root / "scores.json").read_text())
    assert scores["candidates"] == {}  # nothing persisted for the partial eval
    assert e2e.budget_report is None  # nothing val-scored -> no finalize


# ------------------------------------------------------------------ log review


def test_review_counts_and_reviewed_rows(e2e):
    # 8 logged calls total (1 format check + 7 corpus), sample defaults to 50
    assert e2e.review_counts == {
        "reviewed": 8, "accepted": 6, "corrected": 1, "rejected": 1,
    }
    assert len(e2e.reviewed_entries) == 7  # rejected entries carry no target
    assert all(e["verdict"] in ("accept", "corrected") for e in e2e.reviewed_entries)
    assert all(set(r) == {"text", "Loud"} for r in e2e.reviewed_rows)
    assert sum(r["Loud"] == "FIXED!" for r in e2e.reviewed_rows) == 1


def test_reviewed_rows_build_a_fresh_workspace(e2e):
    assert e2e.reviewed_report is None  # NullBackend scored nothing
    counts = json.loads(
        (e2e.reviewed_root / "data" / "split.json").read_text()
    )["counts"]
    assert sum(counts.values()) == 7
    all_rows = [
        row
        for split in ("train", "val", "test")
        for row in data_mod.read_csv(e2e.reviewed_root / "data" / f"{split}.csv")
    ]
    assert sum(r["Loud"] == "FIXED!" for r in all_rows) == 1
