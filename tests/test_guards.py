"""Unit tests for guards.py — standalone: only errors.py and schema.py needed."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from autoprogramming import guards
from autoprogramming.errors import (
    BootstrapModeError,
    DataDisciplineError,
    ValReliabilityWarning,
    WorkspaceError,
)
from autoprogramming.schema import Schema


class Loud(str):
    """The input, uppercased, with an exclamation mark."""


def shout(text: str) -> Loud:
    """Uppercase the text."""


class Confidence(float):
    """A calibrated probability, 0.0-1.0."""


def rate(text: str) -> Confidence:
    """Rate the text."""


SHOUT_SCHEMA = Schema.from_function(shout)
RATE_SCHEMA = Schema.from_function(rate)


class FakeWorkspace:
    """Duck-typed workspace: guards only touches scores_json and split_json."""

    def __init__(self, root: Path, bootstrap: bool = False, val_count: int = 50):
        self.root = root
        (root / "data").mkdir(parents=True, exist_ok=True)
        self.scores_json = root / "scores.json"
        self.split_json = root / "data" / "split.json"
        self.split_json.write_text(json.dumps({
            "seed": 0,
            "ratios": [0.6, 0.2, 0.2],
            "counts": {"train": 6, "val": val_count, "test": 3},
            "data_sha": "0" * 64,
            "bootstrap": bootstrap,
        }))


def register_quietly(ws, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        guards.register_val_candidate(ws, name)


# ---------------------------------------------------------------- constants


def test_constants():
    assert guards.BOOTSTRAP_MIN == 30
    assert guards.BOOTSTRAP_MAX_VAL_CANDIDATES == 5


# ---------------------------------------------------------- eval permission


def test_eval_allowed_on_train_and_val():
    guards.assert_eval_allowed("train", False)
    guards.assert_eval_allowed("train", True)
    guards.assert_eval_allowed("val", False)


def test_eval_refused_on_test():
    with pytest.raises(DataDisciplineError) as exc:
        guards.assert_eval_allowed("test", False)
    msg = str(exc.value)
    assert "finalize()" in msg
    assert "once" in msg


def test_eval_refused_per_instance_on_val():
    with pytest.raises(DataDisciplineError) as exc:
        guards.assert_eval_allowed("val", True)
    msg = str(exc.value)
    assert "aggregate" in msg
    assert "train" in msg


def test_eval_refused_on_unknown_split():
    with pytest.raises(DataDisciplineError) as exc:
        guards.assert_eval_allowed("holdout", False)
    assert "holdout" in str(exc.value)


# ---------------------------------------------------------------- tracing


def test_trace_allowed_on_train_only():
    guards.assert_trace_allowed("train")
    for split in ("val", "test", "anything"):
        with pytest.raises(DataDisciplineError) as exc:
            guards.assert_trace_allowed(split)
        assert "train" in str(exc.value)


# ---------------------------------------------------------------- bootstrap


def test_is_bootstrap_reads_split_json(tmp_path):
    assert guards.is_bootstrap(FakeWorkspace(tmp_path / "a", bootstrap=True)) is True
    assert guards.is_bootstrap(FakeWorkspace(tmp_path / "b", bootstrap=False)) is False


def test_is_bootstrap_missing_split_json(tmp_path):
    ws = FakeWorkspace(tmp_path / "ws")
    ws.split_json.unlink()
    with pytest.raises(WorkspaceError):
        guards.is_bootstrap(ws)


# ------------------------------------------------------------- registration


def test_register_creates_scores_and_appends(tmp_path):
    ws = FakeWorkspace(tmp_path / "ws")
    register_quietly(ws, "candidate_0")
    stored = json.loads(ws.scores_json.read_text())
    assert stored["val_scored"] == ["candidate_0"]
    register_quietly(ws, "candidate_1")
    stored = json.loads(ws.scores_json.read_text())
    assert stored["val_scored"] == ["candidate_0", "candidate_1"]


def test_register_same_name_is_idempotent(tmp_path):
    ws = FakeWorkspace(tmp_path / "ws")
    register_quietly(ws, "candidate_0")
    register_quietly(ws, "candidate_0")
    stored = json.loads(ws.scores_json.read_text())
    assert stored["val_scored"] == ["candidate_0"]


def test_register_preserves_other_scores_content(tmp_path):
    ws = FakeWorkspace(tmp_path / "ws")
    ws.scores_json.write_text(json.dumps({
        "metric_sha": "abc",
        "candidates": {"candidate_0": {"val": {"mean": 0.5}}},
        "val_scored": ["candidate_0"],
        "flags": {"candidate_0": ["memorizer: x"]},
    }))
    register_quietly(ws, "candidate_1")
    stored = json.loads(ws.scores_json.read_text())
    assert stored["metric_sha"] == "abc"
    assert stored["candidates"]["candidate_0"]["val"]["mean"] == 0.5
    assert stored["flags"]["candidate_0"] == ["memorizer: x"]
    assert stored["val_scored"] == ["candidate_0", "candidate_1"]


def test_bootstrap_cap_raises_before_any_append(tmp_path):
    ws = FakeWorkspace(tmp_path / "ws", bootstrap=True)
    for i in range(5):
        register_quietly(ws, f"candidate_{i}")
    with pytest.raises(BootstrapModeError) as exc:
        register_quietly(ws, "candidate_5")
    msg = str(exc.value)
    assert "bootstrap" in msg
    assert "synthetic" in msg
    stored = json.loads(ws.scores_json.read_text())
    assert len(stored["val_scored"]) == 5
    assert "candidate_5" not in stored["val_scored"]


def test_bootstrap_cap_allows_rescoring_registered_candidates(tmp_path):
    ws = FakeWorkspace(tmp_path / "ws", bootstrap=True)
    for i in range(5):
        register_quietly(ws, f"candidate_{i}")
    register_quietly(ws, "candidate_2")


def test_no_cap_outside_bootstrap(tmp_path):
    ws = FakeWorkspace(tmp_path / "ws", bootstrap=False, val_count=100)
    for i in range(8):
        register_quietly(ws, f"candidate_{i}")
    stored = json.loads(ws.scores_json.read_text())
    assert len(stored["val_scored"]) == 8


# ---------------------------------------------------------------- pressure


def test_pressure_status_thresholds():
    assert guards.pressure_status(0, 4) == "ok"
    assert guards.pressure_status(4, 4) == "ok"
    assert guards.pressure_status(5, 4) == "warn"
    assert guards.pressure_status(12, 4) == "warn"
    assert guards.pressure_status(13, 4) == "unreliable"


def test_pressure_status_degenerate_val_size():
    assert guards.pressure_status(0, 0) == "ok"
    assert guards.pressure_status(1, 0) == "unreliable"


def test_register_warns_when_pressure_builds(tmp_path):
    ws = FakeWorkspace(tmp_path / "ws", val_count=2)
    register_quietly(ws, "candidate_0")
    register_quietly(ws, "candidate_1")
    with pytest.warns(ValReliabilityWarning):
        guards.register_val_candidate(ws, "candidate_2")


def test_register_warns_unreliable_past_3x(tmp_path):
    ws = FakeWorkspace(tmp_path / "ws", val_count=1)
    register_quietly(ws, "candidate_0")
    register_quietly(ws, "candidate_1")
    register_quietly(ws, "candidate_2")
    with pytest.warns(ValReliabilityWarning) as record:
        guards.register_val_candidate(ws, "candidate_3")
    assert any("not be reported as final" in str(w.message) for w in record)


def test_register_silent_while_pressure_ok(tmp_path):
    ws = FakeWorkspace(tmp_path / "ws", val_count=50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        guards.register_val_candidate(ws, "candidate_0")


# ------------------------------------------------------------- memorization


def make_train_rows(n=10):
    return [
        {"text": f"input sentence {i}", "Loud": f"OUTPUT SENTENCE NUMBER {i}!"}
        for i in range(n)
    ]


def test_memorization_gap_rule_flags():
    flags = guards.memorization_check("def predict(text): ...", 0.9, 0.5, [], SHOUT_SCHEMA)
    assert len(flags) == 1
    assert flags[0].startswith("memorizer:")
    assert "0.900" in flags[0]


def test_memorization_gap_rule_boundaries():
    assert guards.memorization_check("src", 0.6, 0.45, [], SHOUT_SCHEMA) == []
    assert guards.memorization_check("src", 0.7, 0.5, [], SHOUT_SCHEMA) == []
    assert guards.memorization_check("src", 0.5, 0.1, [], SHOUT_SCHEMA) == []
    assert guards.memorization_check("src", 0.4, 0.1, [], SHOUT_SCHEMA) == []


def test_memorization_verbatim_flags_lookup_table():
    rows = make_train_rows(10)
    source = "TABLE = {\n" + "".join(
        f'    "{r["text"]}": "{r["Loud"]}",\n' for r in rows[:3]
    ) + "}\n"
    flags = guards.memorization_check(source, 0.5, 0.5, rows, SHOUT_SCHEMA)
    assert len(flags) == 1
    assert "verbatim" in flags[0]


def test_memorization_verbatim_below_threshold_is_clean():
    rows = make_train_rows(10)
    source = f'X = ["{rows[0]["Loud"]}", "{rows[1]["Loud"]}"]'
    assert guards.memorization_check(source, 0.5, 0.5, rows, SHOUT_SCHEMA) == []


def test_memorization_verbatim_threshold_scales_with_train_size():
    rows = make_train_rows(40)
    source = "".join(f'"{r["Loud"]}"\n' for r in rows[:3])
    assert guards.memorization_check(source, 0.5, 0.5, rows, SHOUT_SCHEMA) == []
    source = "".join(f'"{r["Loud"]}"\n' for r in rows[:4])
    flags = guards.memorization_check(source, 0.5, 0.5, rows, SHOUT_SCHEMA)
    assert len(flags) == 1


def test_memorization_verbatim_ignores_short_outputs():
    rows = [{"text": f"input {i}", "Loud": f"HI {i}!"} for i in range(10)]
    source = "".join(f'"{r["Loud"]}"\n' for r in rows)
    assert guards.memorization_check(source, 0.5, 0.5, rows, SHOUT_SCHEMA) == []


def test_memorization_verbatim_counts_distinct_outputs_once():
    rows = [{"text": f"input {i}", "Loud": "THE SAME LONG OUTPUT!"} for i in range(10)]
    source = '"THE SAME LONG OUTPUT!"'
    assert guards.memorization_check(source, 0.5, 0.5, rows, SHOUT_SCHEMA) == []


def test_memorization_verbatim_skips_non_string_outputs():
    rows = [{"text": f"input {i}", "Confidence": f"0.12345678{i}"} for i in range(10)]
    source = "".join(f'"{r["Confidence"]}"\n' for r in rows)
    assert guards.memorization_check(source, 0.5, 0.5, rows, RATE_SCHEMA) == []


def test_memorization_both_rules_stack():
    rows = make_train_rows(10)
    source = "".join(f'"{r["Loud"]}"\n' for r in rows[:3])
    flags = guards.memorization_check(source, 0.95, 0.4, rows, SHOUT_SCHEMA)
    assert len(flags) == 2
