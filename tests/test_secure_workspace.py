"""Controller-private splits and immutable score/schema provenance."""

from __future__ import annotations

import json

import pytest

from autoprogramming import Budget
from autoprogramming import metric
from autoprogramming.budget import BudgetLedger
from autoprogramming.candidates import bundle_sha, new_candidate
from autoprogramming.data import load_split
from autoprogramming.errors import DataDisciplineError, WorkspaceError
from autoprogramming.harness import AgentHarness
from autoprogramming.schema import Schema
from autoprogramming.workspace import Workspace


class Label(str):
    pass


def classify(text: str) -> Label:
    """Return a label."""


def make_secure(tmp_path, monkeypatch):
    monkeypatch.setenv("AP_PRIVATE_DATA_DIR", str(tmp_path / "private"))
    rows = {
        "train": [{"text": f"t{i}", "Label": f"l{i}"} for i in range(5)],
        "val": [{"text": "v", "Label": "secret-val"}],
        "test": [{"text": "z", "Label": "secret-test"}],
    }
    ws = Workspace.create(
        tmp_path / "classify_ap",
        Schema.from_function(classify),
        rows,
        seed=0,
        ratios=(0.6, 0.2, 0.2),
        data_sha="secure",
        bootstrap=True,
        secure_data=True,
    )
    BudgetLedger.start(ws.budget_json, Budget(eval_calls=100))
    metric.write_metric(ws, "def metric(p, e):\n    return float(p == e)\n")
    metric.approve(ws, "tester")
    return ws


def test_legacy_workspace_can_atomically_move_splits_private(tmp_path, monkeypatch):
    monkeypatch.setenv("AP_PRIVATE_DATA_DIR", str(tmp_path / "private"))
    rows = {split: [{"text": split, "Label": split}] for split in ("train", "val", "test")}
    ws = Workspace.create(
        tmp_path / "legacy_ap", Schema.from_function(classify), rows,
        seed=0, ratios=(0.6, 0.2, 0.2), data_sha="legacy", bootstrap=True,
    )
    assert (ws.data_dir / "val.csv").exists()
    ws.secure_splits()
    assert not (ws.data_dir / "val.csv").exists()
    assert load_split(ws, "val")[0]["Label"] == "val"
    first_private = ws.private_data_dir
    ws.secure_splits()  # idempotent
    assert ws.private_data_dir == first_private


def test_secure_workspace_has_no_val_or_test_files(tmp_path, monkeypatch):
    ws = make_secure(tmp_path, monkeypatch)
    assert (ws.data_dir / "train.csv").exists()
    assert not (ws.data_dir / "val.csv").exists()
    assert not (ws.data_dir / "test.csv").exists()
    assert ws.private_data_dir.parent == tmp_path / "private"
    assert load_split(ws, "val")[0]["Label"] == "secret-val"
    assert load_split(ws, "test")[0]["Label"] == "secret-test"


def test_naive_candidate_cannot_read_hidden_split_from_workspace(tmp_path, monkeypatch):
    ws = make_secure(tmp_path, monkeypatch)
    source = '''# /// script
# [tool.ap]
# deterministic = true
# ///
import csv, os
from pathlib import Path

def predict(text):
    root = Path(os.environ["AP_WORKSPACE"])
    for split in ("val", "test"):
        with (root / "data" / f"{split}.csv").open() as fh:
            for row in csv.DictReader(fh):
                if row["text"] == text:
                    return row["Label"]
    return "unknown"
'''
    h = AgentHarness(ws)
    h.new_candidate(source=source)
    report = h.eval("candidate_0", n_repeats=1)
    assert report.mean == 0.0
    assert report.errors


def test_candidate_edit_makes_val_score_stale(tmp_path, monkeypatch):
    ws = make_secure(tmp_path, monkeypatch)
    h = AgentHarness(ws)
    cand = h.new_candidate(source=(
        "# /// script\n# [tool.ap]\n# deterministic = true\n# cost_per_call = 0.0\n# ///\n"
        "def predict(text):\n    return 'wrong'\n"
    ))
    h.eval(cand.name, n_repeats=1)
    cand.path.write_text("def predict(text):\n    return 'changed'\n")
    with pytest.raises(DataDisciplineError, match="stale"):
        h.compare(cand.name, cand.name)
    with pytest.raises(Exception, match="stale|no candidate|no stored"):
        h.finalize(top_k=1)


def test_candidate_bundle_hash_pins_declared_artifacts(tmp_path, monkeypatch):
    ws = make_secure(tmp_path, monkeypatch)
    artifact_dir = ws.artifacts_dir / "rules-candidate_0"
    artifact_dir.mkdir()
    model = artifact_dir / "model.bin"
    model.write_bytes(b"one")
    cand = new_candidate(ws, source=(
        "# /// script\n# [tool.ap]\n"
        "# artifact_namespace = \"rules-candidate_0\"\n# ///\n"
        "def predict(text):\n    return text\n"
    ))
    before = bundle_sha(ws, cand)
    model.write_bytes(b"two")
    assert bundle_sha(ws, cand) != before


def test_schema_edit_is_detected_after_reload(tmp_path, monkeypatch):
    ws = make_secure(tmp_path, monkeypatch)
    ws.schema_py.write_text(ws.schema_py.read_text() + "\n# edited\n")
    loaded = Workspace.load(ws.root)
    with pytest.raises(WorkspaceError, match="changed"):
        _ = loaded.schema
