"""Tests for autoprogramming.logs — JSONL traffic logs and the review flow."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from autoprogramming import logs as logs_mod
from autoprogramming.errors import SchemaError
from autoprogramming.schema import Schema


class Loud(str):
    """The input, uppercased, with an exclamation mark."""


def shout(text: str) -> Loud:
    """Uppercase the text."""


SCHEMA = Schema.from_function(shout)

README_LINE = (
    '{"inputs": {"english": "Where is the train station?"}, '
    '"outputs": {"French": "Où est la gare ?"}, '
    '"candidate": "candidate_1", "n_repeat": 1, '
    '"timestamp": "2026-07-04T14:22:01Z"}'
)

TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class FakeWorkspace:
    def __init__(self, root: Path):
        self.root = root
        self.logs_dir = root / "logs"


def scripted(answers):
    """An input_fn that replays a list, then signals end-of-input."""
    iterator = iter(answers)

    def input_fn(prompt=""):
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError

    return input_fn


def add_log(ws, text, loud, candidate="candidate_1"):
    return logs_mod.append_log(
        ws, inputs={"text": text}, outputs={"Loud": loud}, candidate=candidate
    )


# ------------------------------------------------------------------ append_log


def test_append_log_creates_logs_dir_and_dated_file(tmp_path):
    ws = FakeWorkspace(tmp_path)
    path = add_log(ws, "hi", "HI!")
    assert path.parent == ws.logs_dir
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}\.jsonl", path.name)


def test_append_log_writes_readme_format(tmp_path):
    ws = FakeWorkspace(tmp_path)
    path = add_log(ws, "hi", "HI!")
    line = path.read_text(encoding="utf-8").splitlines()[0]
    entry = json.loads(line)
    assert list(entry) == ["inputs", "outputs", "candidate", "n_repeat", "timestamp"]
    assert entry["inputs"] == {"text": "hi"}
    assert entry["outputs"] == {"Loud": "HI!"}
    assert entry["candidate"] == "candidate_1"
    assert entry["n_repeat"] == 1
    assert TIMESTAMP_RE.match(entry["timestamp"])


def test_append_log_reproduces_readme_line_exactly(tmp_path):
    ws = FakeWorkspace(tmp_path)
    path = logs_mod.append_log(
        ws,
        inputs={"english": "Where is the train station?"},
        outputs={"French": "Où est la gare ?"},
        candidate="candidate_1",
    )
    line = path.read_text(encoding="utf-8").splitlines()[0]
    entry = json.loads(line)
    entry["timestamp"] = "2026-07-04T14:22:01Z"
    assert json.dumps(entry, ensure_ascii=False) == README_LINE


def test_append_log_appends_multiple_lines(tmp_path):
    ws = FakeWorkspace(tmp_path)
    add_log(ws, "one", "ONE!")
    path = add_log(ws, "two", "TWO!")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


# ------------------------------------------------------- read_logs / reviewed


def test_read_logs_round_trips_appended_entries(tmp_path):
    ws = FakeWorkspace(tmp_path)
    add_log(ws, "hi", "HI!")
    entries = logs_mod.read_logs(ws)
    assert len(entries) == 1
    assert entries[0]["inputs"] == {"text": "hi"}
    assert entries[0]["outputs"] == {"Loud": "HI!"}
    assert entries[0]["candidate"] == "candidate_1"
    assert entries[0]["n_repeat"] == 1


def test_read_logs_parses_the_exact_readme_line(tmp_path):
    ws = FakeWorkspace(tmp_path)
    ws.logs_dir.mkdir(parents=True)
    (ws.logs_dir / "2026-07-04.jsonl").write_text(README_LINE + "\n", encoding="utf-8")
    entries = logs_mod.read_logs(ws)
    assert entries == [json.loads(README_LINE)]
    assert json.dumps(entries[0], ensure_ascii=False) == README_LINE


def test_read_logs_missing_dir_is_empty(tmp_path):
    assert logs_mod.read_logs(FakeWorkspace(tmp_path)) == []


def test_read_logs_filename_order_and_reviewed_excluded(tmp_path):
    ws = FakeWorkspace(tmp_path)
    ws.logs_dir.mkdir(parents=True)
    early = {"inputs": {"text": "early"}, "outputs": {"Loud": "EARLY!"},
             "candidate": "candidate_0", "n_repeat": 1,
             "timestamp": "2026-07-01T00:00:00Z"}
    late = {"inputs": {"text": "late"}, "outputs": {"Loud": "LATE!"},
            "candidate": "candidate_0", "n_repeat": 1,
            "timestamp": "2026-07-02T00:00:00Z"}
    (ws.logs_dir / "2026-07-02.jsonl").write_text(json.dumps(late) + "\n", encoding="utf-8")
    (ws.logs_dir / "2026-07-01.jsonl").write_text(json.dumps(early) + "\n", encoding="utf-8")
    (ws.logs_dir / "reviewed.jsonl").write_text(
        json.dumps({"inputs": {}, "outputs": {}, "verdict": "accept",
                    "source_sha": "x", "reviewed_at": "2026-07-03T00:00:00Z"}) + "\n",
        encoding="utf-8",
    )
    entries = logs_mod.read_logs(ws)
    assert [e["inputs"]["text"] for e in entries] == ["early", "late"]


def test_read_reviewed_missing_file_is_empty(tmp_path):
    ws = FakeWorkspace(tmp_path)
    ws.logs_dir.mkdir(parents=True)
    assert logs_mod.read_reviewed(ws) == []


def test_read_reviewed_filters_out_rejected(tmp_path):
    ws = FakeWorkspace(tmp_path)
    ws.logs_dir.mkdir(parents=True)
    records = [
        {"inputs": {"text": "a"}, "outputs": {"Loud": "A!"}, "verdict": "accept",
         "source_sha": "s1", "reviewed_at": "2026-07-04T00:00:00Z"},
        {"inputs": {"text": "b"}, "outputs": {"Loud": "B!"}, "verdict": "corrected",
         "source_sha": "s2", "reviewed_at": "2026-07-04T00:00:00Z"},
        {"inputs": {"text": "c"}, "outputs": {"Loud": "C!"}, "verdict": "rejected",
         "source_sha": "s3", "reviewed_at": "2026-07-04T00:00:00Z"},
    ]
    (ws.logs_dir / "reviewed.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    reviewed = logs_mod.read_reviewed(ws)
    assert [r["verdict"] for r in reviewed] == ["accept", "corrected"]


# ---------------------------------------------------------------- logs_to_rows


def test_logs_to_rows_merges_inputs_and_outputs():
    entries = [{"inputs": {"text": "hi"}, "outputs": {"Loud": "HI!"}}]
    assert logs_mod.logs_to_rows(entries, SCHEMA) == [{"text": "hi", "Loud": "HI!"}]


def test_logs_to_rows_missing_columns_refused():
    entries = [{"inputs": {"english": "hi"}, "outputs": {"French": "salut"}}]
    with pytest.raises(SchemaError) as exc:
        logs_mod.logs_to_rows(entries, SCHEMA)
    msg = str(exc.value)
    assert "Loud" in msg and "text" in msg


# ----------------------------------------------------------------- review_logs


def test_review_accept(tmp_path):
    ws = FakeWorkspace(tmp_path)
    add_log(ws, "one", "ONE!")
    add_log(ws, "two", "TWO!")
    printed = []
    counts = logs_mod.review_logs(
        ws, input_fn=scripted(["a", "a"]), print_fn=printed.append
    )
    assert counts == {"reviewed": 2, "accepted": 2, "corrected": 0, "rejected": 0}
    reviewed = logs_mod.read_reviewed(ws)
    assert len(reviewed) == 2
    for record in reviewed:
        assert record["verdict"] == "accept"
        assert re.fullmatch(r"[0-9a-f]{64}", record["source_sha"])
        assert TIMESTAMP_RE.match(record["reviewed_at"])


def test_review_correct_replaces_output_value(tmp_path):
    ws = FakeWorkspace(tmp_path)
    add_log(ws, "one", "WRONG!")
    counts = logs_mod.review_logs(
        ws, input_fn=scripted(["c", "ONE!"]), print_fn=lambda *_: None
    )
    assert counts == {"reviewed": 1, "accepted": 0, "corrected": 1, "rejected": 0}
    (record,) = logs_mod.read_reviewed(ws)
    assert record["verdict"] == "corrected"
    assert record["outputs"] == {"Loud": "ONE!"}
    assert record["inputs"] == {"text": "one"}


def test_review_correct_empty_answer_keeps_current_value(tmp_path):
    ws = FakeWorkspace(tmp_path)
    add_log(ws, "one", "ONE!")
    logs_mod.review_logs(ws, input_fn=scripted(["c", ""]), print_fn=lambda *_: None)
    (record,) = logs_mod.read_reviewed(ws)
    assert record["outputs"] == {"Loud": "ONE!"}


def test_review_reject_recorded_but_not_training_data(tmp_path):
    ws = FakeWorkspace(tmp_path)
    add_log(ws, "one", "GARBAGE")
    counts = logs_mod.review_logs(ws, input_fn=scripted(["r"]), print_fn=lambda *_: None)
    assert counts == {"reviewed": 1, "accepted": 0, "corrected": 0, "rejected": 1}
    assert logs_mod.read_reviewed(ws) == []
    raw = (ws.logs_dir / "reviewed.jsonl").read_text(encoding="utf-8")
    assert json.loads(raw.splitlines()[0])["verdict"] == "rejected"


def test_review_quit_stops_gracefully(tmp_path):
    ws = FakeWorkspace(tmp_path)
    add_log(ws, "one", "ONE!")
    add_log(ws, "two", "TWO!")
    counts = logs_mod.review_logs(ws, input_fn=scripted(["a", "q"]), print_fn=lambda *_: None)
    assert counts["reviewed"] == 1
    assert counts["accepted"] == 1


def test_review_eof_stops_gracefully(tmp_path):
    ws = FakeWorkspace(tmp_path)
    add_log(ws, "one", "ONE!")
    add_log(ws, "two", "TWO!")
    counts = logs_mod.review_logs(ws, input_fn=scripted(["a"]), print_fn=lambda *_: None)
    assert counts["reviewed"] == 1
    assert len(logs_mod.read_reviewed(ws)) == 1


def test_review_unknown_answer_reprompts(tmp_path):
    ws = FakeWorkspace(tmp_path)
    add_log(ws, "one", "ONE!")
    printed = []
    counts = logs_mod.review_logs(ws, input_fn=scripted(["x", "a"]), print_fn=printed.append)
    assert counts["accepted"] == 1
    assert any("a, c, r, or q" in line for line in printed)


def test_review_shows_inputs_and_outputs(tmp_path):
    ws = FakeWorkspace(tmp_path)
    add_log(ws, "one", "ONE!")
    printed = []
    logs_mod.review_logs(ws, input_fn=scripted(["a"]), print_fn=printed.append)
    joined = "\n".join(printed)
    assert "one" in joined and "ONE!" in joined


def test_review_skips_already_reviewed_entries(tmp_path):
    ws = FakeWorkspace(tmp_path)
    add_log(ws, "one", "ONE!")
    logs_mod.review_logs(ws, input_fn=scripted(["a"]), print_fn=lambda *_: None)
    printed = []
    counts = logs_mod.review_logs(ws, input_fn=scripted(["a"]), print_fn=printed.append)
    assert counts["reviewed"] == 0
    assert printed


def test_review_offers_only_new_entries_after_partial_review(tmp_path):
    ws = FakeWorkspace(tmp_path)
    add_log(ws, "one", "ONE!")
    logs_mod.review_logs(ws, input_fn=scripted(["a"]), print_fn=lambda *_: None)
    add_log(ws, "two", "TWO!")
    counts = logs_mod.review_logs(ws, input_fn=scripted(["a"]), print_fn=lambda *_: None)
    assert counts["reviewed"] == 1
    assert len(logs_mod.read_reviewed(ws)) == 2


def test_review_sample_caps_entries(tmp_path):
    ws = FakeWorkspace(tmp_path)
    for i in range(4):
        add_log(ws, f"t{i}", f"T{i}!")
    counts = logs_mod.review_logs(
        ws, sample=2, input_fn=scripted(["a"] * 10), print_fn=lambda *_: None
    )
    assert counts["reviewed"] == 2


def test_review_sampling_is_deterministic_by_seed(tmp_path):
    def build(root):
        ws = FakeWorkspace(root)
        ws.logs_dir.mkdir(parents=True)
        lines = []
        for i in range(6):
            lines.append(json.dumps({
                "inputs": {"text": f"t{i}"}, "outputs": {"Loud": f"T{i}!"},
                "candidate": "candidate_0", "n_repeat": 1,
                "timestamp": "2026-07-04T00:00:00Z",
            }))
        (ws.logs_dir / "2026-07-04.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        return ws

    picked = []
    for sub in ("a", "b"):
        ws = build(tmp_path / sub)
        logs_mod.review_logs(
            ws, sample=2, seed=3, input_fn=scripted(["a", "a"]), print_fn=lambda *_: None
        )
        picked.append([r["source_sha"] for r in logs_mod.read_reviewed(ws)])
    assert picked[0] == picked[1]


def test_review_no_logs_returns_zero_counts(tmp_path):
    ws = FakeWorkspace(tmp_path)
    printed = []
    counts = logs_mod.review_logs(ws, input_fn=scripted([]), print_fn=printed.append)
    assert counts == {"reviewed": 0, "accepted": 0, "corrected": 0, "rejected": 0}
    assert printed


def test_entry_sha_stable_across_disk_round_trip(tmp_path):
    ws = FakeWorkspace(tmp_path)
    add_log(ws, "one", "ONE!")
    (entry,) = logs_mod.read_logs(ws)
    again = json.loads(json.dumps(entry, ensure_ascii=False))
    assert logs_mod.entry_sha(entry) == logs_mod.entry_sha(again)
