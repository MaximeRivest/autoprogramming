"""Tests for autoprogramming.data — normalization, split, guarded access."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoprogramming import data as data_mod
from autoprogramming.errors import DataDisciplineError, SchemaError
from autoprogramming.schema import Schema


class Loud(str):
    """The input, uppercased, with an exclamation mark."""


def shout(text: str) -> Loud:
    """Uppercase the text."""


SCHEMA = Schema.from_function(shout)


def make_rows(n: int) -> list[dict[str, str]]:
    return [{"text": f"hello {i}", "Loud": f"HELLO {i}!"} for i in range(n)]


class FakeWorkspace:
    def __init__(self, root: Path):
        self.root = root
        self.data_dir = root / "data"


def make_workspace(tmp_path: Path, train=None, val=None, test=None) -> FakeWorkspace:
    ws = FakeWorkspace(tmp_path)
    for split, rows in (("train", train), ("val", val), ("test", test)):
        if rows is not None:
            data_mod.write_csv(ws.data_dir / f"{split}.csv", rows, SCHEMA.expected_columns)
    return ws


# ------------------------------------------------------------- normalize_rows


def test_normalize_rows_list_of_dicts_passthrough():
    rows = make_rows(3)
    out = data_mod.normalize_rows(rows, SCHEMA)
    assert out == rows
    assert out[0] is not rows[0]


def test_normalize_rows_values_stay_as_is():
    rows = [{"text": 7, "Loud": 3.5}]
    out = data_mod.normalize_rows(rows, SCHEMA)
    assert out == [{"text": 7, "Loud": 3.5}]


def test_normalize_rows_missing_column_raises_schema_error():
    with pytest.raises(SchemaError) as exc:
        data_mod.normalize_rows([{"text": "hi"}], SCHEMA)
    assert "Loud" in str(exc.value)


def test_normalize_rows_extra_column_warns_and_drops():
    rows = [{"text": "hi", "Loud": "HI!", "note": "extra"}]
    with pytest.warns(UserWarning, match="note"):
        out = data_mod.normalize_rows(rows, SCHEMA)
    assert out == [{"text": "hi", "Loud": "HI!"}]


def test_normalize_rows_non_dict_list_items_refused():
    with pytest.raises(SchemaError):
        data_mod.normalize_rows(["not a dict"], SCHEMA)


def test_normalize_rows_duck_typed_dataframe():
    class FakeFrame:
        def __init__(self, records):
            self._records = records

        def to_dict(self, orient):
            assert orient == "records"
            return [dict(r) for r in self._records]

    out = data_mod.normalize_rows(FakeFrame(make_rows(2)), SCHEMA)
    assert out == make_rows(2)


def test_normalize_rows_csv_path(tmp_path):
    path = tmp_path / "rows.csv"
    data_mod.write_csv(path, make_rows(3), SCHEMA.expected_columns)
    assert data_mod.normalize_rows(str(path), SCHEMA) == make_rows(3)
    assert data_mod.normalize_rows(path, SCHEMA) == make_rows(3)


def test_normalize_rows_jsonl_path(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in make_rows(3)) + "\n", encoding="utf-8"
    )
    assert data_mod.normalize_rows(path, SCHEMA) == make_rows(3)


def test_normalize_rows_bare_string_refused_with_accepted_sources():
    with pytest.raises(SchemaError) as exc:
        data_mod.normalize_rows("logs", SCHEMA)
    msg = str(exc.value)
    assert ".csv" in msg and ".jsonl" in msg
    assert "optimize" in msg


def test_normalize_rows_missing_file_refused(tmp_path):
    with pytest.raises(SchemaError) as exc:
        data_mod.normalize_rows(tmp_path / "nope.csv", SCHEMA)
    assert "does not exist" in str(exc.value)


def test_normalize_rows_unsupported_type_refused():
    with pytest.raises(SchemaError):
        data_mod.normalize_rows(42, SCHEMA)


# ------------------------------------------------------------------- data_sha


def test_data_sha_is_order_independent():
    rows = make_rows(10)
    assert data_mod.data_sha(rows) == data_mod.data_sha(list(reversed(rows)))


def test_data_sha_changes_with_content():
    rows = make_rows(10)
    changed = [dict(r) for r in rows]
    changed[3]["Loud"] = "SOMETHING ELSE!"
    assert data_mod.data_sha(rows) != data_mod.data_sha(changed)


def test_data_sha_ignores_key_order_within_rows():
    a = [{"text": "hi", "Loud": "HI!"}]
    b = [{"Loud": "HI!", "text": "hi"}]
    assert data_mod.data_sha(a) == data_mod.data_sha(b)


# ----------------------------------------------------------------- split_rows


def test_split_rows_same_seed_same_split():
    rows = make_rows(30)
    assert data_mod.split_rows(rows, seed=7) == data_mod.split_rows(rows, seed=7)


def test_split_rows_different_seed_different_split():
    rows = make_rows(30)
    assert data_mod.split_rows(rows, seed=0) != data_mod.split_rows(rows, seed=1)


def test_split_rows_default_sizes():
    splits = data_mod.split_rows(make_rows(10))
    assert len(splits["train"]) == 6
    assert len(splits["val"]) == 2
    assert len(splits["test"]) == 2


def test_split_rows_minimum_viable_sizes():
    splits = data_mod.split_rows(make_rows(5))
    assert len(splits["val"]) >= 1
    assert len(splits["test"]) >= 1
    assert len(splits["train"]) >= 1


def test_split_rows_partitions_without_loss_or_overlap():
    rows = make_rows(23)
    splits = data_mod.split_rows(rows, seed=3)
    combined = splits["train"] + splits["val"] + splits["test"]
    assert len(combined) == len(rows)
    key = lambda r: json.dumps(r, sort_keys=True)
    assert sorted(map(key, combined)) == sorted(map(key, rows))


def test_split_rows_too_few_rows_refused_with_synthetic_offer():
    with pytest.raises(DataDisciplineError) as exc:
        data_mod.split_rows(make_rows(4))
    assert "synthetic" in str(exc.value)


def test_split_rows_ratios_leaving_no_train_refused():
    with pytest.raises(DataDisciplineError) as exc:
        data_mod.split_rows(make_rows(5), ratios=(0.0, 0.6, 0.4))
    assert "train" in str(exc.value)


def test_split_rows_does_not_mutate_input():
    rows = make_rows(12)
    original = [dict(r) for r in rows]
    data_mod.split_rows(rows, seed=5)
    assert rows == original


# ------------------------------------------------------------------- csv i/o


def test_write_read_csv_round_trip(tmp_path):
    path = tmp_path / "data" / "train.csv"
    rows = make_rows(4)
    data_mod.write_csv(path, rows, SCHEMA.expected_columns)
    assert data_mod.read_csv(path) == rows


def test_write_csv_unicode_round_trip(tmp_path):
    path = tmp_path / "fr.csv"
    rows = [{"text": "where is the station", "Loud": "OÙ EST LA GARE ?"}]
    data_mod.write_csv(path, rows, SCHEMA.expected_columns)
    assert data_mod.read_csv(path) == rows


def test_load_split_reads_workspace_csv(tmp_path):
    ws = make_workspace(tmp_path, train=make_rows(3))
    assert data_mod.load_split(ws, "train") == make_rows(3)


# ----------------------------------------------------------------------- Rows


def test_rows_len_iter_getitem():
    view = data_mod.Rows(make_rows(4))
    assert len(view) == 4
    assert list(view) == make_rows(4)
    assert view[1] == make_rows(4)[1]


def test_rows_getitem_returns_a_copy():
    view = data_mod.Rows(make_rows(2))
    row = view[0]
    row["text"] = "mutated"
    assert view[0]["text"] == "hello 0"


def test_rows_columns():
    assert data_mod.Rows(make_rows(1)).columns == ("text", "Loud")
    assert data_mod.Rows([]).columns == ()


def test_rows_sample_is_deterministic():
    view = data_mod.Rows(make_rows(10))
    assert view.sample(4, seed=1) == view.sample(4, seed=1)
    for row in view.sample(4, seed=1):
        assert row in make_rows(10)


# ---------------------------------------------------------------- GuardedRows


def test_guarded_rows_len_works():
    guarded = data_mod.GuardedRows(make_rows(4))
    assert len(guarded) == 4


def test_guarded_rows_getitem_refused():
    guarded = data_mod.GuardedRows(make_rows(4))
    with pytest.raises(DataDisciplineError):
        guarded[0]


def test_guarded_rows_iteration_refused():
    guarded = data_mod.GuardedRows(make_rows(4))
    with pytest.raises(DataDisciplineError):
        list(guarded)


def test_guarded_rows_refusal_message_explains_rule_and_alternative():
    guarded = data_mod.GuardedRows(make_rows(4))
    with pytest.raises(DataDisciplineError) as exc:
        guarded[2]
    msg = str(exc.value).lower()
    assert "selection" in msg
    assert "aggregate" in msg
    assert "train" in msg
    assert "eval" in msg


def test_guarded_rows_repr_does_not_leak_contents():
    guarded = data_mod.GuardedRows(make_rows(3))
    assert "hello" not in repr(guarded)


# ------------------------------------------------------------------ SplitView


def test_split_view_train_is_readable(tmp_path):
    ws = make_workspace(tmp_path, train=make_rows(6), val=make_rows(2))
    view = data_mod.SplitView(ws)
    assert isinstance(view.train, data_mod.Rows)
    assert list(view.train) == make_rows(6)


def test_split_view_val_is_guarded(tmp_path):
    ws = make_workspace(tmp_path, train=make_rows(6), val=make_rows(2))
    view = data_mod.SplitView(ws)
    assert isinstance(view.val, data_mod.GuardedRows)
    assert len(view.val) == 2
    with pytest.raises(DataDisciplineError):
        view.val[0]


def test_split_view_has_no_test_attribute(tmp_path):
    ws = make_workspace(
        tmp_path, train=make_rows(6), val=make_rows(2), test=make_rows(2)
    )
    view = data_mod.SplitView(ws)
    assert not hasattr(view, "test")


def test_split_view_loads_lazily(tmp_path):
    view = data_mod.SplitView(FakeWorkspace(tmp_path / "nowhere"))
    with pytest.raises(FileNotFoundError):
        view.train
