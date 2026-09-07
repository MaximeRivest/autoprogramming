"""Data handling: normalize any accepted source into rows, split once, guard access.

On ``optimize()`` the data is split exactly once into train/val/test. This
module turns sources into uniform rows, computes an order-independent identity
hash, performs the deterministic split, and exposes the splits with the
discipline the README promises: train is readable, val is scoring-only, and
test is not reachable from here at all.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import warnings
from pathlib import Path

from .errors import DataDisciplineError, SchemaError
from .schema import Schema

DEFAULT_RATIOS = (0.6, 0.2, 0.2)
MIN_ROWS = 5


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def normalize_rows(source, schema: Schema) -> list[dict[str, str]]:
    """Turn an accepted data source into a list of row dicts.

    Accepted sources: a list of dicts, a duck-typed DataFrame (anything with
    ``.to_dict("records")``), or a str/Path to an existing ``.csv`` or
    ``.jsonl`` file. The strings ``"logs"`` and ``"logs:reviewed"`` are
    resolved by ``Program.optimize()`` before this function is called, so a
    bare string that is not a data file is refused here.

    Every row must cover ``schema.expected_columns`` (input parameter names
    plus output type names); missing columns are a SchemaError. Extra columns
    are dropped with a UserWarning. Values are kept as-is — CSV gives strings,
    and coercion happens at eval time via the schema.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix not in (".csv", ".jsonl"):
            raise SchemaError(
                f"Refused data source {source!r}: it is a string but not a path to a "
                f".csv or .jsonl file, and only concrete data can be split and scored. "
                f"Accepted sources: a list of dicts, a DataFrame, or a path to a .csv "
                f"or .jsonl file. The special strings 'logs' and 'logs:reviewed' only "
                f"work through Program.optimize()/distill() on a program bound to a "
                f"workspace, because they resolve against that workspace's logs/."
            )
        if not path.exists():
            raise SchemaError(
                f"Refused data source {str(path)!r}: the file does not exist, so there "
                f"are no rows to normalize. Check the path, or pass the rows directly "
                f"as a list of dicts or a DataFrame."
            )
        rows = read_csv(path) if suffix == ".csv" else _read_jsonl(path)
    elif isinstance(source, list):
        for item in source:
            if not isinstance(item, dict):
                raise SchemaError(
                    f"Refused data source: list items must be dicts mapping column "
                    f"names to values (got {type(item).__name__!r}), because rows are "
                    f"matched to the schema by column name. Convert each example to "
                    f"a dict with keys {list(schema.expected_columns)!r}."
                )
        rows = [dict(item) for item in source]
    elif hasattr(source, "to_dict"):
        rows = [dict(r) for r in source.to_dict("records")]
    else:
        raise SchemaError(
            f"Refused data source of type {type(source).__name__!r}: rows must map "
            f"column names onto the schema, so only named-column sources are "
            f"accepted — a list of dicts, a DataFrame (anything with "
            f".to_dict('records')), or a path to a .csv/.jsonl file."
        )

    expected = schema.expected_columns
    missing: set[str] = set()
    extras: set[str] = set()
    for row in rows:
        missing.update(c for c in expected if c not in row)
        extras.update(k for k in row if k not in expected)
    if missing:
        raise SchemaError(
            f"Data does not cover the schema of {schema.name!r}: missing columns "
            f"{sorted(missing)!r}. Every row must provide every input (parameter "
            f"names) and every expected output (output type names) — "
            f"{list(expected)!r} — so candidates can be run and scored against it. "
            f"Add or rename the columns to match."
        )
    if extras:
        warnings.warn(
            f"Dropping data columns {sorted(extras)!r}: they are not part of the "
            f"schema of {schema.name!r} (expected {list(expected)!r}) and cannot "
            f"be used to run or score candidates.",
            UserWarning,
            stacklevel=2,
        )
    return [{c: row[c] for c in expected} for row in rows]


def data_sha(rows: list[dict]) -> str:
    """Order-independent content hash of a row set.

    Each row is serialized to canonical JSON (sorted keys) and the list of
    serializations is sorted before hashing, so reordering rows never changes
    the identity but changing any value does. This is what pins a workspace
    to the data it was split from.
    """
    canonical = sorted(
        json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
        for row in rows
    )
    return hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()


def split_rows(rows, seed: int = 0, ratios=DEFAULT_RATIOS) -> dict[str, list[dict]]:
    """Split rows once into train/val/test, deterministically by seed.

    A copy is shuffled with ``random.Random(seed)``; test and val sizes are
    ``max(1, round(n * ratio))`` and train is the remainder. The same rows and
    seed always produce the same split — the split happens once per workspace
    and is never redone.
    """
    n = len(rows)
    if n < MIN_ROWS:
        raise DataDisciplineError(
            f"Refusing to split {n} row(s): optimization needs at least {MIN_ROWS} "
            f"examples to form train/val/test at all — below that even bootstrap "
            f"mode (which below 30 examples only builds and compares baseline "
            f"candidates) has nothing to select on. Gather more examples, or let "
            f"the agent generate synthetic examples for you to validate."
        )
    if len(tuple(ratios)) != 3:
        raise DataDisciplineError(
            f"Refusing ratios {ratios!r}: a split needs exactly three ratios "
            f"(train, val, test), because the data discipline is built on those "
            f"three roles. Pass e.g. ratios={DEFAULT_RATIOS!r}."
        )
    r_train, r_val, r_test = ratios
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, round(n * r_test))
    n_val = max(1, round(n * r_val))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise DataDisciplineError(
            f"Refusing ratios {tuple(ratios)!r} for {n} rows: they leave "
            f"{n_train} train row(s) after reserving {n_val} for val and {n_test} "
            f"for test, and the agent reflects on train failures only — with no "
            f"train rows there is nothing to learn from. Use ratios that keep at "
            f"least one train row, e.g. the default {DEFAULT_RATIOS!r}."
        )
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def development_partition(
    rows: list[dict], *, seed: int = 0, probe_ratio: float = 0.2
) -> tuple[list[dict], list[dict]]:
    """Split the train split into worker-visible fit and controller-only probe.

    This does not alter val/test and is deterministic for the workspace seed.
    Very small train sets keep all rows visible because withholding one would
    leave too little implementation context; bootstrap mode already caps search.
    """
    if not 0.0 <= probe_ratio < 1.0:
        raise DataDisciplineError(
            f"probe_ratio must be in [0, 1), got {probe_ratio!r}."
        )
    copied = [dict(row) for row in rows]
    if len(copied) < 5 or probe_ratio == 0:
        return copied, []
    random.Random(seed ^ 0xA17E).shuffle(copied)
    n_probe = max(1, round(len(copied) * probe_ratio))
    return copied[n_probe:], copied[:n_probe]


def write_csv(path, rows, columns) -> None:
    """Write rows to a UTF-8 CSV with ``columns`` as the header row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def read_csv(path) -> list[dict[str, str]]:
    """Read a UTF-8 CSV written by :func:`write_csv` back into row dicts."""
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def load_split(workspace, split: str) -> list[dict[str, str]]:
    """Read ``data/<split>.csv`` from a workspace.

    Internal accessor for the scoring harness and finalize(); it carries no
    guard of its own — refusals are guards.py's job, this just reads.
    """
    path = (
        workspace.split_path(split)
        if hasattr(workspace, "split_path")
        else Path(workspace.data_dir) / f"{split}.csv"
    )
    return read_csv(path)


class Rows:
    """A readable sequence view over a split's rows (used for train).

    Supports ``len``, iteration, indexing (returning dict copies so the
    underlying split cannot be mutated), ``.columns``, and deterministic
    ``.sample(k, seed=0)``.
    """

    def __init__(self, rows: list[dict], split: str = "train"):
        self._rows = tuple(rows)
        self._split = split

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        return (dict(row) for row in self._rows)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [dict(row) for row in self._rows[index]]
        return dict(self._rows[index])

    @property
    def columns(self) -> tuple[str, ...]:
        """Column names, in row order; empty tuple when there are no rows."""
        return tuple(self._rows[0].keys()) if self._rows else ()

    def sample(self, k: int, seed: int = 0) -> list[dict]:
        """A deterministic random sample of ``k`` rows (as dict copies)."""
        picked = random.Random(seed).sample(self._rows, k)
        return [dict(row) for row in picked]

    def __repr__(self) -> str:
        return f"Rows(split={self._split!r}, n={len(self._rows)}, columns={self.columns!r})"


class GuardedRows:
    """The val split: its size is visible, its contents are not.

    Only ``len()`` works. Iteration and indexing are refused because val
    exists for selection only — every candidate is scored on the identical
    val set and the agent must never see *why* a val row scored low, only the
    aggregate; otherwise it could edit candidates to fix specific val
    examples, which silently turns val into train.
    """

    def __init__(self, rows: list[dict], split: str = "val"):
        self._n = len(rows)
        self._split = split

    def __len__(self) -> int:
        return self._n

    def _refuse(self, what: str) -> DataDisciplineError:
        return DataDisciplineError(
            f"Refused: {what} on the {self._split} split. Val rows are for "
            f"selection only — candidates are compared on the identical val set, "
            f"and the agent never sees why a val row scored low, only the "
            f"aggregate — so it cannot edit candidates to fix specific val "
            f"examples. Reflect on train instead (prg.data.train, or "
            f"prg.run(name, split='train', row=i) for a full trace) and read val "
            f"as an aggregate via prg.eval(name)."
        )

    def __iter__(self):
        raise self._refuse("iterating over rows")

    def __getitem__(self, index):
        raise self._refuse(f"reading row {index!r}")

    def __repr__(self) -> str:
        return f"GuardedRows(split={self._split!r}, n={self._n})"


class SplitView:
    """Lazy view of a workspace's splits: ``.train`` readable, ``.val`` guarded.

    There is deliberately no ``.test`` attribute at all — test belongs to
    ``finalize()``, which evaluates it exactly once at the end.
    """

    def __init__(self, workspace):
        self._workspace = workspace
        self._train: Rows | None = None
        self._val: GuardedRows | None = None

    @property
    def train(self) -> Rows:
        """The train rows — readable, iterable, sampleable."""
        if self._train is None:
            self._train = Rows(load_split(self._workspace, "train"), split="train")
        return self._train

    @property
    def val(self) -> GuardedRows:
        """The val rows — sized but unreadable (scoring only)."""
        if self._val is None:
            self._val = GuardedRows(load_split(self._workspace, "val"), split="val")
        return self._val

    def __repr__(self) -> str:
        return f"SplitView(train={len(self.train)}, val={len(self.val)})"
