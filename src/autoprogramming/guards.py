"""Data-discipline guards — enforced by the harness, not the agent's good manners.

Every rule the README promises — test belongs to finalize(), reflection
happens on train only, val selection pressure is tracked and capped,
memorizers are flagged — has its single enforcement point here, so every
code path refuses (or warns) with exactly the same reasoning.
"""

from __future__ import annotations

import json
import math
import warnings

from .errors import (
    BootstrapModeError,
    DataDisciplineError,
    ValReliabilityWarning,
    WorkspaceError,
)

BOOTSTRAP_MIN = 30
BOOTSTRAP_MAX_VAL_CANDIDATES = 5

_MEMORIZATION_GAP = 0.2
_MEMORIZATION_TRAIN_FLOOR = 0.5
_VERBATIM_MIN_LEN = 8


def assert_eval_allowed(split: str, per_instance: bool) -> None:
    """Refuse evaluations that would breach the data-splitting discipline.

    Allowed: aggregate or per-row scores on train; aggregate-only scores on
    val. Everything else raises DataDisciplineError with the reasoning.
    """
    if split == "test":
        raise DataDisciplineError(
            "Refusing to eval on 'test': test belongs to finalize() — it is "
            "evaluated once, at the end, on the top candidates only. If "
            "candidates could be scored on test during the search, the final "
            "report card would become just another val set and the number you "
            "tell your boss would be meaningless. Keep selecting on val, and "
            "call prg.finalize() when the budget is done."
        )
    if split == "val" and per_instance:
        raise DataDisciplineError(
            "Refusing per-row scores on 'val': the agent never sees why a val "
            "row scored low — only the aggregate — so it cannot edit "
            "candidates to fix specific val examples. Per-row scores would "
            "turn the selection set into a second training set. Reflect "
            "per-row on train instead: "
            "prg.eval(name, split='train', per_instance=True)."
        )
    if split not in ("train", "val"):
        raise DataDisciplineError(
            f"Unknown split {split!r}: evaluation runs on 'train' (reflection, "
            f"per-row allowed) or 'val' (selection, aggregate only). 'test' "
            f"belongs to finalize()."
        )


def assert_trace_allowed(split: str) -> None:
    """Refuse traced runs on anything but train rows.

    A trace shows exactly why a row failed; seeing that for val or test rows
    would let the agent edit candidates to fix the very examples used for
    selection and the final report.
    """
    if split != "train":
        raise DataDisciplineError(
            f"Refusing to trace a {split!r} row: the agent reflects on train "
            f"failures only. A full trace reveals exactly why a row scored "
            f"low, and having that for val or test rows would let candidates "
            f"be tuned to the examples that decide selection and the final "
            f"report. Trace train rows instead: "
            f"prg.run(name, split='train', row=<i>)."
        )


def is_bootstrap(workspace) -> bool:
    """Whether this workspace was split in bootstrap mode (data/split.json)."""
    return bool(_read_split(workspace).get("bootstrap", False))


def register_val_candidate(workspace, name: str) -> None:
    """Record that a candidate is being scored on val, before any spend.

    In bootstrap mode the cap on distinct val-scored candidates raises
    BEFORE the name is recorded and before any evaluation cost is incurred.
    After registration, selection pressure on val is re-checked and a
    ValReliabilityWarning is emitted once val has absorbed too many
    selection decisions for its size.
    """
    scores = _load_scores(workspace)
    val_scored = list(scores.get("val_scored", []))
    if name not in val_scored:
        if is_bootstrap(workspace) and len(set(val_scored)) >= BOOTSTRAP_MAX_VAL_CANDIDATES:
            raise BootstrapModeError(
                f"Refusing to score {name!r} on val: this workspace is in "
                f"bootstrap mode (fewer than {BOOTSTRAP_MIN} examples), and "
                f"{BOOTSTRAP_MAX_VAL_CANDIDATES} distinct candidates have "
                f"already been compared on val. At this data size a "
                f"0.92-vs-0.86 difference is one row of noise, so fine-grained "
                f"mutation loops would be selecting on noise — bootstrap mode "
                f"builds and compares baselines only. Pick the best of the "
                f"candidates already scored, or generate synthetic examples "
                f"for the user to validate to unlock full optimization."
            )
        val_scored.append(name)
        scores["val_scored"] = val_scored
        _save_scores(workspace, scores)

    val_size = int(_read_split(workspace).get("counts", {}).get("val", 0))
    n_distinct = len(set(val_scored))
    status = pressure_status(n_distinct, val_size)
    if status == "warn":
        warnings.warn(ValReliabilityWarning(
            f"val has now selected among {n_distinct} candidates but has only "
            f"{val_size} rows. Every comparison leaks a little of val into "
            f"the search, so its scores are starting to lose meaning — "
            f"prefer fewer, more different candidates over many small tweaks."
        ))
    elif status == "unreliable":
        warnings.warn(ValReliabilityWarning(
            f"val has selected among {n_distinct} candidates with only "
            f"{val_size} rows — more than three times its size. Val scores "
            f"have lost their meaning, will not be reported as final, and the "
            f"one-time test evaluation in finalize() is the only number left "
            f"worth quoting."
        ))


def pressure_status(n_distinct: int, val_size: int) -> str:
    """Selection pressure on val: 'ok', 'warn', or 'unreliable'.

    'warn' once more distinct candidates than val rows have been compared;
    'unreliable' once that count exceeds three times the val size.
    """
    if val_size <= 0:
        return "unreliable" if n_distinct > 0 else "ok"
    if n_distinct > 3 * val_size:
        return "unreliable"
    if n_distinct > val_size:
        return "warn"
    return "ok"


def memorization_check(candidate_source: str, train_mean: float, val_mean: float,
                       train_rows: list[dict], schema) -> list[str]:
    """Flag candidates that memorized train instead of learning the task.

    Two rules: a train score vastly exceeding val (gap > 0.2 with train mean
    above 0.5), and verbatim training outputs embedded in the candidate
    source (a lookup table over train inputs). Returns the flag strings,
    empty when clean. Verbatim counting is over distinct qualifying string
    outputs (len >= 8 after stripping), against a threshold of
    max(3, ceil(0.1 * len(train_rows))).
    """
    flags: list[str] = []

    gap = train_mean - val_mean
    if gap > _MEMORIZATION_GAP and train_mean > _MEMORIZATION_TRAIN_FLOOR:
        flags.append(
            f"memorizer: train mean {train_mean:.3f} vastly exceeds val mean "
            f"{val_mean:.3f} (gap {gap:.3f} > {_MEMORIZATION_GAP}) — the "
            f"candidate learned the train rows, not the task, so its score "
            f"will not transfer to data it never saw; excluded from selection."
        )

    threshold = max(3, math.ceil(0.1 * len(train_rows)))
    seen: set[str] = set()
    found = 0
    for row in train_rows:
        for field in schema.outputs:
            if field.base is not str:
                continue
            value = row.get(field.name)
            if not isinstance(value, str):
                continue
            stripped = value.strip()
            if len(stripped) < _VERBATIM_MIN_LEN or stripped in seen:
                continue
            seen.add(stripped)
            if stripped in candidate_source:
                found += 1
    if found >= threshold:
        flags.append(
            f"memorizer: {found} distinct train outputs appear verbatim in the "
            f"candidate source (threshold {threshold}) — a lookup table over "
            f"train inputs scores perfectly on rows it has seen and collapses "
            f"on rows it hasn't; excluded from selection. A regex or rules "
            f"candidate may still win, but only on data it never saw."
        )

    return flags


def _scores_skeleton() -> dict:
    return {"metric_sha": None, "candidates": {}, "val_scored": [], "flags": {}}


def _load_scores(workspace) -> dict:
    path = workspace.scores_json
    if path.exists():
        scores = json.loads(path.read_text())
    else:
        scores = _scores_skeleton()
    scores.setdefault("val_scored", [])
    return scores


def _save_scores(workspace, scores: dict) -> None:
    workspace.scores_json.write_text(json.dumps(scores, indent=2) + "\n")


def _read_split(workspace) -> dict:
    path = workspace.split_json
    if not path.exists():
        raise WorkspaceError(
            f"data/split.json is missing from {getattr(workspace, 'root', path.parent)} "
            f"— the data is split exactly once, at optimize() time, and that "
            f"record is what the guards enforce against. Create the workspace "
            f"through optimize() (or Workspace.create) instead of by hand."
        )
    return json.loads(path.read_text())
