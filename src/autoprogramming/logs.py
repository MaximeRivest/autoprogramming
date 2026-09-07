"""Production traffic logs and the human review flow.

The shipped package appends one JSONL entry per call (the README's exact
format) to ``logs/<UTC date>.jsonl``. Logs alone can only be imitated
(distill); making the program *better* needs a correction signal, so
``review_logs`` walks a human through accept/correct/reject and only the
reviewed entries in ``logs/reviewed.jsonl`` become new training data.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from .errors import SchemaError
from .schema import Schema

_REVIEWED_NAME = "reviewed.jsonl"
_PROMPT = "(a)ccept / (c)orrect / (r)eject / (q)uit: "


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def entry_sha(entry: dict) -> str:
    """sha256 of a log entry's canonical JSON — its identity for review.

    Canonical means sorted keys, so an entry read back from disk hashes the
    same as the entry that was written.
    """
    canonical = json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


def _append_jsonl(path: Path, entry: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def append_log(workspace, inputs: dict, outputs: dict, candidate: str, n_repeat: int = 1) -> Path:
    """Append one production-traffic entry to ``logs/<UTC date>.jsonl``.

    The line is exactly the README's format: ``inputs`` (parameter names),
    ``outputs`` (output type names — the two live in separate objects so they
    can never collide), ``candidate``, ``n_repeat``, and a UTC ``timestamp``.
    Creates ``logs/`` lazily. Returns the file the entry was appended to.
    """
    logs_dir = Path(workspace.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "inputs": dict(inputs),
        "outputs": dict(outputs),
        "candidate": candidate,
        "n_repeat": int(n_repeat),
        "timestamp": _timestamp(),
    }
    path = logs_dir / f"{_utc_now().strftime('%Y-%m-%d')}.jsonl"
    _append_jsonl(path, entry)
    return path


def read_logs(workspace) -> list[dict]:
    """All production log entries, in filename order.

    Reads every ``logs/*.jsonl`` except ``reviewed.jsonl`` (review verdicts
    are not traffic). Returns [] when nothing has been logged yet.
    """
    logs_dir = Path(workspace.logs_dir)
    if not logs_dir.is_dir():
        return []
    entries: list[dict] = []
    for path in sorted(logs_dir.glob("*.jsonl")):
        if path.name == _REVIEWED_NAME:
            continue
        entries.extend(_read_jsonl(path))
    return entries


def read_reviewed(workspace) -> list[dict]:
    """Reviewed entries usable as training data.

    Only verdicts ``"accept"`` and ``"corrected"`` qualify — rejected entries
    carry no usable target. Returns [] when nothing has been reviewed.
    """
    path = Path(workspace.logs_dir) / _REVIEWED_NAME
    if not path.exists():
        return []
    return [e for e in _read_jsonl(path) if e.get("verdict") in ("accept", "corrected")]


def logs_to_rows(entries: list[dict], schema: Schema) -> list[dict]:
    """Turn log entries into data rows: ``{**inputs, **outputs}`` per entry.

    Refuses entries that do not cover the schema's expected columns — logs
    written by a different program (or an older schema) cannot be scored
    against this one.
    """
    expected = schema.expected_columns
    rows: list[dict] = []
    missing: set[str] = set()
    for entry in entries:
        row = {**entry.get("inputs", {}), **entry.get("outputs", {})}
        missing.update(c for c in expected if c not in row)
        rows.append(row)
    if missing:
        raise SchemaError(
            f"Refusing to use these log entries as data for {schema.name!r}: "
            f"columns {sorted(missing)!r} are missing (expected inputs+outputs "
            f"{list(expected)!r}). Every training row must provide every input "
            f"and every expected output so candidates can be scored. These logs "
            f"were likely written by a different program or schema — log fresh "
            f"traffic with this program, or fix the entries."
        )
    return rows


def review_logs(workspace, sample: int | None = None, input_fn=input, print_fn=print, seed: int = 0) -> dict:
    """Interactive review of sampled, not-yet-reviewed log entries.

    Samples ``min(sample or 50, n_unreviewed)`` entries (deterministic by
    ``seed``; identity is :func:`entry_sha`, so an entry is never offered
    twice). For each entry the inputs and outputs are shown and the reviewer
    answers accept / correct / reject / quit; ``correct`` prompts for a
    replacement value per output field (empty keeps the current value).
    ``q`` or end of input stops gracefully — progress already written to
    ``logs/reviewed.jsonl`` is kept.

    ``input_fn`` and ``print_fn`` exist so the loop is drivable without a
    terminal. Returns ``{"reviewed": n, "accepted": n, "corrected": n,
    "rejected": n}``.
    """
    counts = {"reviewed": 0, "accepted": 0, "corrected": 0, "rejected": 0}
    entries = read_logs(workspace)
    reviewed_path = Path(workspace.logs_dir) / _REVIEWED_NAME
    seen: set[str] = set()
    if reviewed_path.exists():
        seen = {e.get("source_sha") for e in _read_jsonl(reviewed_path)}
    unreviewed = [e for e in entries if entry_sha(e) not in seen]
    if not unreviewed:
        print_fn(
            "No unreviewed log entries. Enable logging (program.enable_logging()) "
            "and gather traffic first, or everything logged so far has already "
            "been reviewed."
        )
        return counts

    k = min(sample if sample is not None else 50, len(unreviewed))
    picked = random.Random(seed).sample(unreviewed, k)
    verdict_counter = {"accept": "accepted", "corrected": "corrected", "rejected": "rejected"}

    for i, entry in enumerate(picked, 1):
        print_fn(f"[{i}/{k}] inputs:  {json.dumps(entry.get('inputs', {}), ensure_ascii=False)}")
        print_fn(f"        outputs: {json.dumps(entry.get('outputs', {}), ensure_ascii=False)}")
        try:
            while True:
                choice = input_fn(_PROMPT).strip().lower()
                if choice in ("a", "c", "r", "q"):
                    break
                print_fn("Please answer a, c, r, or q.")
            if choice == "q":
                print_fn("Stopping review; verdicts recorded so far are kept.")
                break
            outputs = dict(entry.get("outputs", {}))
            if choice == "a":
                verdict = "accept"
            elif choice == "r":
                verdict = "rejected"
            else:
                for name, current in outputs.items():
                    replacement = input_fn(f"  {name} [{current}]: ")
                    if replacement != "":
                        outputs[name] = replacement
                verdict = "corrected"
        except EOFError:
            print_fn("Input ended; stopping review. Verdicts recorded so far are kept.")
            break
        record = {
            "inputs": entry.get("inputs", {}),
            "outputs": outputs,
            "verdict": verdict,
            "source_sha": entry_sha(entry),
            "reviewed_at": _timestamp(),
        }
        _append_jsonl(reviewed_path, record)
        counts["reviewed"] += 1
        counts[verdict_counter[verdict]] += 1
    return counts
