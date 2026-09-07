"""The evaluation metric: proposed, demonstrated, approved — never silently trusted.

The entire search optimizes whatever the workspace's ``metric.py`` says, so a
wrong metric produces a confidently-scored wrong program. This module owns the
metric file's lifecycle: writing it, hashing it, recording user sign-off, and
invalidating recorded scores when the metric changes — scores under different
metrics are never comparable.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .errors import MetricChangedError, MetricNotApprovedError, SchemaError

if TYPE_CHECKING:
    from .schema import Schema
    from .workspace import Workspace

AUTO_APPROVE_ENV = "AP_AUTO_APPROVE_METRIC"
_TRUTHY = ("1", "true", "yes")

_METRIC_CACHE: dict[tuple[str, str], Callable] = {}


_STAMP_PREFIX = "# metric.py — approved by"


def _without_stamp(code: str) -> str:
    """The metric code with any approval-stamp first line removed.

    The stamp is sign-off metadata, not metric semantics: approving a metric
    (or re-approving it on a later date) must never look like the metric
    itself changed, or the very act of blessing a metric would invalidate
    the scores recorded under it.
    """
    lines = code.splitlines()
    if lines and lines[0].startswith(_STAMP_PREFIX):
        lines = lines[1:]
    return "\n".join(lines)


def metric_sha(workspace) -> str | None:
    """sha256 hex digest of metric.py's code, or None if absent.

    The approval stamp line is excluded from the hash — see
    :func:`_without_stamp` — so the sha changes exactly when the scoring
    behavior can change.
    """
    path = Path(workspace.metric_py)
    if not path.exists():
        return None
    code = _without_stamp(path.read_text(encoding="utf-8"))
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def scores_skeleton(sha: str | None) -> dict:
    """The empty scores.json structure, tied to the given metric sha.

    ``objectives`` (name -> "max"/"min") and ``primary`` are the multi-objective
    additions; they stay empty/None until an ``evaluate`` populates them.
    """
    return {
        "metric_sha": sha,
        "objectives": {},
        "primary": None,
        "candidates": {},
        "val_scored": [],
        "flags": {},
    }


def invalidate_scores(workspace, new_sha: str | None) -> None:
    """Reconcile scores.json to a changed metric — recover from cache, else archive.

    Called when the metric changes — by :func:`write_metric` here and by
    ``scoring.load_scores`` (whoever notices the change first). Candidates whose
    outputs are still cached are re-scored under the new metric for free; only
    candidates whose cache is stale or absent are unrecoverable, so those are
    archived to scores.archive/ and warned about (scores under different metrics
    are never comparable). Cost/latency objectives survive untouched. The heavy
    lifting lives in ``scoring.reconcile_metric_change`` (it owns the outputs
    cache and the objective aggregation); this thin wrapper avoids a circular
    import at module load.
    """
    from . import scoring

    scoring.reconcile_metric_change(workspace, new_sha)


def write_metric(workspace, code: str) -> None:
    """Write metric.py; if its code changes, archive and clear recorded scores.

    A changed metric also voids any prior approval implicitly: the approval
    record pins the old sha, so :func:`ensure_approved` will refuse until the
    new metric is signed off again. Proposing code identical to what is
    already on disk (approval stamp aside) is a no-op that keeps the stamped
    file, the approval, and every recorded score — re-proposing the same
    metric in a fresh session must not wipe anything.
    """
    path = Path(workspace.metric_py)
    if path.exists() and _without_stamp(path.read_text(encoding="utf-8")) == _without_stamp(code):
        return
    old = metric_sha(workspace)
    path.write_text(code)
    new = metric_sha(workspace)
    if old is not None and old != new:
        invalidate_scores(workspace, new)


def _read_approval(workspace) -> dict | None:
    path = Path(workspace.metric_approval)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, dict) else None


def is_approved(workspace) -> bool:
    """True when an approval record exists and matches the current metric.py sha."""
    sha = metric_sha(workspace)
    if sha is None:
        return False
    approval = _read_approval(workspace)
    return approval is not None and approval.get("sha") == sha


def ensure_approved(workspace) -> None:
    """Refuse to proceed unless the current metric.py carries a valid sign-off.

    This is the only place the ``AP_AUTO_APPROVE_METRIC`` environment variable
    is consulted: a truthy value ("1"/"true"/"yes") auto-approves an
    unapproved metric for unattended runs. A metric whose approval sha no
    longer matches is never auto-approved — it changed after sign-off.
    """
    sha = metric_sha(workspace)
    if sha is None:
        raise MetricNotApprovedError(
            "Refused to score: this workspace has no metric.py. The entire search "
            "optimizes whatever metric.py says, so scoring without a metric would "
            "optimize an undefined objective. Propose one with "
            "harness.propose_metric(code, examples), or write it with "
            "autoprogramming.metric.write_metric(ws, code) and approve it."
        )
    approval = _read_approval(workspace)
    if approval is not None:
        if approval.get("sha") == sha:
            return
        raise MetricChangedError(
            f"metric.py changed after it was approved (approved sha "
            f"{str(approval.get('sha'))[:8]}…, on disk {sha[:8]}…). Scores under "
            f"different metrics are never comparable, so the old sign-off is void. "
            f"Demonstrate the new metric on real examples and re-approve it via "
            f"harness.propose_metric(...) or autoprogramming.metric.approve(ws, "
            f"'your name')."
        )
    if os.environ.get(AUTO_APPROVE_ENV, "").strip().lower() in _TRUTHY:
        approve(workspace, f"auto ({AUTO_APPROVE_ENV})")
        return
    raise MetricNotApprovedError(
        "metric.py exists but has not been approved. The entire search optimizes "
        "whatever metric.py says — a wrong metric produces a confidently-scored "
        "wrong program — so scoring is refused until someone signs off. Show the "
        "metric scoring real examples (harness.propose_metric(code, examples) "
        "runs that conversation), or approve directly with "
        "autoprogramming.metric.approve(ws, 'your name'). Unattended runs may "
        f"set {AUTO_APPROVE_ENV}=1 to skip the sign-off deliberately."
    )


def approve(
    workspace,
    approved_by: str,
    weights: dict | None = None,
    primary: str | None = None,
) -> None:
    """Record sign-off for the current metric.py.

    Stamps the file's first line with ``# metric.py — approved by {who} on
    {date}`` (replacing any previous stamp) and writes metric_approval.json
    pinning the resulting sha, so any later edit to the file is detectable.
    ``weights`` records the per-field aggregate weighting for multi-output
    programs — that weighting is part of the sign-off. ``primary`` names which
    quality metric is the headline (ranking/activation driver); it must name a
    metric defined in metric.py (or be None to let :func:`primary_name` choose).
    Both ``primary`` and ``weights`` are config, not code — changing them never
    changes ``metric_sha`` and so never invalidates recorded scores.
    """
    path = Path(workspace.metric_py)
    if not path.exists():
        raise MetricNotApprovedError(
            "There is no metric.py to approve in this workspace. Approval records "
            "a sign-off on a concrete metric file, so an approval without one "
            "would be meaningless — write the metric first (write_metric or "
            "harness.propose_metric), then approve it."
        )
    if primary is not None:
        metrics = quality_metrics(workspace)
        if primary not in metrics:
            raise SchemaError(
                f"primary={primary!r} does not name a quality metric defined in "
                f"metric.py (defined: {sorted(metrics)}). The primary is the "
                f"headline metric that drives ranking and the default activation, "
                f"so it must be one of the metrics being scored. Name a defined "
                f"metric, or pass primary=None to use the sole/first one."
            )
    now = datetime.now(timezone.utc)
    header = f"{_STAMP_PREFIX} {approved_by} on {now:%Y-%m-%d}"
    lines = path.read_text().splitlines()
    if lines and lines[0].startswith(_STAMP_PREFIX):
        lines[0] = header
    else:
        lines.insert(0, header)
    path.write_text("\n".join(lines) + "\n")
    record = {
        "sha": metric_sha(workspace),
        "approved_by": approved_by,
        "approved_at": now.isoformat(),
        "weights": weights,
        "primary": primary,
    }
    Path(workspace.metric_approval).write_text(json.dumps(record, indent=2) + "\n")


def _cost_objective_names() -> tuple[str, ...]:
    """The reserved cost-objective names a quality metric may not shadow."""
    from . import scoring

    return tuple(scoring.COST_OBJECTIVES)


def _validate_metric_names(metrics: dict) -> None:
    """Reject empty, non-callable, or cost-colliding quality metric names."""
    reserved = set(_cost_objective_names())
    for name, fn in metrics.items():
        if not isinstance(name, str) or not name.strip():
            raise SchemaError(
                f"metric.py defines a quality metric with a non-string or empty "
                f"name ({name!r}); every objective needs a real name so its scores "
                f"can be stored and ranked. Give each METRICS key a non-empty name."
            )
        if not callable(fn):
            raise SchemaError(
                f"metric.py's METRICS[{name!r}] is not callable; each quality "
                f"metric must be a function metric(predicted, expected). Point the "
                f"key at a callable."
            )
        if name in reserved:
            raise SchemaError(
                f"metric name {name!r} collides with a built-in cost objective "
                f"({sorted(reserved)}); cost_dollars and latency_s are measured "
                f"automatically off every run, so a quality metric cannot reuse "
                f"those names. Rename the metric."
            )


def _import_metric_module(workspace, path: Path, sha: str):
    spec = importlib.util.spec_from_file_location(f"_ap_metric_{sha[:16]}", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise SchemaError(
            f"metric.py failed to import ({exc!r}). The metric must be a "
            f"self-contained module defining metric(predicted, expected) (or a "
            f"METRICS dict of them); fix the error, then re-approve the file."
        ) from exc
    return module


def quality_metrics(workspace) -> dict[str, Callable]:
    """The workspace's named quality metrics: ``{name: callable}``.

    A single ``def metric(predicted, expected)`` becomes ``{"quality": metric}``;
    a ``METRICS = {"<name>": callable, ...}`` dict is returned as-is (and wins if
    both are present). Cached by (path, sha) so editing the file yields a fresh
    import. Raises SchemaError on bad/colliding names or if neither form exists.
    """
    path = Path(workspace.metric_py)
    sha = metric_sha(workspace)
    if sha is None:
        raise MetricNotApprovedError(
            "Cannot load a metric: this workspace has no metric.py. Write one "
            "with autoprogramming.metric.write_metric(ws, code) or propose one "
            "via harness.propose_metric(code, examples)."
        )
    key = (str(path.resolve()), sha)
    cached = _METRIC_CACHE.get(key)
    if cached is not None:
        return cached
    module = _import_metric_module(workspace, path, sha)
    metrics_attr = getattr(module, "METRICS", None)
    single = getattr(module, "metric", None)
    if metrics_attr is not None:
        if not isinstance(metrics_attr, dict) or not metrics_attr:
            raise SchemaError(
                "metric.py defines METRICS but it is not a non-empty dict of "
                "{name: callable}. The multi-metric form is a dict of named "
                "quality metrics; make it a non-empty dict, or delete it and "
                "define a single def metric(predicted, expected)."
            )
        result = dict(metrics_attr)
    elif callable(single):
        result = {"quality": single}
    else:
        raise SchemaError(
            "metric.py does not define a callable named metric, nor a METRICS "
            "dict of named metrics. The scoring contract is metric(predicted, "
            "expected) returning a float (or a per-field dict for multi-output "
            "programs); define that function (or a METRICS dict) and re-approve "
            "the file."
        )
    _validate_metric_names(result)
    _METRIC_CACHE[key] = result
    return result


def primary_name(workspace) -> str:
    """The approved primary quality metric's name.

    The approval record's ``primary`` if set and it names a defined metric, else
    the sole metric's name, else the first key of METRICS (insertion order).
    """
    metrics = quality_metrics(workspace)
    approval = _read_approval(workspace)
    if approval is not None:
        primary = approval.get("primary")
        if primary in metrics:
            return primary
    return next(iter(metrics))


def load_metric(workspace) -> Callable:
    """The PRIMARY quality metric's callable (back-compat single-metric handle).

    Existing callers (``harness.run``, ``demonstrate``) score against the one
    headline metric; multi-metric evaluation goes through
    :func:`quality_metrics`. Loading does not require approval.
    """
    return quality_metrics(workspace)[primary_name(workspace)]


def demonstrate(workspace, examples: list[tuple]) -> list[dict]:
    """Score (predicted, expected) pairs under EVERY quality metric.

    Works whether or not the metric is approved — this is what feeds the
    sign-off conversation, where the user checks the scores against their
    intuition of "good" before anything optimizes them. Each row carries a
    ``scores`` dict (one entry per quality metric) and a back-compat ``score``
    equal to the primary metric's value.
    """
    metrics = quality_metrics(workspace)
    primary = primary_name(workspace)
    rows: list[dict] = []
    for predicted, expected in examples:
        scores = {name: fn(predicted, expected) for name, fn in metrics.items()}
        rows.append({
            "predicted": predicted,
            "expected": expected,
            "scores": scores,
            "score": scores[primary],
        })
    return rows


def score_pair(
    metric_fn: Callable,
    schema,
    predicted: dict,
    expected: dict,
    weights: dict | None = None,
) -> tuple[float, dict | None]:
    """Score one prediction against one expected row.

    Single-output programs call ``metric_fn(predicted_value, expected_value)``
    and expect a number; per-field scores are None. Multi-output programs call
    ``metric_fn(predicted_dict, expected_dict)`` and expect a dict keyed by
    output names; the aggregate is the weighted mean (``weights`` from the
    approval record, defaulting to 1.0 per field). Returns
    ``(aggregate, per_field_dict_or_None)``.
    """
    names = schema.output_names
    if len(names) == 1:
        name = names[0]
        raw = metric_fn(predicted[name], expected[name])
        try:
            return float(raw), None
        except (TypeError, ValueError) as exc:
            raise SchemaError(
                f"metric() returned {raw!r} for single-output program "
                f"{schema.name!r}; the metric contract for a single output is one "
                f"numeric score, so aggregation stays well-defined. Return a float."
            ) from exc
    raw = metric_fn(dict(predicted), dict(expected))
    if not isinstance(raw, dict):
        raise SchemaError(
            f"metric() returned {type(raw).__name__} for multi-output program "
            f"{schema.name!r}; the metric contract for multiple outputs is a dict "
            f"keyed by output names {list(names)} so every field is scored "
            f"explicitly. Return e.g. {{{names[0]!r}: 0.9, ...}}."
        )
    missing = [n for n in names if n not in raw]
    if missing:
        raise SchemaError(
            f"metric() result is missing scores for {missing!r}; the metric "
            f"contract for multi-output programs is one score per output name "
            f"({list(names)}) so no field silently drops out of the aggregate. "
            f"Add the missing fields."
        )
    per_field = {n: float(raw[n]) for n in names}
    w = {n: float((weights or {}).get(n, 1.0)) for n in names}
    total = sum(w.values())
    if total <= 0:
        raise SchemaError(
            "The approved metric weights sum to zero, so the aggregate score "
            "would be undefined; give at least one output a positive weight "
            "(weights are part of the metric sign-off)."
        )
    aggregate = sum(per_field[n] * w[n] for n in names) / total
    return aggregate, per_field
