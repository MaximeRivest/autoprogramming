"""Metric-suite roles and precommitted candidate selection policy.

Acceptance metrics define what the user means by good and require sign-off.
Diagnostic metrics help search discover blind spots but cannot silently decide
the final winner. Operational objectives are supplied by the scoring harness.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .errors import DataDisciplineError, SchemaError


@dataclass(frozen=True)
class SelectionPolicy:
    """A selection rule committed before the test split is opened."""

    floors: dict[str, float] = field(default_factory=dict)
    preference_order: tuple[str, ...] = ()
    include_cost: bool = True
    include_latency: bool = True
    max_test_finalists: int = 6

    def __post_init__(self) -> None:
        object.__setattr__(self, "floors", {str(k): float(v) for k, v in self.floors.items()})
        object.__setattr__(self, "preference_order", tuple(self.preference_order))
        if self.max_test_finalists < 1:
            raise SchemaError("max_test_finalists must be at least 1.")


@dataclass(frozen=True)
class MetricSuite:
    """User-approved acceptance lenses plus orchestrator-owned diagnostics."""

    acceptance: tuple[str, ...]
    diagnostic: tuple[str, ...] = ()
    policy: SelectionPolicy = field(default_factory=SelectionPolicy)

    def __post_init__(self) -> None:
        object.__setattr__(self, "acceptance", tuple(self.acceptance))
        object.__setattr__(self, "diagnostic", tuple(self.diagnostic))
        if not self.acceptance:
            raise SchemaError("A metric suite needs at least one acceptance metric.")
        overlap = set(self.acceptance) & set(self.diagnostic)
        if overlap:
            raise SchemaError(
                f"Metric roles overlap for {sorted(overlap)}; each lens is either "
                "acceptance or diagnostic."
            )
        unknown_floor = set(self.policy.floors) - set(self.acceptance)
        if unknown_floor:
            raise SchemaError(
                f"Acceptance floors name non-acceptance metrics: {sorted(unknown_floor)}."
            )
        unknown_preference = set(self.policy.preference_order) - set(self.acceptance)
        if unknown_preference:
            raise SchemaError(
                f"Selection preference names non-acceptance metrics: "
                f"{sorted(unknown_preference)}."
            )

    def to_dict(self) -> dict:
        return {
            "acceptance": list(self.acceptance),
            "diagnostic": list(self.diagnostic),
            "policy": asdict(self.policy),
        }

    @classmethod
    def from_dict(cls, value: dict) -> "MetricSuite":
        return cls(
            acceptance=tuple(value.get("acceptance", ())),
            diagnostic=tuple(value.get("diagnostic", ())),
            policy=SelectionPolicy(**value.get("policy", {})),
        )


def validate_suite(workspace, suite: MetricSuite) -> None:
    from . import metric

    names = set(metric.quality_metrics(workspace))
    assigned = set(suite.acceptance) | set(suite.diagnostic)
    missing = assigned - names
    unassigned = names - assigned
    if missing:
        raise SchemaError(f"Metric suite names undefined metrics: {sorted(missing)}.")
    if unassigned:
        raise SchemaError(
            f"Metric suite leaves metrics without a role: {sorted(unassigned)}. "
            "Classify each as acceptance or diagnostic."
        )


def approve_suite(workspace, approved_by: str, suite: MetricSuite, *, weights=None) -> None:
    """Approve metric code and its roles/selection policy in one record."""
    from . import metric

    validate_suite(workspace, suite)
    scores_path = Path(workspace.scores_json)
    approval_path = Path(workspace.metric_approval)
    if scores_path.exists():
        try:
            scores = json.loads(scores_path.read_text())
        except json.JSONDecodeError:
            scores = {}
        if scores.get("val_scored") and approval_path.exists():
            try:
                existing_record = json.loads(approval_path.read_text())
            except json.JSONDecodeError:
                existing_record = {}
            existing_raw = existing_record.get("suite")
            if not isinstance(existing_raw, dict):
                raise DataDisciplineError(
                    "Cannot introduce a suite selection policy after val has "
                    "already selected candidates. Acceptance roles and the default "
                    "operating-point policy must be committed before search; start "
                    "a fresh workspace."
                )
            existing = MetricSuite.from_dict(existing_raw)
            if (
                existing.acceptance != suite.acceptance
                or existing.policy != suite.policy
            ):
                raise DataDisciplineError(
                    "Cannot change acceptance metrics, floors, or preference order "
                    "after val selection began. Diagnostic lenses may evolve, but "
                    "the final selection policy is precommitted."
                )
    # Keep a compatibility headline for old callers. It is presentation only;
    # suite-aware selection uses the full acceptance vector below.
    primary = (
        suite.policy.preference_order[0]
        if suite.policy.preference_order
        else suite.acceptance[0]
    )
    metric.approve(workspace, approved_by, weights=weights, primary=primary)
    path = Path(workspace.metric_approval)
    record = json.loads(path.read_text())
    record["suite"] = suite.to_dict()
    path.write_text(json.dumps(record, indent=2) + "\n")


def metric_suite(workspace) -> MetricSuite:
    """Approved suite, or a backwards-compatible all-acceptance suite."""
    from . import metric

    path = Path(workspace.metric_approval)
    if path.exists():
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            record = {}
        if isinstance(record.get("suite"), dict):
            suite = MetricSuite.from_dict(record["suite"])
            validate_suite(workspace, suite)
            return suite
    names = tuple(metric.quality_metrics(workspace))
    primary = metric.primary_name(workspace)
    order = (primary, *(n for n in names if n != primary))
    return MetricSuite(
        acceptance=names,
        policy=SelectionPolicy(preference_order=order),
    )


def meets_floors(vector: dict[str, float], suite: MetricSuite) -> bool:
    return all(vector.get(name, float("-inf")) >= floor for name, floor in suite.policy.floors.items())


def selection_goals(suite: MetricSuite) -> dict[str, str]:
    goals = {name: "max" for name in suite.acceptance}
    if suite.policy.include_cost:
        goals["cost_dollars"] = "min"
    if suite.policy.include_latency:
        goals["latency_s"] = "min"
    return goals


def preference_key(vector: dict[str, float], suite: MetricSuite) -> tuple:
    """Lexicographic key for the precommitted default among frontier points."""
    order = suite.policy.preference_order or suite.acceptance
    key: list[float] = [-float(vector.get(name, float("-inf"))) for name in order]
    if suite.policy.include_cost:
        key.append(float(vector.get("cost_dollars", float("inf"))))
    if suite.policy.include_latency:
        key.append(float(vector.get("latency_s", float("inf"))))
    return tuple(key)
