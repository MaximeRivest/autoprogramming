"""Acceptance/diagnostic metric roles and precommitted selection policy."""

from __future__ import annotations

import json

import pytest

from autoprogramming import MetricSuite, SelectionPolicy
from autoprogramming import metric
from autoprogramming.errors import SchemaError
from autoprogramming.objectives import (
    approve_suite,
    meets_floors,
    metric_suite,
    preference_key,
    selection_goals,
)
from autoprogramming.schema import Schema
from autoprogramming.workspace import Workspace


class Label(str):
    pass


def classify(text: str) -> Label:
    """Classify text."""


CODE = (
    "def exact(p, e):\n    return float(p == e)\n"
    "def graded(p, e):\n    return 1.0 if p == e else 0.25\n"
    "METRICS = {'exact': exact, 'graded': graded}\n"
)


def ws(tmp_path):
    rows = [{"text": "a", "Label": "x"}]
    return Workspace.create(
        tmp_path / "classify_ap",
        Schema.from_function(classify),
        {"train": rows, "val": rows, "test": rows},
        seed=0,
        ratios=(0.6, 0.2, 0.2),
        data_sha="x",
        bootstrap=True,
    )


def test_suite_approval_records_roles_and_policy(tmp_path):
    workspace = ws(tmp_path)
    metric.write_metric(workspace, CODE)
    suite = MetricSuite(
        acceptance=("graded",),
        diagnostic=("exact",),
        policy=SelectionPolicy(
            floors={"graded": 0.5},
            preference_order=("graded",),
            max_test_finalists=4,
        ),
    )
    approve_suite(workspace, "tester", suite)
    assert metric.is_approved(workspace)
    assert metric_suite(workspace) == suite
    record = json.loads(workspace.metric_approval.read_text())
    assert record["suite"]["acceptance"] == ["graded"]
    assert record["primary"] == "graded"  # compatibility presentation only


def test_suite_requires_every_metric_to_have_one_role(tmp_path):
    workspace = ws(tmp_path)
    metric.write_metric(workspace, CODE)
    with pytest.raises(SchemaError, match="without a role"):
        approve_suite(workspace, "tester", MetricSuite(acceptance=("exact",)))


def test_suite_roles_cannot_overlap():
    with pytest.raises(SchemaError, match="overlap"):
        MetricSuite(acceptance=("quality",), diagnostic=("quality",))


def test_acceptance_policy_cannot_change_after_val_selection(tmp_path):
    from autoprogramming.errors import DataDisciplineError

    workspace = ws(tmp_path)
    metric.write_metric(workspace, CODE)
    original = MetricSuite(
        acceptance=("graded",), diagnostic=("exact",),
        policy=SelectionPolicy(preference_order=("graded",)),
    )
    approve_suite(workspace, "tester", original)
    scores = json.loads(workspace.scores_json.read_text())
    scores["val_scored"] = ["candidate_0"]
    workspace.scores_json.write_text(json.dumps(scores))
    with pytest.raises(DataDisciplineError, match="precommitted|Cannot change"):
        approve_suite(
            workspace,
            "tester",
            MetricSuite(
                acceptance=("exact",), diagnostic=("graded",),
                policy=SelectionPolicy(preference_order=("exact",)),
            ),
        )
    # Reapproval with unchanged acceptance/policy remains legal.
    approve_suite(workspace, "tester", original)


def test_policy_floor_and_preference():
    suite = MetricSuite(
        acceptance=("semantic", "strict"),
        policy=SelectionPolicy(
            floors={"semantic": 0.8},
            preference_order=("strict", "semantic"),
        ),
    )
    assert meets_floors({"semantic": 0.9, "strict": 0.1}, suite)
    assert not meets_floors({"semantic": 0.7, "strict": 1.0}, suite)
    assert selection_goals(suite) == {
        "semantic": "max",
        "strict": "max",
        "cost_dollars": "min",
        "latency_s": "min",
    }
    better = {"semantic": 0.8, "strict": 0.9, "cost_dollars": 1, "latency_s": 2}
    worse = {"semantic": 1.0, "strict": 0.8, "cost_dollars": 0, "latency_s": 0}
    assert preference_key(better, suite) < preference_key(worse, suite)


def test_prepared_run_approves_recorded_suite(tmp_path, monkeypatch):
    import autoprogramming as ap

    monkeypatch.setenv("AP_PRIVATE_DATA_DIR", str(tmp_path / "private"))

    class ProposalBackend:
        def run(self, harness, context):
            metric.write_metric(harness.workspace, CODE)
            (harness.workspace.root / "metric_proposal.json").write_text(json.dumps({
                "suite": MetricSuite(
                    acceptance=("graded",), diagnostic=("exact",),
                    policy=SelectionPolicy(preference_order=("graded",)),
                ).to_dict(),
                "rationale": "two independent lenses",
            }))

    @ap.program
    def label(text: str) -> Label:
        """Classify text."""

    resources = ap.Resources(
        search=ap.SearchResources(
            max_parallel_agents=1,
            allow_package_installs=False,
            allow_model_downloads=False,
        ),
        runtime=ap.RuntimeResources(network=False),
        data=ap.DataPolicy(external_egress=False),
        confirmed=True,
    )
    rows = [{"text": str(i), "Label": str(i)} for i in range(5)]
    prepared = label.prepare(
        rows, ap.Budget(eval_calls=10), resources=resources,
        workspace=tmp_path / "label_ap", backend=ProposalBackend(),
    )
    assert prepared.metric_proposal["rationale"]
    assert prepared.demonstrate_metrics([("x", "x")])[0]["scores"]["exact"] == 1.0
    assert prepared.approve_metrics("user") is prepared
    assert prepared.workspace is label.workspace
    assert metric_suite(label.workspace).acceptance == ("graded",)


def test_legacy_approval_treats_all_metrics_as_acceptance(tmp_path):
    workspace = ws(tmp_path)
    metric.write_metric(workspace, CODE)
    metric.approve(workspace, "tester", primary="graded")
    suite = metric_suite(workspace)
    assert suite.acceptance == ("exact", "graded")
    assert suite.diagnostic == ()
    assert suite.policy.preference_order[0] == "graded"
