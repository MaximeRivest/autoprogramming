"""Planning is analysis-first and falsifiable; evaluation is staged.

The ladder is a breadth checklist, not a quota: a plan must attempt or
consciously skip each feasible family, every host avenue must say what would
disprove it, a cheap probe gates full evaluation, and workers can surface
alternative ideas without acting on them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import autoprogramming as ap
from autoprogramming import metric, scoring
from autoprogramming.analysis import ProblemAnalysis, ensure_analyzed
from autoprogramming.budget import Budget, BudgetLedger
from autoprogramming.harness import AgentHarness
from autoprogramming.pi_backend import _collect_worker_ideas, _task_document
from autoprogramming.pi_rpc import PiResult
from autoprogramming.portfolio import (
    UNCOVERED_TIER_REASON,
    ApproachTier,
    AvenueSpec,
    AvenueState,
    Portfolio,
    PortfolioPolicy,
)
from autoprogramming.research import SearchReport, SearchResult, record_report
from autoprogramming.schema import Schema
from autoprogramming.workspace import Workspace


class Label(str):
    pass


def classify(text: str) -> Label:
    """Classify text."""


ANALYSIS = dict(
    structure="short free text mapped onto a small closed label set; no dynamics",
    knowledge="label definitions are known in prose; no simulator or reference corpus",
    data_regime="tiny and clean; labels are unambiguous for these examples",
    generalization="must hold on unseen phrasings of the same intents",
    evidence="held-out accuracy plus inspection of confusions between labels",
)


def resources(**search) -> ap.Resources:
    options = {"allow_package_installs": False, "allow_model_downloads": False}
    options.update(search)
    return ap.Resources(
        search=ap.SearchResources(**options),
        runtime=ap.RuntimeResources(network=False),
        data=ap.DataPolicy(external_egress=True),
        confirmed=True,
    )


def workspace(tmp_path, n_rows: int = 3) -> Workspace:
    rows = [{"text": f"t{i}", "Label": f"t{i}"} for i in range(n_rows)]
    ws = Workspace.create(
        tmp_path / "hyp_ap", Schema.from_function(classify),
        {"train": rows, "val": rows, "test": rows},
        seed=0, ratios=(0.6, 0.2, 0.2), data_sha="hyp", bootstrap=True,
    )
    ws.resources_json.write_text(json.dumps(resources().to_dict()))
    metric.write_metric(ws, "def metric(p, e):\n    return float(p == e)\n")
    metric.approve(ws, "tester")
    BudgetLedger.start(ws.budget_json, Budget(eval_calls=1000))
    return ws


def researched(ws) -> None:
    for i in range(2):
        record_report(ws, SearchReport(
            query=f"q{i}", searched_at="2026-01-01T00:00:00+00:00",
            results=(SearchResult(title=f"s{i}", url=f"https://example.com/{i}"),),
        ))


def rules_spec(**overrides) -> AvenueSpec:
    base = dict(
        id="cue-rules", tier=ApproachTier.CODE_AND_RULES, title="Cue rules",
        hypothesis="Labels are separable by a few cue words.",
        implementation_brief="Match generalized cue patterns per label.",
        mechanism="cue-word rule table",
        falsifier="if unseen phrasings share no cue words, accuracy collapses",
    )
    base.update(overrides)
    return AvenueSpec(**base)


# ------------------------------------------------------------- analysis gate


def test_problem_analysis_rejects_thin_answers():
    with pytest.raises(ap.ProblemAnalysisError, match="too thin"):
        ProblemAnalysis(structure="text", knowledge="none", data_regime="small",
                        generalization="yes", evidence="accuracy")


def test_analyze_problem_is_persisted_and_gates_planning(tmp_path):
    ws = workspace(tmp_path)
    prg = AgentHarness(ws)
    assert prg.problem_analysis is None
    with pytest.raises(ap.ProblemAnalysisError):
        ensure_analyzed(ws)
    recorded = prg.analyze_problem(
        **ANALYSIS,
        sub_problems=[ap.SubProblem("detect", "spot the cue phrase", needs="pattern")],
    )
    assert recorded["sub_problems"][0]["id"] == "detect"
    assert ws.analysis_json.exists()
    assert ensure_analyzed(ws).sub_problem_ids == ("detect",)


# --------------------------------------------------------- hypothesis contract


def test_host_avenue_needs_a_falsifier_and_known_targets(tmp_path):
    ws = workspace(tmp_path)
    prg = AgentHarness(ws)
    prg.analyze_problem(**ANALYSIS, sub_problems=[ap.SubProblem("detect", "spot cues")])
    researched(ws)
    with pytest.raises(ValueError, match="falsifier"):
        prg.plan_portfolio([rules_spec(falsifier="")])
    with pytest.raises(ValueError, match="targets sub-problems"):
        prg.plan_portfolio([rules_spec(targets=("nope",))])
    plan = prg.plan_portfolio([rules_spec(targets=("detect",))])
    spec = plan["avenues"][0]["spec"]
    assert tuple(spec["targets"]) == ("detect",)
    assert spec["falsifier"].startswith("if unseen")


def test_avenue_spec_validates_expected_value_fields():
    with pytest.raises(ValueError, match="confidence"):
        rules_spec(confidence=1.5)
    with pytest.raises(ValueError, match="expected_cost"):
        rules_spec(expected_cost_dollars=-1)
    spec = rules_spec(expected_quality=0.7, expected_cost_dollars=0.0, confidence=0.4,
                      long_shot=True)
    assert AvenueSpec.from_dict(spec.to_dict()) == spec


# ---------------------------------------------------- ladder as a checklist


def test_plan_refuses_undecided_tiers_but_accepts_reasoned_skips(tmp_path):
    ws = workspace(tmp_path)
    ws.resources_json.write_text(json.dumps(
        resources(allow_package_installs=True).to_dict()
    ))  # tier 6 (classical ML) is now feasible alongside tier 7
    prg = AgentHarness(ws)
    prg.analyze_problem(**ANALYSIS)
    researched(ws)
    with pytest.raises(ValueError, match="undecided.*classical"):
        prg.plan_portfolio([rules_spec()])
    plan = prg.plan_portfolio(
        [rules_spec()], exclusions={6: "three examples cannot fit an estimator"}
    )
    assert plan["exclusions"]["6"] == "three examples cannot fit an estimator"
    # A bare marker skips under the standard reason.
    ws.portfolio_json.unlink()
    plan = prg.plan_portfolio([rules_spec()], exclusions={6: True})
    assert plan["exclusions"]["6"] == UNCOVERED_TIER_REASON


def test_plan_is_not_padded_with_generic_avenues_unless_asked(tmp_path):
    ws = workspace(tmp_path)
    ws.resources_json.write_text(json.dumps(
        resources(allow_package_installs=True).to_dict()
    ))
    prg = AgentHarness(ws)
    prg.analyze_problem(**ANALYSIS)
    researched(ws)
    plan = prg.plan_portfolio([rules_spec()], fill_missing=True)
    ids = [a["spec"]["id"] for a in plan["avenues"]]
    assert "classical-ml" in ids  # explicit opt-in keeps the old padding


def test_single_bet_is_not_a_search():
    only_7 = resources()
    with pytest.raises(ValueError, match="fewer than two"):
        Portfolio.create(
            only_7, [rules_spec()], fill_missing=False,
            policy=PortfolioPolicy(require_wildcard=False),
        )


def test_long_shot_satisfies_the_wildcard_policy():
    long_shot = rules_spec(id="odd-bet", mechanism="phonetic hashing", long_shot=True)
    portfolio = Portfolio.create(
        resources(), [rules_spec(), long_shot], fill_missing=False
    )
    assert [a.spec.id for a in portfolio.avenues] == ["cue-rules", "odd-bet"]
    portfolio = Portfolio.create(resources(), [rules_spec()], fill_missing=False)
    assert [a.spec.id for a in portfolio.avenues] == ["cue-rules", "wildcard"]
    assert portfolio.avenues[-1].spec.long_shot is True


# ------------------------------------------------------------------- probe


def test_probe_runs_a_few_train_rows_without_persisting(tmp_path):
    ws = workspace(tmp_path, n_rows=5)
    prg = AgentHarness(ws)
    good = prg.new_candidate(source=(
        "# /// script\n# [tool.ap]\n# deterministic = true\n# cost_per_call = 0.0\n# ///\n"
        "def predict(text):\n    return text\n"
    ))
    report = prg.probe(good.name, n_rows=2)
    assert (report.n_rows, report.mean, report.errors) == (2, 1.0, [])
    assert not report.all_failed
    assert scoring.load_scores(ws)["candidates"] == {}
    assert BudgetLedger(ws.budget_json).spent["eval_calls"] == 2

    broken = prg.new_candidate(source="def predict(text):\n    raise KeyError('boom')\n")
    report = prg.probe(broken.name, n_rows=3)
    assert report.all_failed and len(report.errors) == 3
    assert "KeyError" in report.errors[0]
    with pytest.raises(ValueError):
        prg.probe(good.name, n_rows=0)


def test_controller_stops_at_probe_when_every_row_fails(tmp_path, monkeypatch):
    """A build that cannot run eight rows must not be charged a full evaluation."""
    ws = workspace(tmp_path, n_rows=5)
    prg = AgentHarness(ws)
    portfolio = Portfolio.create(resources(), [rules_spec()], fill_missing=False)
    state = portfolio.avenues[0]
    path = tmp_path / "portfolio.json"
    calls: list[str] = []
    monkeypatch.setattr(prg, "eval", lambda *a, **k: calls.append("eval"))

    from autoprogramming.pi_backend import PiOrchestratorBackend

    backend = PiOrchestratorBackend(max_implementation_repairs=0)
    root = tmp_path / "avenue"
    root.mkdir()
    ok = backend._evaluate_solution_bundle(
        prg, state, resources(), None, root,
        "def predict(text):\n    raise RuntimeError('no module named torch')\n",
        portfolio, path,
    )
    assert ok is False
    assert calls == []  # never reached the full train/val evaluation
    assert state.failures[0]["kind"] == "candidate-implementation"
    assert "no module named torch" in state.failures[0]["details"][0]
    assert any(note.startswith("probe candidate_0: all runs failed") for note in state.notes)


# ------------------------------------------------------------- worker ideas


def test_worker_ideas_are_collected_not_acted_on(tmp_path):
    state = AvenueState(spec=rules_spec())
    root = tmp_path / "avenue"
    root.mkdir()
    _collect_worker_ideas(state, root)
    assert state.proposed_ideas == []
    (root / "ideas.md").write_text("Hypothesis: embed then kNN.\nFalsifier: ...\n")
    _collect_worker_ideas(state, root)
    _collect_worker_ideas(state, root)  # idempotent
    assert len(state.proposed_ideas) == 1
    assert "kNN" in state.proposed_ideas[0]
    assert AvenueState.from_dict(state.to_dict()).proposed_ideas == state.proposed_ideas


def test_task_document_carries_hypothesis_and_ideas_channel():
    doc = _task_document(
        Schema.from_function(classify),
        rules_spec(targets=("detect",)),
        resources(),
    )
    assert "What would disprove it: if unseen phrasings" in doc
    assert "Targets these parts of the task: detect" in doc
    assert "ideas.md" in doc


# ----------------------------------------------------- headless orchestrator


def test_headless_plan_without_analysis_is_refused(tmp_path, monkeypatch):
    from autoprogramming import pi_backend
    from autoprogramming.errors import RunnerError

    ws = workspace(tmp_path)
    prg = AgentHarness(ws)
    research = [{"query": f"q{i}", "searchedAt": "2026-01-01T00:00:00+00:00",
                 "results": [{"url": f"https://example.com/{i}", "title": "t"}]}
                for i in range(2)]
    text = json.dumps({"avenues": [], "exclusions": {}})
    result = PiResult(text=text, messages=[
        {"role": "toolResult", "toolName": "web_search",
         "content": [{"type": "text", "text": json.dumps(r)}]}
        for r in research
    ])

    class Client:
        def __enter__(self): return self
        def __exit__(self, *_): return None
        def prompt(self, _p): return result

    backend = pi_backend.PiOrchestratorBackend(resources=resources())
    monkeypatch.setattr(backend, "_orchestrator", lambda *a, **k: Client())
    with pytest.raises(RunnerError, match="problem analysis"):
        backend._create_portfolio(prg, resources())
