"""Current-source research is a required host-orchestration phase."""

from __future__ import annotations

import json

import pytest

import autoprogramming as ap
from autoprogramming import metric
from autoprogramming.harness import AgentHarness
from autoprogramming.research import (
    SearchReport,
    SearchResult,
    WebResearchError,
    ensure_researched,
    record_report,
    search_web,
)
from autoprogramming.schema import Schema
from autoprogramming.workspace import Workspace


class Label(str):
    pass


def classify(text: str) -> Label:
    """Classify text."""


def workspace(tmp_path):
    rows = [{"text": "x", "Label": "x"}]
    ws = Workspace.create(
        tmp_path / "research_ap",
        Schema.from_function(classify),
        {"train": rows, "val": rows, "test": rows},
        seed=0,
        ratios=(0.6, 0.2, 0.2),
        data_sha="research",
        bootstrap=True,
    )
    resources = ap.Resources(
        search=ap.SearchResources(
            allow_package_installs=False,
            allow_model_downloads=False,
        ),
        runtime=ap.RuntimeResources(network=False),
        data=ap.DataPolicy(external_egress=True),
        confirmed=True,
    )
    ws.resources_json.write_text(json.dumps(resources.to_dict()))
    metric.write_metric(
        ws,
        "def metric(predicted, expected):\n"
        "    return 1.0 if predicted == expected else 0.0\n",
    )
    metric.approve(ws, "research-test")
    return ws


def test_keyless_web_search_parses_titles_urls_and_snippets(monkeypatch):
    page = b'''<html>
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fnew">Current Model</a>
      <a class="result__snippet">Released this month &amp; benchmarked.</a>
    </html>'''

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): return None
        def read(self, _limit): return page

    monkeypatch.setattr("autoprogramming.research.urlopen", lambda *_a, **_k: Response())
    report = search_web("latest task model", limit=3)
    assert report.results == (
        SearchResult(
            title="Current Model",
            url="https://example.com/new",
            snippet="Released this month & benchmarked.",
        ),
    )


ANALYSIS = dict(
    structure="short free text mapped onto a small closed label set; no dynamics",
    knowledge="label definitions are known in prose; no simulator or reference corpus",
    data_regime="tiny and clean; labels are unambiguous for these examples",
    generalization="must hold on unseen phrasings of the same intents",
    evidence="held-out accuracy plus inspection of confusions between labels",
)


def test_portfolio_plan_requires_analysis_then_two_recorded_web_searches(tmp_path):
    ws = workspace(tmp_path)
    harness = AgentHarness(ws)
    with pytest.raises(ap.ProblemAnalysisError, match="analyze_problem"):
        harness.plan_portfolio([])
    harness.analyze_problem(**ANALYSIS)
    with pytest.raises(WebResearchError, match="two|need 2"):
        harness.plan_portfolio([])

    for index in range(2):
        record_report(ws, SearchReport(
            query=f"query {index}",
            searched_at=f"2026-01-0{index + 1}T00:00:00+00:00",
            results=(SearchResult(
                title=f"source {index}",
                url=f"https://example.com/{index}",
            ),),
        ))
    evidence = ensure_researched(ws)
    assert len(evidence["searches"]) == 2
    # An empty plan is no longer padded with generic per-tier avenues; only
    # the library's long-shot wildcard is added, and one bet is not a search.
    with pytest.raises(ValueError, match="fewer than two"):
        harness.plan_portfolio([])
    rules = ap.AvenueSpec(
        id="keyword-rules", tier=ap.ApproachTier.CODE_AND_RULES,
        title="Keyword rules", hypothesis="Labels are separable by a few cue words.",
        implementation_brief="Match generalized cue patterns per label.",
        mechanism="cue-word rule table",
        falsifier="if unseen phrasings share no cue words, accuracy collapses",
    )
    plan = harness.plan_portfolio([rules])
    assert [a["spec"]["id"] for a in plan["avenues"]] == ["keyword-rules", "wildcard"]
    assert all(
        avenue["spec"]["research_sources"]
        for avenue in plan["avenues"]
    )

    class NoDispatch:
        def run(self, _harness, context):
            assert context["host_orchestrated"] is True

    status = harness.orchestrate_portfolio(
        "breadth", budget=ap.Budget(dollars=7), backend=NoDispatch()
    )
    assert status["may_finalize"] is False
    assert json.loads(ws.budget_json.read_text())["limits"]["dollars"] == 7


def test_web_search_refuses_without_egress_permission(tmp_path):
    ws = workspace(tmp_path)
    resources = ap.Resources(
        search=ap.SearchResources(
            allow_package_installs=False,
            allow_model_downloads=False,
        ),
        runtime=ap.RuntimeResources(network=False),
        data=ap.DataPolicy(external_egress=False),
        confirmed=True,
    )
    ws.resources_json.write_text(json.dumps(resources.to_dict()))
    with pytest.raises(WebResearchError, match="external_egress"):
        AgentHarness(ws).web_search("latest classifier")
