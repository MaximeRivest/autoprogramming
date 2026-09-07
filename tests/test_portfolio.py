"""Deterministic breadth and stopping policy around LLM portfolio proposals."""

from __future__ import annotations

import pytest

import autoprogramming as ap
from autoprogramming.portfolio import (
    ApproachTier,
    AvenueSpec,
    AvenueStatus,
    Portfolio,
    PortfolioPolicy,
)


def resources():
    return ap.Resources(
        search=ap.SearchResources(
            max_parallel_agents=3,
            allow_package_installs=True,
            allow_model_downloads=False,
        ),
        runtime=ap.RuntimeResources(network=False, memory_gb=4),
        data=ap.DataPolicy(external_egress=False),
        confirmed=True,
    )


def spec(name="custom-rules", mechanism="finite state parser"):
    return AvenueSpec(
        id=name,
        tier=ApproachTier.CODE_AND_RULES,
        title="Parser",
        hypothesis="Structure can be parsed directly.",
        implementation_brief="Implement a generalized parser.",
        mechanism=mechanism,
    )


def test_portfolio_fills_every_feasible_tier():
    portfolio = Portfolio.create(resources(), [spec()])
    represented = {int(a.spec.tier) for a in portfolio.avenues}
    feasible = {
        tier for tier, item in resources().feasibility().items()
        if item["feasible"] and tier <= 7
    }
    assert feasible <= represented
    assert set(portfolio.exclusions) >= {
        tier for tier, item in resources().feasibility().items()
        if not item["feasible"] and tier <= 7
    }


def test_pi_subscription_gets_its_own_measured_avenue_beside_raw_api():
    profile = ap.Resources(
        search=ap.SearchResources(
            pi_models=("openai-codex/gpt-5:max",),
            candidate_api_providers=("openai",),
            allow_package_installs=False,
            allow_model_downloads=False,
        ),
        runtime=ap.RuntimeResources(
            network=True, api_providers=("openai",)
        ),
        data=ap.DataPolicy(external_egress=True),
        confirmed=True,
    )
    portfolio = Portfolio.create(profile, [])
    model_calls = [
        avenue for avenue in portfolio.avenues
        if avenue.spec.tier == ApproachTier.SINGLE_MODEL_CALL
    ]
    assert any(avenue.spec.allowed_api_providers for avenue in model_calls)
    pi_avenues = [
        avenue for avenue in model_calls
        if any(
            cap.startswith("pi-model:")
            for cap in avenue.spec.required_capabilities
        )
    ]
    assert len(pi_avenues) == 1
    assert pi_avenues[0].spec.allowed_api_providers == ()
    assert any("subscription" in note for note in pi_avenues[0].spec.deployment_notes)


def test_worker_model_must_come_from_discovered_pi_registry():
    profile = ap.Resources(
        search=ap.SearchResources(
            pi_models=("openai-codex/active:max", "openai-codex/strong"),
            allow_package_installs=False,
            allow_model_downloads=False,
        ),
        runtime=ap.RuntimeResources(network=False),
        data=ap.DataPolicy(external_egress=True),
        confirmed=True,
    )
    strong = AvenueSpec(
        id="strong-rules",
        tier=ApproachTier.CODE_AND_RULES,
        title="Strong worker",
        hypothesis="A stronger reasoning worker may engineer better rules.",
        implementation_brief="Build generalized rules.",
        mechanism="generalized rule synthesis",
        worker_model="openai-codex/strong",
    )
    portfolio = Portfolio.create(profile, [strong])
    selected = next(a for a in portfolio.avenues if a.spec.id == "strong-rules")
    assert selected.spec.worker_model == "openai-codex/strong"

    with pytest.raises(ValueError, match="unavailable Pi worker model"):
        Portfolio.create(
            profile,
            [AvenueSpec(
                id="unknown-worker",
                tier=ApproachTier.CODE_AND_RULES,
                title="Unknown",
                hypothesis="Test validation.",
                implementation_brief="Build rules.",
                mechanism="different rule synthesis",
                worker_model="other/missing",
            )],
        )


def test_duplicate_mechanisms_within_tier_refused():
    with pytest.raises(ValueError, match="repeat"):
        Portfolio.create(
            resources(),
            [spec("one"), spec("two", mechanism="  finite  STATE parser ")],
            fill_missing=False,
            exclusions={i: "not selected" for i in range(1, 7)},
        )


def test_budget_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum"):
        PortfolioPolicy(
            breadth_fraction=0.5,
            deepening_fraction=0.5,
            composition_fraction=0.5,
        )


def test_broken_implementations_do_not_satisfy_breadth_or_finalize():
    portfolio = Portfolio.create(resources(), [spec()])
    assert portfolio.breadth_complete is False
    assert portfolio.may_finalize is False
    for avenue in portfolio.avenues:
        avenue.status = AvenueStatus.FAILED
    portfolio.avenues[0].record_result("candidate_0", {"quality": 0.5}, improved=True)
    assert portfolio.breadth_complete is False
    assert portfolio.may_finalize is False

    # Explicitly unavailable families are conclusive, while the successful
    # family needs a materially different second configuration.
    for avenue in portfolio.avenues[1:]:
        avenue.status = AvenueStatus.INFEASIBLE
    portfolio.avenues[0].record_result(
        "candidate_1", {"quality": 0.6}, improved=True
    )
    assert portfolio.breadth_complete is True
    assert portfolio.may_finalize is True


def test_candidate_journal_is_persisted_and_cleared_by_result(tmp_path):
    portfolio = Portfolio.create(
        resources(), [spec()],
        policy=PortfolioPolicy(min_configs_before_abandon=1),
    )
    state = next(a for a in portfolio.avenues if a.spec.id == "custom-rules")
    for avenue in portfolio.avenues:
        avenue.status = AvenueStatus.INFEASIBLE
    state.begin_candidate("candidate_7")
    assert portfolio.breadth_complete is False
    assert portfolio.may_finalize is False
    path = tmp_path / "portfolio.json"
    portfolio.write(path)

    loaded = Portfolio.load(path)
    recovered = next(a for a in loaded.avenues if a.spec.id == "custom-rules")
    assert recovered.pending_candidate == "candidate_7"
    recovered.record_result("candidate_7", {"quality": 0.8}, improved=True)
    assert recovered.pending_candidate is None
    assert recovered.candidates == ["candidate_7"]
    assert loaded.breadth_complete is True
    assert loaded.may_finalize is True


def test_state_stagnates_after_two_no_progress_rounds():
    portfolio = Portfolio.create(resources(), [spec()], fill_missing=True)
    state = next(a for a in portfolio.avenues if a.spec.id == "custom-rules")
    state.record_result("candidate_0", {"quality": 0.5}, improved=False)
    assert state.status == AvenueStatus.EVALUATED
    state.record_result("candidate_1", {"quality": 0.5}, improved=False)
    assert state.status == AvenueStatus.STAGNANT


def test_portfolio_state_round_trips_for_resume(tmp_path):
    portfolio = Portfolio.create(resources(), [spec()])
    state = next(a for a in portfolio.avenues if a.spec.id == "custom-rules")
    state.record_result("candidate_3", {"quality": 0.75}, improved=True)
    path = tmp_path / "portfolio.json"
    portfolio.write(path)
    loaded = Portfolio.load(path)
    loaded_state = next(a for a in loaded.avenues if a.spec.id == "custom-rules")
    assert loaded.resources == portfolio.resources
    assert loaded.policy == portfolio.policy
    assert loaded_state.candidates == ["candidate_3"]
    assert loaded_state.last_objectives == {"quality": 0.75}


def test_blocked_avenue_needs_human_resolution_before_breadth(tmp_path):
    portfolio = Portfolio.create(resources(), [spec()])
    state = next(a for a in portfolio.avenues if a.spec.id == "custom-rules")
    for avenue in portfolio.avenues:
        avenue.status = AvenueStatus.FAILED
    state.record_blocker("environment-preflight", ["GPU unavailable"])
    assert portfolio.breadth_complete is False
    assert portfolio.may_finalize is False
    assert portfolio.unresolved_blockers == [state]

    path = tmp_path / "portfolio.json"
    portfolio.write(path)
    loaded = Portfolio.load(path)
    loaded.resolve_blocker("custom-rules", "retry", "human")
    retried = next(a for a in loaded.avenues if a.spec.id == "custom-rules")
    assert retried.status == AvenueStatus.PLANNED
    assert retried.blocker is None
    assert retried.human_retry_confirmed is True


def test_human_retry_after_one_good_candidate_resumes_same_family_deepening():
    portfolio = Portfolio.create(resources(), [spec()])
    state = next(a for a in portfolio.avenues if a.spec.id == "custom-rules")
    state.record_result("candidate_0", {"quality": 0.5}, improved=True)
    state.record_blocker("second-config-setup", ["temporary package failure"])
    portfolio.resolve_blocker("custom-rules", "retry", "human-user")
    assert state.status == AvenueStatus.EVALUATED
    assert state.candidates == ["candidate_0"]
    assert state.human_retry_confirmed is True


def test_human_can_confirm_blocked_approach_is_really_unavailable():
    portfolio = Portfolio.create(resources(), [spec()])
    state = next(a for a in portfolio.avenues if a.spec.id == "custom-rules")
    state.record_blocker("candidate-environment-failure", ["credential missing"])
    portfolio.resolve_blocker("custom-rules", "exclude", "human-user")
    assert state.status == AvenueStatus.INFEASIBLE
    assert state.blocker is None
    assert "human-user" in state.notes[-1]


def test_public_exports():
    assert ap.ApproachTier is ApproachTier
    assert ap.AvenueSpec is AvenueSpec
    assert ap.PortfolioPolicy is PortfolioPolicy
