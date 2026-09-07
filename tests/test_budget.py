"""Unit tests for autoprogramming.budget (Budget, BudgetLedger)."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone

import pytest

from autoprogramming.budget import Budget, BudgetLedger
from autoprogramming.errors import BudgetError, BudgetExceededError

# ------------------------------------------------------------------ Budget


def test_budget_requires_at_least_one_unit():
    with pytest.raises(BudgetError, match="no default"):
        Budget()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dollars": 0},
        {"dollars": -1},
        {"eval_calls": 0},
        {"eval_calls": -10},
        {"minutes": 0},
        {"minutes": -5.0},
        {"dollars": 5, "eval_calls": -1},
        {"dollars": float("nan")},
        {"minutes": float("inf")},
    ],
)
def test_budget_units_must_be_positive(kwargs):
    with pytest.raises(BudgetError, match="positive"):
        Budget(**kwargs)


def test_budget_units_are_combinable():
    budget = Budget(dollars=5, eval_calls=10)
    assert budget.to_dict() == {"dollars": 5, "eval_calls": 10, "minutes": None}


def test_budget_is_frozen():
    budget = Budget(dollars=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        budget.dollars = 100


# ------------------------------------------------------------------ ledger


@pytest.fixture()
def path(tmp_path):
    return tmp_path / "budget.json"


def test_start_writes_budget_json(path):
    BudgetLedger.start(path, Budget(dollars=2, eval_calls=10))
    state = json.loads(path.read_text())
    assert state["limits"] == {"dollars": 2, "eval_calls": 10, "minutes": None}
    assert state["spent"] == {"dollars": 0.0, "eval_calls": 0}
    assert "started_at" in state


def test_charge_accumulates_and_persists_across_reopen(path):
    ledger = BudgetLedger.start(path, Budget(dollars=10, eval_calls=100))
    ledger.charge(eval_calls=3, dollars=0.5)
    ledger.charge(eval_calls=1)

    reopened = BudgetLedger(path)
    assert reopened.spent["eval_calls"] == 4
    assert reopened.spent["dollars"] == pytest.approx(0.5)

    reopened.charge(dollars=0.25)
    third = BudgetLedger(path)
    assert third.spent["dollars"] == pytest.approx(0.75)
    assert third.spent["eval_calls"] == 4


def test_remaining_and_none_where_no_limit(path):
    ledger = BudgetLedger.start(path, Budget(dollars=2))
    ledger.charge(dollars=0.5)
    remaining = ledger.remaining()
    assert remaining["dollars"] == pytest.approx(1.5)
    assert remaining["eval_calls"] is None
    assert remaining["minutes"] is None


def test_remaining_clamps_at_zero(path):
    ledger = BudgetLedger.start(path, Budget(dollars=2))
    ledger.charge(dollars=5)
    assert ledger.remaining()["dollars"] == 0


def test_exhausted_none_and_check_passes_under_limit(path):
    ledger = BudgetLedger.start(path, Budget(dollars=2, eval_calls=10))
    ledger.charge(eval_calls=9, dollars=1.99)
    assert ledger.exhausted() is None
    ledger.check()


def test_exhausted_and_check_raise_at_limit(path):
    ledger = BudgetLedger.start(path, Budget(eval_calls=2))
    ledger.charge(eval_calls=2)
    assert ledger.exhausted() == "eval_calls"
    with pytest.raises(BudgetExceededError, match="Budget exhausted"):
        ledger.check()


def test_exhausted_reports_first_unit_in_order(path):
    ledger = BudgetLedger.start(path, Budget(dollars=1, eval_calls=1))
    ledger.charge(eval_calls=5, dollars=5)
    assert ledger.exhausted() == "dollars"


def test_check_message_explains_what_to_do(path):
    ledger = BudgetLedger.start(path, Budget(dollars=1))
    ledger.charge(dollars=1)
    with pytest.raises(BudgetExceededError, match="finalize"):
        ledger.check()


def test_minutes_limit_via_persisted_clock(path):
    BudgetLedger.start(path, Budget(minutes=1))
    state = json.loads(path.read_text())
    state["started_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=10)
    ).isoformat()
    path.write_text(json.dumps(state))

    reopened = BudgetLedger(path)
    assert reopened.elapsed_minutes() >= 10
    assert reopened.exhausted() == "minutes"
    with pytest.raises(BudgetExceededError):
        reopened.check()


def test_categorized_spend_preserves_public_shape_and_restart(path):
    ledger = BudgetLedger.start(path, Budget(dollars=10))
    ledger.charge(dollars=1.25, category="agent")
    ledger.charge(eval_calls=2, dollars=0.5, category="candidate")
    assert ledger.spent["dollars"] == pytest.approx(1.75)
    assert set(ledger.spent) == {"dollars", "eval_calls", "minutes"}
    assert ledger.breakdown == {
        "agent": {"dollars": 1.25, "eval_calls": 0},
        "candidate": {"dollars": 0.5, "eval_calls": 2},
    }
    restarted = BudgetLedger.start(path, Budget(dollars=20))
    assert restarted.breakdown == ledger.breakdown


def test_start_preserves_spend_and_restarts_clock(path):
    ledger = BudgetLedger.start(path, Budget(dollars=5))
    ledger.charge(dollars=1.5, eval_calls=2)
    state = json.loads(path.read_text())
    state["started_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=30)
    ).isoformat()
    path.write_text(json.dumps(state))

    restarted = BudgetLedger.start(path, Budget(dollars=10, minutes=5))
    assert restarted.spent["dollars"] == pytest.approx(1.5)
    assert restarted.spent["eval_calls"] == 2
    assert restarted.limits == {"dollars": 10, "eval_calls": None, "minutes": 5}
    assert restarted.elapsed_minutes() < 1
    assert restarted.remaining()["dollars"] == pytest.approx(8.5)


def test_negative_charges_are_refused(path):
    ledger = BudgetLedger.start(path, Budget(dollars=1))
    with pytest.raises(BudgetError, match="non-negative"):
        ledger.charge(dollars=-1)
    with pytest.raises(BudgetError, match="non-negative"):
        ledger.charge(eval_calls=-1)
    with pytest.raises(BudgetError, match="finite"):
        ledger.charge(dollars=float("nan"))


def test_dollar_reservation_reduces_headroom_and_settles_actual_spend(path):
    ledger = BudgetLedger.start(path, Budget(dollars=10))
    token = ledger.reserve_dollars(4, category="agent")
    assert ledger.spent["dollars"] == 0
    assert ledger.remaining()["dollars"] == pytest.approx(6)
    assert ledger.reservations[token]["dollars"] == 4
    with pytest.raises(BudgetExceededError, match="Cannot reserve"):
        ledger.reserve_dollars(7, category="agent")

    ledger.settle_reservation(token, dollars=2.5)
    assert ledger.reservations == {}
    assert ledger.spent["dollars"] == pytest.approx(2.5)
    assert ledger.remaining()["dollars"] == pytest.approx(7.5)
    assert ledger.breakdown["agent"]["dollars"] == pytest.approx(2.5)


def test_reservations_can_be_released_and_stale_ones_clear_on_restart(path):
    ledger = BudgetLedger.start(path, Budget(dollars=5))
    released = ledger.reserve_dollars(2)
    ledger.release_reservation(released)
    assert ledger.remaining()["dollars"] == pytest.approx(5)

    ledger.reserve_dollars(3)
    restarted = BudgetLedger.start(path, Budget(dollars=5))
    assert restarted.reservations == {}
    assert restarted.remaining()["dollars"] == pytest.approx(5)


def test_fully_reserved_budget_blocks_additional_work(path):
    ledger = BudgetLedger.start(path, Budget(dollars=2))
    ledger.reserve_dollars(2)
    assert ledger.exhausted() == "dollars"
    with pytest.raises(BudgetExceededError, match="Budget exhausted"):
        ledger.check()


def test_ledger_refuses_accounting_before_start(path):
    ledger = BudgetLedger(path)
    with pytest.raises(BudgetError, match="No budget"):
        ledger.charge(eval_calls=1)
    with pytest.raises(BudgetError):
        ledger.remaining()
    with pytest.raises(BudgetError):
        ledger.check()
