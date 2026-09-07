"""Budgets are explicit and have units.

``Budget(dollars=20)``, ``Budget(eval_calls=2000)``, or ``Budget(minutes=30)``
— combinable; optimization stops when the first limit is hit. Evaluation cost
counts against the budget. There is no default; you must say what you're
willing to spend.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .errors import BudgetError, BudgetExceededError


@dataclass(frozen=True)
class Budget:
    dollars: float | None = None
    eval_calls: int | None = None
    minutes: float | None = None

    def __post_init__(self) -> None:
        if self.dollars is None and self.eval_calls is None and self.minutes is None:
            raise BudgetError(
                "Budget has no default — pass at least one of dollars=, "
                "eval_calls=, or minutes= to say what you're willing to spend."
            )
        for name in ("dollars", "eval_calls", "minutes"):
            v = getattr(self, name)
            if v is not None and (
                not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
            ):
                raise BudgetError(f"Budget {name} must be positive and finite, got {v!r}.")

    def to_dict(self) -> dict:
        return {"dollars": self.dollars, "eval_calls": self.eval_calls, "minutes": self.minutes}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BudgetLedger:
    """Workspace-backed spend tracking (budget.json).

    Every eval run charges here. ``check()`` raises before work starts once
    any limit is hit; ``finalize()`` deliberately does not check — the
    one-time test evaluation happens even on an exhausted budget, it just
    keeps being recorded.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        if self.path.exists():
            self._state = json.loads(self.path.read_text())
        else:
            self._state = None

    # ------------------------------------------------------------- lifecycle

    @classmethod
    def start(cls, path: Path, budget: Budget) -> "BudgetLedger":
        """Set (or replace) the limits. Prior spend is preserved — money
        already burned in this workspace stays spent; the minutes clock
        restarts because a new optimization run is starting."""
        ledger = cls(path)
        spent = (ledger._state or {}).get("spent", {"dollars": 0.0, "eval_calls": 0})
        previous = ledger._state or {}
        ledger._state = {
            "limits": budget.to_dict(),
            "spent": spent,
            "started_at": _utcnow().isoformat(),
        }
        if previous.get("breakdown"):
            ledger._state["breakdown"] = previous["breakdown"]
        ledger._save()
        return ledger

    def _require_state(self) -> dict:
        if self._state is None:
            raise BudgetError(
                f"No budget has been set for this workspace ({self.path}); "
                f"optimize() records one via BudgetLedger.start()."
            )
        return self._state

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._state, indent=2) + "\n")

    # ------------------------------------------------------------ accounting

    def charge(
        self,
        eval_calls: int = 0,
        dollars: float = 0.0,
        *,
        category: str | None = None,
    ) -> None:
        """Record spend, optionally attributing it to a subsystem.

        ``spent`` retains its stable public shape; categorized accounting lives
        beside it under ``breakdown``.  The total dollar limit covers both
        candidate evaluation and optimizer-agent calls.
        """
        state = self._require_state()
        calls = int(eval_calls)
        amount = float(dollars)
        if calls < 0 or not math.isfinite(amount) or amount < 0:
            raise BudgetError(
                f"Budget charges must be non-negative and finite, got "
                f"eval_calls={calls}, dollars={amount}."
            )
        state["spent"]["eval_calls"] += calls
        state["spent"]["dollars"] += amount
        if category:
            bucket = state.setdefault("breakdown", {}).setdefault(
                str(category), {"dollars": 0.0, "eval_calls": 0}
            )
            bucket["dollars"] += amount
            bucket["eval_calls"] += calls
        self._save()

    def reserve_dollars(
        self, dollars: float, *, category: str = "agent"
    ) -> str:
        """Commit dollar headroom before launching an in-flight operation.

        Reservations reduce ``remaining()['dollars']`` and make ``check()``
        refuse additional work, but are not spend. Settle with the actual cost
        after completion or release after a launch failure. A new optimization
        run clears stale reservations because no prior subprocess can remain
        attached to the restarted controller.
        """
        state = self._require_state()
        amount = float(dollars)
        if not math.isfinite(amount) or amount <= 0:
            raise BudgetError(
                f"Budget reservations must be positive and finite, got {dollars!r}."
            )
        limit = state["limits"].get("dollars")
        if limit is not None:
            available = (
                float(limit)
                - float(state["spent"]["dollars"])
                - self._reserved_dollars(state)
            )
            if amount > available + 1e-12:
                raise BudgetExceededError(
                    f"Cannot reserve ${amount:g} for {category}: only "
                    f"${max(0.0, available):g} remains after spend and in-flight "
                    "reservations. Reduce parallelism, increase the budget, or "
                    "resume later with a new explicit budget."
                )
        token = uuid.uuid4().hex
        state.setdefault("reservations", {})[token] = {
            "dollars": amount,
            "category": str(category),
        }
        self._save()
        return token

    def settle_reservation(self, token: str, *, dollars: float) -> None:
        """Replace one reservation with actual categorized spend."""
        state = self._require_state()
        reservation = state.get("reservations", {}).get(token)
        if reservation is None:
            raise BudgetError(f"Unknown or already-settled budget reservation {token!r}.")
        amount = float(dollars)
        if not math.isfinite(amount) or amount < 0:
            raise BudgetError(
                f"Reservation settlement must be non-negative and finite, got {dollars!r}."
            )
        state["reservations"].pop(token)
        state["spent"]["dollars"] += amount
        category = str(reservation.get("category") or "agent")
        bucket = state.setdefault("breakdown", {}).setdefault(
            category, {"dollars": 0.0, "eval_calls": 0}
        )
        bucket["dollars"] += amount
        self._save()

    def release_reservation(self, token: str) -> None:
        """Release headroom after an operation failed before reporting usage."""
        state = self._require_state()
        reservation = state.get("reservations", {}).pop(token, None)
        if reservation is None:
            raise BudgetError(f"Unknown or already-settled budget reservation {token!r}.")
        self._save()

    @staticmethod
    def _reserved_dollars(state: dict) -> float:
        return sum(
            float(item.get("dollars", 0.0))
            for item in state.get("reservations", {}).values()
        )

    @property
    def reservations(self) -> dict[str, dict]:
        state = self._require_state()
        return {
            token: {
                "dollars": float(item.get("dollars", 0.0)),
                "category": str(item.get("category") or "agent"),
            }
            for token, item in state.get("reservations", {}).items()
        }

    @property
    def breakdown(self) -> dict:
        """Spend attribution without changing the limits' unit semantics."""
        state = self._require_state()
        return {
            name: {
                "dollars": float(values.get("dollars", 0.0)),
                "eval_calls": int(values.get("eval_calls", 0)),
            }
            for name, values in state.get("breakdown", {}).items()
        }

    @property
    def limits(self) -> dict:
        return dict(self._require_state()["limits"])

    @property
    def spent(self) -> dict:
        state = self._require_state()
        return {
            "dollars": state["spent"]["dollars"],
            "eval_calls": state["spent"]["eval_calls"],
            "minutes": self.elapsed_minutes(),
        }

    def elapsed_minutes(self) -> float:
        state = self._require_state()
        started = datetime.fromisoformat(state["started_at"])
        return (_utcnow() - started).total_seconds() / 60.0

    def remaining(self) -> dict:
        """Remaining headroom per unit; None where no limit was set."""
        limits, spent = self.limits, self.spent
        result = {
            unit: (None if limits[unit] is None else max(0, limits[unit] - spent[unit]))
            for unit in ("dollars", "eval_calls", "minutes")
        }
        if result["dollars"] is not None:
            result["dollars"] = max(
                0.0,
                result["dollars"]
                - self._reserved_dollars(self._require_state()),
            )
        return result

    def exhausted(self) -> str | None:
        """Name of the first exhausted limit, or None."""
        limits, spent = self.limits, self.spent
        committed_dollars = spent["dollars"] + self._reserved_dollars(
            self._require_state()
        )
        for unit in ("dollars", "eval_calls", "minutes"):
            used = committed_dollars if unit == "dollars" else spent[unit]
            if limits[unit] is not None and used >= limits[unit]:
                return unit
        return None

    def check(self) -> None:
        unit = self.exhausted()
        if unit is not None:
            limits, spent = self.limits, self.spent
            raise BudgetExceededError(
                f"Budget exhausted: {unit} limit {limits[unit]} reached "
                f"(spent {spent[unit]:.4g}). Optimization stops at the first "
                f"limit hit; run finalize() to evaluate the top candidates on "
                f"test and activate a winner, or optimize() again with a new budget."
            )
