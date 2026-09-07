"""Bridge from the human-facing Pi session to the trusted portfolio controller.

A live Pi conversation must remain the strategy orchestrator: spawning another
stateless strategy model loses user context and duplicates judgment.  This
backend therefore never makes a strategy/model call. It pauses with the next
host action; the current agent uses ``AgentHarness.web_search()``,
``plan_portfolio()``, and ``orchestrate_portfolio()`` to drive isolated workers.
"""

from __future__ import annotations

import json
import os

from . import metric as metric_mod
from .research import WebResearchError, ensure_researched


def in_live_pi_session() -> bool:
    return bool(os.environ.get("PI_SESSION_ID") and os.environ.get("PI_MODEL"))


class HostOrchestratorBackend:
    """Non-recursive backend for a Pi session already talking to the human."""

    defer_finalization = True
    host_orchestrated = True

    def __init__(self, *, resources=None):
        self.resources = resources

    def run(self, harness, context: dict) -> None:
        ws = harness.workspace
        if not metric_mod.is_approved(ws):
            print(
                "[autoprogramming] the current human-facing Pi session is the "
                "orchestrator; no second strategy agent was created. Propose and "
                "demonstrate the metric suite in this conversation, then approve "
                "it with prg.approve_metric_suite(...)."
            )
            return
        try:
            ensure_researched(ws)
        except WebResearchError as exc:
            print(f"[autoprogramming] {exc}")
            return
        if not ws.portfolio_json.exists():
            print(
                "[autoprogramming] current-source research is recorded. The active "
                "Pi session must now design the approach portfolio and call "
                "prg.plan_portfolio([...]); the library will not spawn a second "
                "orchestrator to do it."
            )
            return
        if context.get("prepare_only") or context.get("mode") == "prepare":
            print(
                "[autoprogramming] host preparation is complete; no implementation "
                "workers were dispatched. Start breadth explicitly from this Pi "
                "session after confirming the search budget."
            )
            return
        print(
            "[autoprogramming] host portfolio is ready. Dispatching its breadth "
            "phase under the trusted controller; strategy remains in this same "
            "Pi session."
        )
        from .pi_backend import PiOrchestratorBackend
        from .resources import Resources

        resources = Resources.from_dict(
            json.loads(ws.resources_json.read_text())
        )
        PiOrchestratorBackend(
            resources=resources,
            host_orchestrated=True,
        ).run(harness, context={
            "mode": "optimize",
            "host_orchestrated": True,
            "host_phase": context.get("host_phase", "breadth"),
            "selected_ids": context.get("selected_ids", []),
        })
