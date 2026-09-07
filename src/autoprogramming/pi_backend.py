"""Trusted controller for resource-aware Pi portfolio search.

Low-level Pi protocol handling lives in :mod:`autoprogramming.pi_rpc` and
implementation-worker isolation/bundling lives in
:mod:`autoprogramming.pi_worker`. This module owns portfolio policy, evaluation,
and finalization.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path

from . import metric as metric_mod
from .adherence import (
    ApproachAudit,
    deterministic_audit,
    semantic_audit_prompt,
)
from .budget import BudgetLedger
from .errors import BudgetExceededError, RunnerError
from .objectives import MetricSuite, SelectionPolicy, approve_suite
from .pi_rpc import ORCHESTRATOR_SYSTEM as _ORCHESTRATOR_SYSTEM
from .pi_rpc import PiResult, PiRpcClient, PiUsage
from .pi_rpc import json_object as _json_object
from .pi_worker import WORKER_SYSTEM as _WORKER_SYSTEM
from .pi_worker import PiWorkerRunner
from .pi_worker import avenue_dir as _avenue_dir
from .pi_worker import materialize_bundle as _materialize_bundle
from .pi_worker import task_document as _task_document
from .pi_worker import worker_env as _worker_env
from .pi_worker import worker_run_dir as _worker_run_dir
from .portfolio import (
    ApproachTier,
    AvenueSpec,
    AvenueStatus,
    Portfolio,
    ensure_avenue_contract,
)
from .remote import (
    RemoteAdmission,
    RemoteExecutor,
    record_candidate_placement,
    use_remote_for_avenue,
)
from .resources import ResourceError, Resources

__all__ = [
    "PiOrchestratorBackend",
    "PiResult",
    "PiRpcClient",
    "PiUsage",
    "PiWorkerRunner",
    "_WORKER_SYSTEM",
    "_json_object",
    "_materialize_bundle",
    "_task_document",
    "_worker_env",
    "_worker_run_dir",
]


def _normalize_metric_suite_proposal(
    proposal: dict, names: tuple[str, ...]
) -> tuple[MetricSuite, list[str]]:
    """Repair stale role names after a critic rewrites ``metric_code``.

    Model output remains a proposal requiring user sign-off, but it must be a
    structurally valid proposal that cannot wedge ``prepare()``. Unknown names
    are dropped, every real metric gets one role, and any repair is recorded for
    review in ``metric_proposal.json``.
    """
    if not names:
        raise RunnerError("Pi metric proposal defined no quality metrics.")
    known = set(names)
    adjustments: list[str] = []

    def known_unique(raw) -> list[str]:
        result: list[str] = []
        for value in raw if isinstance(raw, (list, tuple)) else ():
            name = str(value)
            if name in known and name not in result:
                result.append(name)
        return result

    raw_acceptance = proposal.get("acceptance")
    acceptance = known_unique(raw_acceptance)
    if not acceptance:
        acceptance = [names[0]]
        adjustments.append(
            f"acceptance names did not match revised METRICS; defaulted to {names[0]!r}"
        )
    elif list(raw_acceptance or ()) != acceptance:
        adjustments.append("removed unknown or duplicate acceptance metric names")

    raw_diagnostic = proposal.get("diagnostic")
    diagnostic = [
        name for name in known_unique(raw_diagnostic) if name not in acceptance
    ]
    unassigned = [name for name in names if name not in acceptance and name not in diagnostic]
    if unassigned:
        diagnostic.extend(unassigned)
        adjustments.append(
            f"classified unassigned revised metrics as diagnostic: {unassigned!r}"
        )
    if list(raw_diagnostic or ()) != diagnostic and not unassigned:
        adjustments.append("removed unknown, duplicate, or overlapping diagnostic names")

    raw_preference = proposal.get("preference_order")
    preference = [
        name for name in known_unique(raw_preference) if name in acceptance
    ]
    if not preference:
        preference = list(acceptance)
        adjustments.append("defaulted preference_order to the acceptance metrics")
    elif list(raw_preference or ()) != preference:
        adjustments.append("removed non-acceptance names from preference_order")

    floors: dict[str, float] = {}
    raw_floors = proposal.get("floors")
    if isinstance(raw_floors, dict):
        for raw_name, raw_value in raw_floors.items():
            name = str(raw_name)
            if name not in acceptance:
                adjustments.append(f"dropped floor for non-acceptance metric {name!r}")
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                adjustments.append(f"dropped non-numeric floor for {name!r}")
                continue
            if not math.isfinite(value):
                adjustments.append(f"dropped non-finite floor for {name!r}")
                continue
            floors[name] = value

    return MetricSuite(
        acceptance=tuple(acceptance),
        diagnostic=tuple(diagnostic),
        policy=SelectionPolicy(
            floors=floors,
            preference_order=tuple(preference),
        ),
    ), adjustments


def _execute_pi_calls(
    items,
    *,
    max_workers: int,
    ledger: BudgetLedger,
    reservation_dollars: float | None,
    invoke,
):
    """Run Pi calls with bounded in-flight dollar commitments.

    If a dollar budget exists but no per-call bound was confirmed, calls are
    serialized so only one can overshoot at a time. With a confirmed bound,
    each in-flight call reserves that headroom before launch. New calls are
    submitted only after completed usage has been settled.
    """
    pending = list(items)
    completed: list[tuple[object, object | None, Exception | None]] = []
    undispatched: list[object] = []
    dollar_limited = ledger.limits.get("dollars") is not None
    parallelism = max(1, int(max_workers))
    if dollar_limited and reservation_dollars is None:
        parallelism = 1

    def result_usage_dollars(result) -> float:
        pi_result = result[0] if isinstance(result, tuple) else result
        if not isinstance(pi_result, PiResult):
            raise RunnerError("Internal Pi call did not return a PiResult.")
        return float(pi_result.usage.dollars)

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as pool:
        futures: dict[concurrent.futures.Future, tuple[object, str | None]] = {}
        stopped = False

        def fill_slots() -> None:
            nonlocal stopped
            while pending and len(futures) < parallelism and not stopped:
                item = pending.pop(0)
                token = None
                try:
                    ledger.check()
                    if dollar_limited and reservation_dollars is not None:
                        token = ledger.reserve_dollars(
                            reservation_dollars, category="agent"
                        )
                except BudgetExceededError:
                    pending.insert(0, item)
                    # Existing reservations may be the only reason headroom is
                    # unavailable. Wait for a completion and retry with actual
                    # spend before declaring the remaining calls undispatchable.
                    if futures:
                        return
                    undispatched.extend(pending)
                    pending.clear()
                    stopped = True
                    return
                futures[pool.submit(invoke, item)] = (item, token)

        fill_slots()
        while futures:
            done, _pending_futures = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                item, token = futures.pop(future)
                try:
                    result = future.result()
                    actual = result_usage_dollars(result)
                    if token is None:
                        ledger.charge(dollars=actual, category="agent")
                    else:
                        ledger.settle_reservation(token, dollars=actual)
                    completed.append((item, result, None))
                except Exception as exc:
                    if token is not None:
                        ledger.release_reservation(token)
                    completed.append((item, None, exc))
            fill_slots()

    return completed, undispatched


def _missing_avenue_capabilities(spec: AvenueSpec, resources: Resources) -> list[str]:
    """Controller-side preflight for explicitly required build capabilities."""
    missing: list[str] = []
    candidate_providers = set(resources.search.candidate_api_providers or ())
    for capability in spec.required_capabilities:
        if capability == "package-installs" and resources.search.allow_package_installs is not True:
            missing.append("third-party package installation is not confirmed")
        elif (
            capability == "package-installs"
            and resources.search.remote_compute is None
            and shutil.which("uv") is None
        ):
            missing.append("uv is not installed, so isolated dependencies cannot be resolved")
        elif capability == "model-downloads" and resources.search.allow_model_downloads is not True:
            missing.append("pretrained model downloads are not confirmed")
        elif capability == "fine-tuning" and not resources.search.fine_tuning:
            missing.append("fine-tuning access is not available")
        elif (
            capability == "gpu"
            and resources.search.gpu is None
            and (
                resources.search.remote_compute is None
                or resources.search.remote_compute.gpu is None
            )
        ):
            missing.append("no local or user-provided remote search GPU is recorded")
        elif capability == "runtime-network" and resources.runtime.network is not True:
            missing.append("runtime network access is not confirmed")
        elif capability.startswith("candidate-api:"):
            provider = capability.split(":", 1)[1]
            if provider not in candidate_providers:
                missing.append(
                    f"candidate evaluation has no confirmed {provider!r} provider access"
                )
        elif capability.startswith("pi-model:"):
            model = capability.split(":", 1)[1]
            if shutil.which("pi") is None:
                missing.append(
                    "Pi is not installed, so the authenticated subscription model "
                    f"{model!r} cannot be used by this candidate"
                )
            elif model not in set(resources.search.pi_models):
                missing.append(
                    f"Pi model {model!r} is not in the confirmed authenticated model set"
                )
    lowered_requirements = " ".join(spec.runtime_requirements).lower()
    if (
        any(token in lowered_requirements for token in ("gpu", "cuda"))
        and resources.search.gpu is None
        and (
            resources.search.remote_compute is None
            or resources.search.remote_compute.gpu is None
        )
    ):
        missing.append(
            "this avenue explicitly requires GPU/CUDA but no search-time GPU is recorded"
        )
    return list(dict.fromkeys(missing))


_ENVIRONMENT_ERROR_MARKERS = (
    "api key", "api_key", "apikey", "credential", "authentication", "unauthorized",
    "no module named", "modulenotfounderror", "importerror", "cuda",
    "no gpu", "out of memory", "connection error", "network is unreachable",
    "name or service not known", "model not found", "package not found",
    "no solution found", "failed to resolve", "failed to download",
    "failed to build", "uv was not found", "uv is not installed", "permission denied",
)


def _artifact_tree_sha(root: Path) -> str | None:
    """Stable artifact identity for independent-configuration checks."""
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _environment_blocker_text(text: str) -> list[str]:
    lowered = str(text).lower()
    if any(marker in lowered for marker in _ENVIRONMENT_ERROR_MARKERS):
        compact = " ".join(str(text).split())
        return [compact[-1200:] or "worker reported an unavailable capability"]
    return []


def _web_research_records(result: PiResult) -> list[dict]:
    """Extract auditable search evidence from strategy-agent tool results."""
    records: list[dict] = []
    for message in result.messages:
        if message.get("role") != "toolResult" or message.get("toolName") != "web_search":
            continue
        texts = [
            part.get("text", "")
            for part in message.get("content", [])
            if part.get("type") == "text"
        ]
        for text in texts:
            try:
                value = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(value, dict) and value.get("results"):
                records.append(value)
    return records


def _complete_environment_failure(report) -> list[str]:
    """Return setup errors when every run failed for an environmental reason."""
    expected = int(report.n_rows) * int(report.n_repeats)
    if expected <= 0 or len(report.errors) < expected:
        return []
    details = [str(error) for error in report.errors]
    joined = "\n".join(details).lower()
    if not any(marker in joined for marker in _ENVIRONMENT_ERROR_MARKERS):
        return []
    return details[:8]


class PiOrchestratorBackend:
    """Breadth-first optimizer driven by one Pi orchestrator and Pi workers."""

    def __init__(
        self,
        command: tuple[str, ...] = ("pi",),
        *,
        orchestrator_model: str | None = None,
        worker_model: str | None = None,
        resources: Resources | None = None,
        strict_isolation: bool = False,
        orchestrator_timeout: float = 900.0,
        worker_timeout: float = 1200.0,
        max_compliance_repairs: int = 2,
        max_clean_restarts: int = 1,
        max_implementation_repairs: int = 2,
        host_orchestrated: bool = False,
        semantic_adherence_review: bool = True,
    ):
        if orchestrator_timeout <= 0 or worker_timeout <= 0:
            raise ValueError("Pi timeouts must be positive.")
        if (
            max_compliance_repairs < 0
            or max_clean_restarts < 0
            or max_implementation_repairs < 0
        ):
            raise ValueError("Repair/restart counts cannot be negative.")
        self.command = tuple(command)
        self.orchestrator_model = orchestrator_model
        self.worker_model = worker_model
        self.resources = resources
        self.strict_isolation = strict_isolation
        self.orchestrator_timeout = float(orchestrator_timeout)
        self.worker_timeout = float(worker_timeout)
        self.max_compliance_repairs = int(max_compliance_repairs)
        self.max_clean_restarts = int(max_clean_restarts)
        self.max_implementation_repairs = int(max_implementation_repairs)
        self.host_orchestrated = bool(host_orchestrated)
        self.semantic_adherence_review = bool(semantic_adherence_review)

    def run(self, harness, context: dict) -> None:
        ws = harness.workspace
        resources = self.resources or _load_resources(ws)
        resources.ensure_confirmed()
        if not resources.pi_may_receive_task_data:
            raise ResourceError(
                "This resource profile forbids task-data egress but does not mark "
                "the configured Pi models as local. Pi orchestration and workers "
                "must inspect task context and development examples. Set "
                "search.pi_local=True for an actually local Pi provider, or "
                "explicitly permit external egress; API credentials alone are "
                "never consent."
            )
        if self.strict_isolation:
            raise RunnerError(
                "strict_isolation=True requires the forthcoming OS sandbox worker "
                "adapter; cooperative isolation is available now but must not be "
                "misrepresented as a security boundary."
            )
        if not metric_mod.is_approved(ws):
            if self.host_orchestrated or context.get("host_orchestrated"):
                print(
                    "[autoprogramming] host orchestration paused: the current Pi "
                    "session must propose and obtain approval for the metric suite."
                )
                return
            if not self._propose_metric_suite(harness, resources):
                return
        if context.get("mode") == "prepare" or context.get("prepare_only"):
            return

        portfolio_path = _orchestration_dir(ws) / "portfolio.json"
        if portfolio_path.exists():
            try:
                portfolio = Portfolio.load(portfolio_path)
                if portfolio.resources != resources:
                    raise ValueError("resource profile changed")
            except Exception as exc:
                raise RunnerError(
                    f"Cannot resume portfolio state at {portfolio_path}: {exc}. "
                    "Restore it or start a fresh workspace; silently replacing "
                    "orchestration history would lose budget and coverage state."
                ) from exc
        else:
            if self.host_orchestrated or context.get("host_orchestrated"):
                raise RunnerError(
                    "Host orchestration has no portfolio plan. The human-facing "
                    "Pi session must run prg.web_search(...) and "
                    "prg.plan_portfolio([...]); a second strategy orchestrator "
                    "will not be spawned automatically."
                )
            if resources.data.external_egress is not True:
                raise ResourceError(
                    "Current-source web research is required before portfolio "
                    "planning, but data.external_egress does not permit sending "
                    "task-derived search queries. Ask the human to permit abstract "
                    "research or use a pre-researched, host-authored plan; do not "
                    "fall back to stale model memory."
                )
            portfolio = self._create_portfolio(harness, resources)
            portfolio.write(portfolio_path)
        from .guards import is_bootstrap
        if is_bootstrap(ws) and portfolio.policy.min_configs_before_abandon != 1:
            from dataclasses import replace

            portfolio.policy = replace(
                portfolio.policy, min_configs_before_abandon=1
            )
            portfolio.write(portfolio_path)
        self._recover_pending_avenues(harness, portfolio, portfolio_path)
        # Old/resumed plans and Pi-authored plans cannot weaken a tier contract by
        # omitting its machine-readable constraints.
        for state in portfolio.avenues:
            state.spec = ensure_avenue_contract(state.spec, resources)
            if state.status in (
                AvenueStatus.FAILED,
                AvenueStatus.NONCOMPLIANT,
            ) and state.blocker is None:
                state.record_blocker(
                    "legacy-implementation-failure-needs-human",
                    state.notes[-4:] or [
                        "older controller marked a broken implementation as a "
                        "family outcome; it now requires investigation"
                    ],
                )
        remote_failure = None
        if (
            resources.search.remote_compute is not None
            and any(
                use_remote_for_avenue(state.spec)
                for state in portfolio.avenues
                if state.status != AvenueStatus.INFEASIBLE
            )
        ):
            try:
                remote_executor = RemoteExecutor(resources.search.remote_compute)
                requirements = ["command -v tar", "command -v python3"]
                remote_executor.ssh(" && ".join(requirements))
            except Exception as exc:
                remote_failure = str(exc)
        if remote_failure:
            for state in portfolio.avenues:
                if (
                    not state.candidates
                    and state.status == AvenueStatus.PLANNED
                    and use_remote_for_avenue(state.spec)
                ):
                    state.record_blocker(
                        "remote-compute-preflight",
                        [remote_failure],
                    )
            portfolio.write(portfolio_path)
            self._print_human_blockers(portfolio, portfolio_path)
            return

        human_retried: set[str] = set()
        for state in portfolio.avenues:
            if state.human_retry_confirmed:
                human_retried.add(state.spec.id)
                state.notes.append(
                    "implementation/preflight retried after human confirmation "
                    "that the blocker was fixed or should be attempted again"
                )
                state.human_retry_confirmed = False
                continue
            if state.candidates or state.status != AvenueStatus.PLANNED:
                continue
            missing = _missing_avenue_capabilities(state.spec, resources)
            if missing:
                state.record_blocker("environment-preflight", missing)
        portfolio.write(portfolio_path)

        def worker_for(spec: AvenueSpec) -> PiWorkerRunner:
            remote = (
                resources.search.remote_compute
                if use_remote_for_avenue(spec)
                else None
            )
            return PiWorkerRunner(
                self.command,
                timeout=self.worker_timeout,
                remote_compute=remote,
            )

        worker = PiWorkerRunner(self.command, timeout=self.worker_timeout)
        admission = RemoteAdmission(resources.search.remote_compute)
        runnable_states = [
            a for a in portfolio.avenues
            if not a.candidates
            and a.status in (AvenueStatus.PLANNED, AvenueStatus.READY)
            and a.spec.tier != ApproachTier.COMPOSITION
        ]
        specs = [a.spec for a in runnable_states]
        workers = {spec.id: worker_for(spec) for spec in specs}
        for state in runnable_states:
            state.status = AvenueStatus.RUNNING
        portfolio.write(portfolio_path)

        # Bootstrap mode can expose at most five distinct candidates to val.
        from .guards import BOOTSTRAP_MAX_VAL_CANDIDATES, is_bootstrap
        if is_bootstrap(ws) and len(specs) > BOOTSTRAP_MAX_VAL_CANDIDATES:
            deferred = specs[BOOTSTRAP_MAX_VAL_CANDIDATES:]
            specs = specs[:BOOTSTRAP_MAX_VAL_CANDIDATES]
            for spec in deferred:
                state = next(a for a in portfolio.avenues if a.spec.id == spec.id)
                state.status = AvenueStatus.INFEASIBLE
                state.notes.append(
                    "not dispatched: bootstrap data permits at most five distinct "
                    "val candidates; gather 30+ validated examples for full breadth"
                )
            portfolio.write(portfolio_path)

        results: dict[str, tuple[PiResult, Path]] = {}
        ledger = BudgetLedger(ws.budget_json)
        completed, undispatched = _execute_pi_calls(
            specs,
            max_workers=min(
                resources.search.max_parallel_agents, max(1, len(specs))
            ),
            ledger=ledger,
            reservation_dollars=resources.search.max_dollars_per_agent_call,
            invoke=lambda spec: self._run_with_admission(
                admission,
                spec,
                lambda: self._run_avenue(
                    harness, spec, resources, workers[spec.id],
                    human_retry_confirmed=(spec.id in human_retried),
                ),
            ),
        )
        for spec in undispatched:
            state = next(a for a in portfolio.avenues if a.spec.id == spec.id)
            state.status = AvenueStatus.PLANNED
            state.notes.append("not dispatched: agent dollar headroom is exhausted")
        for spec, item, error in completed:
            state = next(a for a in portfolio.avenues if a.spec.id == spec.id)
            if error is not None:
                state.record_failure(
                    "worker-turn",
                    [str(error)],
                    repairable=True,
                )
                root = _avenue_dir(ws, spec.id)
                repaired = self._repair_worker_output(
                    harness,
                    state,
                    resources,
                    workers[spec.id],
                    root,
                    [str(error)],
                )
                if repaired is None:
                    continue
                repaired_result, _source = repaired
                results[spec.id] = (repaired_result, root)
                state.status = AvenueStatus.READY
                continue
            result, sandbox = item
            results[spec.id] = (result, sandbox)
            state.status = AvenueStatus.READY
            if result.stderr.strip():
                state.notes.append(result.stderr.strip()[-1000:])
        portfolio.write(portfolio_path)

        for spec in specs:
            state = next(a for a in portfolio.avenues if a.spec.id == spec.id)
            item = results.get(spec.id)
            if item is None:
                continue
            pi_result, sandbox = item
            solution = sandbox / "solution.py"
            source = (
                solution.read_text(encoding="utf-8") if solution.exists() else ""
            )
            if not solution.exists() or not re.search(
                r"(?m)^def predict\s*\(", source
            ):
                raw = f"{pi_result.text}\n{pi_result.stderr}\n{source}"
                details = _environment_blocker_text(raw) or [
                    "worker did not create a solution.py with a top-level predict"
                ]
                state.record_failure(
                    "incomplete-worker-output", details, repairable=True
                )
                repaired = self._repair_worker_output(
                    harness,
                    state,
                    resources,
                    workers[spec.id],
                    sandbox,
                    details,
                )
                if repaired is None:
                    portfolio.write(portfolio_path)
                    continue
                pi_result, source = repaired
            try:
                source = self._ensure_adherent_solution(
                    harness, state, resources, workers[spec.id], sandbox,
                    initial_task=(
                        "Implement the function described in task.md using only the "
                        "non-negotiable approach contract. Inspect examples.jsonl, "
                        "create solution.py, test what the environment permits, and "
                        "never substitute another approach family."
                    ),
                )
            except BudgetExceededError:
                portfolio.write(portfolio_path)
                raise
            except Exception as exc:
                # Fail closed: an unavailable auditor cannot turn unreviewed
                # source into a valid avenue or satisfy breadth. Resume later.
                state.record_blocker(
                    "adherence-audit-unavailable",
                    [str(exc)],
                )
                portfolio.write(portfolio_path)
                continue
            if source is None:
                portfolio.write(portfolio_path)
                continue
            self._evaluate_solution_bundle(
                harness,
                state,
                resources,
                workers[spec.id],
                sandbox,
                source,
                portfolio,
                portfolio_path,
            )

        portfolio.write(portfolio_path)
        if portfolio.unresolved_blockers:
            self._print_human_blockers(portfolio, portfolio_path)
            # Ambiguous implementation/environment failures require the human;
            # do not spend on deepening healthy avenues while exclusions are
            # unresolved.
            return
        if not any(a.candidates for a in portfolio.avenues):
            if not portfolio.unresolved_blockers:
                print("[autoprogramming] Pi portfolio produced no evaluable implementations; inspect " + str(portfolio_path))
            return

        # Breadth is complete before exploitation. Give each successful family
        # a materially independent Pi configuration so one brittle implementation
        # cannot dismiss the mechanism.
        host_mode = self.host_orchestrated or context.get("host_orchestrated")
        if host_mode:
            phase = context.get("host_phase", "breadth")
            if phase == "breadth" and not is_bootstrap(ws):
                self._deepen_avenues(
                    harness, portfolio, resources, worker, portfolio_path
                )
            elif phase == "deepen":
                selected = set(context.get("selected_ids") or ())
                self._deepen_avenues(
                    harness, portfolio, resources, worker, portfolio_path,
                    selected_ids=selected,
                )
            elif phase == "compose":
                self._compose_frontier(
                    harness, portfolio, resources, worker, portfolio_path
                )
            portfolio.write(portfolio_path)
            if portfolio.unresolved_blockers:
                self._print_human_blockers(portfolio, portfolio_path)
            # Strategy and finalization remain in the same session that talks to
            # the human. Return all evidence through prg.portfolio_status().
            return

        if not is_bootstrap(ws):
            self._deepen_avenues(harness, portfolio, resources, worker, portfolio_path)
            decision = self._round_decision(harness, portfolio, resources)
            selected = set(decision.get("deepen") or ())
            if selected:
                self._deepen_avenues(
                    harness, portfolio, resources, worker, portfolio_path,
                    selected_ids=selected,
                )
            if decision.get("compose", True):
                self._compose_frontier(harness, portfolio, resources, worker, portfolio_path)

        portfolio.write(portfolio_path)
        if not portfolio.may_finalize:
            if portfolio.unresolved_blockers:
                self._print_human_blockers(portfolio, portfolio_path)
                return
            raise RunnerError(
                "Portfolio policy refused early finalization: feasible breadth or "
                "the required exploration reserve is incomplete. Resume the run "
                f"from {portfolio_path}."
            )
        harness.finalize()

    @staticmethod
    def _run_with_admission(admission, spec, invoke, *, exclusive=False):
        """Hold a target-specific CPU/GPU lease for a whole worker turn."""
        with admission.lease(spec, exclusive=exclusive):
            return invoke()

    def _charged_rpc_prompt(self, ws, resources, prompt: str, *, system: str) -> PiResult:
        """Run one synchronous Pi review call with ordinary budget accounting."""
        ledger = BudgetLedger(ws.budget_json)
        ledger.check()
        token = None
        if (
            ledger.limits.get("dollars") is not None
            and resources.search.max_dollars_per_agent_call is not None
        ):
            token = ledger.reserve_dollars(
                resources.search.max_dollars_per_agent_call, category="agent"
            )
        try:
            with PiRpcClient(
                self.command,
                cwd=ws.root,
                model=(
                    self.orchestrator_model
                    or (
                        resources.search.pi_models[0]
                        if resources.search.pi_models else None
                    )
                ),
                system_prompt=system,
                timeout=self.orchestrator_timeout,
            ) as client:
                result = client.prompt(prompt)
            if token is None:
                ledger.charge(dollars=result.usage.dollars, category="agent")
            else:
                ledger.settle_reservation(token, dollars=result.usage.dollars)
            return result
        except Exception:
            if token is not None:
                ledger.release_reservation(token)
            raise

    def _charged_worker_turn(
        self, ws, resources, worker, root, task: str, *, session_id: str,
        allowed_api_providers: tuple[str, ...],
        model: str | None = None,
    ) -> PiResult:
        ledger = BudgetLedger(ws.budget_json)
        ledger.check()
        token = None
        if (
            ledger.limits.get("dollars") is not None
            and resources.search.max_dollars_per_agent_call is not None
        ):
            token = ledger.reserve_dollars(
                resources.search.max_dollars_per_agent_call, category="agent"
            )
        try:
            model = self.worker_model or model or (
                resources.search.pi_models[0] if resources.search.pi_models else None
            )
            result = worker.run(
                root, task, model=model, session_id=session_id,
                allowed_api_providers=allowed_api_providers,
            )
            if token is None:
                ledger.charge(dollars=result.usage.dollars, category="agent")
            else:
                ledger.settle_reservation(token, dollars=result.usage.dollars)
            return result
        except Exception:
            if token is not None:
                ledger.release_reservation(token)
            raise

    def _audit_solution(self, harness, state, resources, source: str) -> ApproachAudit:
        static = deterministic_audit(state.spec, source)
        state.record_audit(static.to_dict())
        if not static.adherent or not self.semantic_adherence_review:
            return static
        result = self._charged_rpc_prompt(
            harness.workspace,
            resources,
            semantic_audit_prompt(state.spec, source),
            system=(
                "You are an independent implementation-mechanism auditor. You do "
                "not solve the function and never judge its score. Reject any code "
                "path that substitutes another approach when required infrastructure "
                "is absent. Return only the requested JSON object."
            ),
        )
        audit = ApproachAudit.from_dict(_json_object(result.text), reviewer="pi")
        state.record_audit(audit.to_dict())
        return audit

    def _ensure_adherent_solution(
        self, harness, state, resources, worker, root: Path, *, initial_task: str,
    ) -> str | None:
        """Audit, repair, and if necessary clean-restart one avenue.

        Invalid source never reaches ``new_candidate``. A poor but faithful
        implementation may score zero; a cross-tier substitute is not an
        implementation of this avenue at all.
        """
        solution = root / "solution.py"
        repairs = 0
        restarts = 0
        while solution.exists():
            source = solution.read_text(encoding="utf-8")
            audit = self._audit_solution(harness, state, resources, source)
            if audit.adherent:
                return source

            detail = "; ".join(audit.violations) or "mechanism adherence was not demonstrated"
            state.notes.append(f"rejected noncompliant worker output: {detail}")
            if repairs < self.max_compliance_repairs:
                repairs += 1
                task = f"""Your current solution.py was rejected because it violates the
non-negotiable approach contract:
- {chr(10).join(audit.violations) or 'Required mechanism was not demonstrated.'}

Required repair:
{audit.repair_instructions or 'Implement the assigned mechanism as the only answer-producing path.'}

Edit solution.py in place. Do not preserve a fallback for safety. Missing
packages, credentials, GPU, models, or network must produce a precise failure;
they never justify another approach. Re-read task.md, keep the assigned
mechanism, and syntax-check the repaired file."""
                self._run_with_admission(
                    RemoteAdmission(resources.search.remote_compute),
                    state.spec,
                    lambda: self._charged_worker_turn(
                        harness.workspace, resources, worker, root, task,
                        session_id=state.spec.id,
                        allowed_api_providers=state.spec.allowed_api_providers,
                        model=state.spec.worker_model,
                    ),
                    exclusive=True,
                )
                continue

            if restarts < self.max_clean_restarts:
                restarts += 1
                state.restart_count += 1
                repairs = 0
                solution.unlink(missing_ok=True)
                artifact_root = root / "artifacts" / state.spec.id
                if artifact_root.exists():
                    shutil.rmtree(artifact_root)
                task = (
                    initial_task
                    + "\n\nThis is a clean restart because a previous engineer "
                    "substituted another mechanism. Start from task.md; do not "
                    "recreate or retain any cross-family fallback."
                )
                self._run_with_admission(
                    RemoteAdmission(resources.search.remote_compute),
                    state.spec,
                    lambda: self._charged_worker_turn(
                        harness.workspace, resources, worker, root, task,
                        session_id=f"{state.spec.id}-compliance-restart-{restarts}",
                        allowed_api_providers=state.spec.allowed_api_providers,
                        model=state.spec.worker_model,
                    ),
                    exclusive=True,
                )
                continue

            state.record_blocker(
                "implementation-noncompliance-needs-human",
                [
                    "avenue exhausted compliance repairs/restarts; no faithful "
                    "source was imported or evaluated"
                ],
            )
            return None
        state.record_blocker(
            "implementation-noncompliance-needs-human",
            ["worker did not leave solution.py after compliance repair"],
        )
        return None

    def _diagnose_suspicious_candidate(
        self, harness, state, resources, source: str, train_report
    ) -> dict:
        """Independently inspect a zero-quality but runnable implementation."""
        trace = ""
        try:
            traced = harness.run(
                train_report.candidate, split="train", row=0
            )
            trace = str(traced)
        except Exception as exc:
            trace = f"trace unavailable: {exc}"
        prompt = f"""Act as an implementation-debugging reviewer, not a strategy
planner. This faithful avenue ran but produced a suspicious zero aggregate on
its development rows. Decide whether an implementation mistake is likely before
the mechanism is judged. Inspect dependency/model choice, preprocessing,
input/output contract, parsing, device behavior, and obvious placeholder logic.
Do not propose another approach family.

Return JSON only:
{{"implementation_issue_likely": true, "findings": ["..."],
  "repair_instructions": "..."}}

Approach contract:
{json.dumps(state.spec.to_dict(), indent=2)}
Schema:
{harness.schema.describe()}
One development trace:
{trace[-6000:]}
Source:
---
{source}
---
"""
        result = self._charged_rpc_prompt(
            harness.workspace,
            resources,
            prompt,
            system=(
                "You are an independent implementation debugger. Diagnose the "
                "assigned implementation only; never plan or substitute another "
                "approach. Return only the requested JSON object."
            ),
        )
        diagnosis = _json_object(result.text)
        state.audits.append({"reviewer": "implementation-debugger", **diagnosis})
        return diagnosis

    def _repair_worker_output(
        self,
        harness,
        state,
        resources,
        worker,
        root: Path,
        details: list[str],
    ) -> tuple[PiResult, str] | None:
        """Give implementation failures bounded same-family repair attempts.

        The worker may change dependencies, model variants, batching, device
        placement, parsing, or other engineering details, but the avenue's hard
        mechanism contract remains fixed. Exhaustion becomes a human blocker,
        never evidence that the approach family failed.
        """
        last_details = [str(v) for v in details]
        for attempt in range(1, self.max_implementation_repairs + 1):
            task = f"""The implementation did not establish that the assigned
mechanism works. Investigate and repair the implementation; do not pivot to
another family.

Observed evidence:
- {chr(10).join(last_details[:12])}

Adapt creatively within the same mechanism: verify dependency declarations and
versions, inspect setup, choose a compatible variant of the assigned model or
algorithm, fix input/output handling, batching, device placement, and resource
use. An absent raw API-key environment variable is not a blocker when task.md
lists an authenticated Pi model; use Pi's OAuth-backed CLI/RPC runtime. If GPU
memory is temporarily occupied, reduce safe batch/cache use and leave a precise
error so the controller can reschedule exclusively. Edit solution.py materially,
syntax-check it, and never add a cross-family fallback."""
            try:
                result = self._run_with_admission(
                    RemoteAdmission(resources.search.remote_compute),
                    state.spec,
                    lambda: self._charged_worker_turn(
                        harness.workspace,
                        resources,
                        worker,
                        root,
                        task,
                        session_id=(
                            f"{state.spec.id}-implementation-repair-{attempt}"
                        ),
                        allowed_api_providers=state.spec.allowed_api_providers,
                        model=state.spec.worker_model,
                    ),
                    exclusive=True,
                )
            except BudgetExceededError:
                raise
            except Exception as exc:
                last_details = [*last_details, f"repair turn {attempt} failed: {exc}"]
                state.record_failure(
                    "implementation-repair-turn",
                    [str(exc)],
                    repairable=True,
                )
                continue
            solution = root / "solution.py"
            if not solution.exists():
                last_details = [f"repair turn {attempt} left no solution.py"]
                continue
            source = solution.read_text(encoding="utf-8")
            if not re.search(r"(?m)^def predict\s*\(", source):
                last_details = [f"repair turn {attempt} left no top-level predict"]
                continue
            try:
                source = self._ensure_adherent_solution(
                    harness,
                    state,
                    resources,
                    worker,
                    root,
                    initial_task=(
                        "Repair the implementation failure while preserving the "
                        "exact assigned mechanism."
                    ),
                )
            except Exception as exc:
                last_details = [f"repair adherence review failed: {exc}"]
                continue
            if source is not None:
                return result, source

        # Repairs of one file are not enough evidence to dismiss a mechanism.
        # Make a bounded clean, independent implementation attempt with a fresh
        # Pi context before escalating to the human.
        for restart in range(1, self.max_clean_restarts + 1):
            solution = root / "solution.py"
            solution.unlink(missing_ok=True)
            artifact_root = root / "artifacts" / state.spec.id
            if artifact_root.exists():
                shutil.rmtree(artifact_root)
            state.restart_count += 1
            task = f"""Start a materially independent implementation of the exact
same approach contract in task.md. Earlier implementation/repair attempts failed
for these reasons:
- {chr(10).join(last_details[:12])}

Do not recover or imitate the deleted solution. Reconsider dependency and model
variants, setup, preprocessing, batching, parsing, and device use while keeping
the assigned mechanism as the only answer-producing path. Create solution.py,
test what is possible, syntax-check it, and never add a cross-family fallback."""
            try:
                result = self._run_with_admission(
                    RemoteAdmission(resources.search.remote_compute),
                    state.spec,
                    lambda: self._charged_worker_turn(
                        harness.workspace,
                        resources,
                        worker,
                        root,
                        task,
                        session_id=(
                            f"{state.spec.id}-independent-restart-{restart}"
                        ),
                        allowed_api_providers=state.spec.allowed_api_providers,
                        model=state.spec.worker_model,
                    ),
                    exclusive=True,
                )
                if not solution.exists():
                    last_details.append(
                        f"independent restart {restart} left no solution.py"
                    )
                    continue
                source = solution.read_text(encoding="utf-8")
                source = self._ensure_adherent_solution(
                    harness,
                    state,
                    resources,
                    worker,
                    root,
                    initial_task=(
                        "Build an independent faithful implementation of task.md."
                    ),
                )
                if source is not None:
                    state.notes.append(
                        "recovered through a clean independent implementation restart"
                    )
                    return result, source
            except BudgetExceededError:
                raise
            except Exception as exc:
                last_details.append(
                    f"independent restart {restart} failed: {exc}"
                )
                state.record_failure(
                    "independent-implementation-restart",
                    [str(exc)],
                    repairable=True,
                )
        state.record_blocker(
            "implementation-failure-needs-human",
            last_details,
        )
        return None

    def _evaluate_solution_bundle(
        self,
        harness,
        state,
        resources,
        worker,
        root: Path,
        source: str,
        portfolio,
        portfolio_path,
        *,
        baseline_name: str | None = None,
    ) -> bool:
        """Import/evaluate with repair loops; never equate a broken build to a tier."""
        from .candidates import next_name

        ws = harness.workspace
        current_source = source
        admission = RemoteAdmission(resources.search.remote_compute)
        for attempt in range(self.max_implementation_repairs + 1):
            candidate_name = None
            failure_details: list[str] = []
            try:
                expected_name = next_name(ws)
                state.begin_candidate(expected_name)
                portfolio.write(portfolio_path)
                materialized = _materialize_bundle(
                    current_source, root, ws, state.spec.id
                )
                cand = harness.new_candidate(source=materialized)
                candidate_name = cand.name
                record_candidate_placement(
                    ws,
                    cand.name,
                    (
                        "remote"
                        if (
                            resources.search.remote_compute is not None
                            and use_remote_for_avenue(state.spec)
                        )
                        else "local"
                    ),
                )
                if cand.name != expected_name:
                    raise RunnerError(
                        f"Candidate journal expected {expected_name}, got {cand.name}."
                    )
                with admission.lease(state.spec, exclusive=(attempt > 0)):
                    train = harness.eval(
                        cand.name, split="train", per_instance=True
                    )
                    expected_runs = int(train.n_rows) * int(train.n_repeats)
                    if expected_runs > 0 and len(train.errors) >= expected_runs:
                        failure_details = list(train.errors[:12])
                    elif float(train.mean) <= 0.0:
                        diagnosis = self._diagnose_suspicious_candidate(
                            harness,
                            state,
                            resources,
                            materialized,
                            train,
                        )
                        if diagnosis.get("implementation_issue_likely") is not False:
                            findings = diagnosis.get("findings") or []
                            if isinstance(findings, str):
                                findings = [findings]
                            failure_details = [str(v) for v in findings]
                            repair = str(
                                diagnosis.get("repair_instructions") or ""
                            ).strip()
                            if repair:
                                failure_details.append(repair)
                            if not failure_details:
                                failure_details = [
                                    "independent debugger could not clear the "
                                    "suspicious zero-quality implementation"
                                ]
                        else:
                            val = harness.eval(cand.name)
                    else:
                        val = harness.eval(cand.name)
                if not failure_details:
                    improved = True
                    comparison = None
                    if baseline_name is not None:
                        comparison = harness.compare(baseline_name, cand.name)
                        improved = bool(comparison.improved)
                    state.record_result(
                        cand.name,
                        {
                            name: float(obj["mean"])
                            for name, obj in val.objectives.items()
                        },
                        improved=improved,
                    )
                    state.notes.append(
                        f"train aggregate {train.mean:.4g}; val aggregate stored privately"
                    )
                    if comparison is not None:
                        state.notes.append(str(comparison))
                    portfolio.write(portfolio_path)
                    return True
            except BudgetExceededError:
                portfolio.write(portfolio_path)
                raise
            except Exception as exc:
                failure_details = [str(exc)]

            state.pending_candidate = None
            state.record_failure(
                "candidate-implementation",
                failure_details or ["candidate failed without diagnostic output"],
                candidate=candidate_name,
                repairable=True,
            )
            portfolio.write(portfolio_path)
            if attempt >= self.max_implementation_repairs:
                state.record_blocker(
                    "implementation-or-environment-needs-human",
                    failure_details or ["candidate repeatedly failed"],
                    candidate=candidate_name,
                )
                portfolio.write(portfolio_path)
                return False
            repaired = self._repair_worker_output(
                harness,
                state,
                resources,
                worker,
                root,
                failure_details,
            )
            if repaired is None:
                portfolio.write(portfolio_path)
                return False
            _result, current_source = repaired
        return False

    @staticmethod
    def _print_human_blockers(portfolio, portfolio_path) -> None:
        print(
            "[autoprogramming] one or more assigned approaches are completely "
            "blocked by environment/setup failures. They have NOT been replaced "
            "with fallback approaches and will not be discarded without human "
            "confirmation:"
        )
        for avenue in portfolio.unresolved_blockers:
            details = "; ".join((avenue.blocker or {}).get("details", []))
            print(f"  - {avenue.spec.id} ({avenue.spec.title}): {details}")
        print(
            "Fix the capability, then confirm a retry; or explicitly confirm that "
            "the approach is unavailable:\n"
            "    import autoprogramming as ap\n"
            "    prg = ap.attach(<workspace>)\n"
            "    prg.resolve_blocker('<avenue-id>', 'retry', confirmed_by='user')\n"
            "    # or, only after the user agrees:\n"
            "    prg.resolve_blocker('<avenue-id>', 'exclude', confirmed_by='user')\n"
            f"Then resume optimize(). State: {portfolio_path}"
        )

    def _recover_pending_avenues(self, harness, portfolio, portfolio_path) -> None:
        """Finish an import/evaluation journal entry after controller restart."""
        from . import scoring
        from .candidates import load_candidate

        for avenue in portfolio.avenues:
            name = avenue.pending_candidate
            if not name:
                continue
            try:
                load_candidate(harness.workspace, name)
            except Exception as exc:
                avenue.pending_candidate = None
                avenue.status = AvenueStatus.PLANNED
                avenue.notes.append(f"discarded missing pending candidate {name}: {exc}")
                portfolio.write(portfolio_path)
                continue
            try:
                scores = scoring.load_scores(harness.workspace)
                sub = scores.get("candidates", {}).get(name, {}).get("val")
                if (
                    isinstance(sub, dict)
                    and scoring.score_provenance_current(
                        harness.workspace, name, "val"
                    )
                ):
                    objectives = {
                        objective: float(stats["mean"])
                        for objective, stats in sub.get("objectives", {}).items()
                    }
                else:
                    harness.eval(name, split="train", per_instance=True)
                    val = harness.eval(name)
                    objectives = {
                        objective: float(stats["mean"])
                        for objective, stats in val.objectives.items()
                    }
                improved = True
                if avenue.candidates:
                    try:
                        improved = bool(
                            harness.compare(avenue.candidates[-1], name).improved
                        )
                    except Exception:
                        improved = True
                avenue.record_result(name, objectives, improved=improved)
                avenue.notes.append(
                    "recovered candidate evaluation journal after controller restart"
                )
            except BudgetExceededError:
                portfolio.write(portfolio_path)
                raise
            except Exception as exc:
                avenue.pending_candidate = None
                if name not in avenue.candidates:
                    avenue.candidates.append(name)
                avenue.rounds += 1
                avenue.record_blocker(
                    "pending-candidate-recovery-needs-human",
                    [str(exc)],
                    candidate=name,
                )
            portfolio.write(portfolio_path)

    def _round_decision(self, harness, portfolio, resources) -> dict:
        try:
            BudgetLedger(harness.workspace.budget_json).check()
        except BudgetExceededError:
            return {"deepen": [], "compose": False, "rationale": "budget exhausted"}
        prompt = f"""Allocate the last exploitation round of this breadth-first
portfolio. You may deepen avenues with plausible headroom and request one
cross-family composition. Do not write code. Return JSON:
{{"deepen": [avenue ids], "compose": true, "rationale": "..."}}
Choose at most {max(1, resources.search.max_parallel_agents)} avenues. An avenue
with acceptable numbers is not a reason to ignore a different mechanism.
Portfolio state (aggregate vectors only):\n{json.dumps(portfolio.to_dict(), default=str)}"""
        try:
            with self._orchestrator(harness.workspace) as client:
                result = client.prompt(prompt)
            BudgetLedger(harness.workspace.budget_json).charge(
                dollars=result.usage.dollars, category="agent"
            )
            decision = _json_object(result.text)
            valid = {a.spec.id for a in portfolio.avenues}
            decision["deepen"] = [
                name for name in decision.get("deepen", []) if name in valid
            ][: resources.search.max_parallel_agents]
            return decision
        except Exception as exc:
            return {"deepen": [], "compose": True, "rationale": f"fallback: {exc}"}

    def _deepen_avenues(
        self, harness, portfolio, resources, _worker, portfolio_path,
        *, selected_ids: set[str] | None = None,
    ) -> None:
        ws = harness.workspace
        active = [
            avenue for avenue in portfolio.avenues
            if avenue.candidates
            and (selected_ids is None or avenue.spec.id in selected_ids)
            and avenue.rounds < avenue.spec.max_rounds
            and avenue.status not in (
                AvenueStatus.FAILED, AvenueStatus.INFEASIBLE,
                AvenueStatus.BLOCKED, AvenueStatus.NONCOMPLIANT,
            )
        ]
        if not active:
            return
        try:
            BudgetLedger(ws.budget_json).check()
        except BudgetExceededError:
            return

        admission = RemoteAdmission(resources.search.remote_compute)
        avenue_workers = {
            avenue.spec.id: PiWorkerRunner(
                self.command,
                timeout=self.worker_timeout,
                remote_compute=(
                    resources.search.remote_compute
                    if use_remote_for_avenue(avenue.spec)
                    else None
                ),
            )
            for avenue in active
        }

        def deepen(avenue):
            root = _avenue_dir(ws, avenue.spec.id)
            task = (
                "Re-open task.md and the current solution.py. This is a second "
                "engineering pass that must push the SAME non-negotiable mechanism, "
                "not merely solve the function. Find weaknesses in generalization, "
                "edge cases, parsing, failure handling, startup, and repeated-call "
                "efficiency. Improve solution.py materially. Never add a fallback "
                "from another family for missing packages, credentials, GPU, model, "
                "or network; fail clearly instead. Syntax-check it."
            )
            model = self.worker_model or avenue.spec.worker_model or (
                resources.search.pi_models[0] if resources.search.pi_models else None
            )
            # A fresh Pi context supplies a materially independent engineering
            # configuration while its only durable context remains this avenue's
            # own files. GPU-heavy avenues hold an exclusive/default remote lease.
            session = f"{avenue.spec.id}-configuration-{avenue.rounds + 1}"
            return self._run_with_admission(
                admission,
                avenue.spec,
                lambda: avenue_workers[avenue.spec.id].run(
                    root, task, model=model, session_id=session,
                    allowed_api_providers=avenue.spec.allowed_api_providers,
                ),
            )

        ledger = BudgetLedger(ws.budget_json)
        completed, undispatched = _execute_pi_calls(
            active,
            max_workers=min(resources.search.max_parallel_agents, len(active)),
            ledger=ledger,
            reservation_dollars=resources.search.max_dollars_per_agent_call,
            invoke=deepen,
        )
        for avenue in undispatched:
            avenue.notes.append(
                "refinement not dispatched: agent dollar headroom is exhausted"
            )

        from .candidates import load_candidate
        for avenue, result, error in completed:
            root = _avenue_dir(ws, avenue.spec.id)
            solution = root / "solution.py"
            if error is not None:
                avenue.record_failure(
                    "independent-configuration-turn",
                    [str(error)],
                    repairable=True,
                )
                repaired = self._repair_worker_output(
                    harness, avenue, resources,
                    avenue_workers[avenue.spec.id], root, [str(error)]
                )
                if repaired is None:
                    portfolio.write(portfolio_path)
                    continue
                _result, source = repaired
            elif not solution.exists():
                details = ["independent configuration left no solution.py"]
                avenue.record_failure(
                    "independent-configuration-output", details, repairable=True
                )
                repaired = self._repair_worker_output(
                    harness, avenue, resources,
                    avenue_workers[avenue.spec.id], root, details
                )
                if repaired is None:
                    portfolio.write(portfolio_path)
                    continue
                _result, source = repaired
            else:
                source = solution.read_text(encoding="utf-8")
            try:
                source = self._ensure_adherent_solution(
                    harness, avenue, resources,
                    avenue_workers[avenue.spec.id], root,
                    initial_task=(
                        "Re-implement the function in task.md as a materially "
                        "improved configuration of the exact same non-negotiable "
                        "approach. Never substitute another family."
                    ),
                )
            except BudgetExceededError:
                portfolio.write(portfolio_path)
                return
            except Exception as exc:
                avenue.record_blocker(
                    "refinement-audit-needs-human", [str(exc)]
                )
                portfolio.write(portfolio_path)
                continue
            if source is None:
                portfolio.write(portfolio_path)
                continue
            baseline = load_candidate(ws, avenue.candidates[-1])
            baseline_source = baseline.source
            versioned_namespace = f"{avenue.spec.id}-{baseline.name}"
            baseline_source = baseline_source.replace(
                f'"{versioned_namespace}"', f'"{avenue.spec.id}"'
            ).replace(
                f"'{versioned_namespace}'", f"'{avenue.spec.id}'"
            )
            baseline_sha = hashlib.sha256(baseline_source.encode()).hexdigest()
            baseline_artifacts = _artifact_tree_sha(
                ws.artifacts_dir / versioned_namespace
            )
            current_artifacts = _artifact_tree_sha(
                root / "artifacts" / avenue.spec.id
            )
            if (
                baseline_sha == hashlib.sha256(source.encode()).hexdigest()
                and baseline_artifacts == current_artifacts
            ):
                details = [
                    "independent configuration made no material source change; "
                    "this does not count as a second implementation"
                ]
                avenue.record_failure(
                    "non-independent-configuration", details, repairable=True
                )
                repaired = self._repair_worker_output(
                    harness, avenue, resources,
                    avenue_workers[avenue.spec.id], root, details
                )
                if repaired is None:
                    portfolio.write(portfolio_path)
                    continue
                _result, source = repaired
                current_artifacts = _artifact_tree_sha(
                    root / "artifacts" / avenue.spec.id
                )
                if (
                    baseline_sha == hashlib.sha256(source.encode()).hexdigest()
                    and baseline_artifacts == current_artifacts
                ):
                    avenue.record_blocker(
                        "independent-configuration-needs-human", details
                    )
                    portfolio.write(portfolio_path)
                    continue
            self._evaluate_solution_bundle(
                harness,
                avenue,
                resources,
                avenue_workers[avenue.spec.id],
                root,
                source,
                portfolio,
                portfolio_path,
                baseline_name=baseline.name,
            )

    def _compose_frontier(self, harness, portfolio, resources, _worker, portfolio_path) -> None:
        tradeoffs = harness.tradeoffs()
        if len(tradeoffs.nondominated) < 2:
            return
        if any(a.spec.id == "frontier-composition" for a in portfolio.avenues):
            return
        try:
            BudgetLedger(harness.workspace.budget_json).check()
        except BudgetExceededError:
            return
        from .candidates import load_candidate

        names = tradeoffs.nondominated[:3]
        spec = ensure_avenue_contract(AvenueSpec(
            id="frontier-composition",
            tier=ApproachTier.COMPOSITION,
            title="Complementary implementation composition",
            hypothesis="The supplied mechanisms have complementary strengths that a bounded router or cascade can combine.",
            implementation_brief=(
                "Study component_*.py and build one self-contained solution.py that "
                "combines them as a router, cascade, pipeline, or ensemble. Inline "
                "everything needed; do not import component files at runtime."
            ),
            mechanism="task-specific bounded router or cascade over frontier mechanisms",
            compose_from=tuple(names),
            max_rounds=1,
        ), resources)
        root = _avenue_dir(harness.workspace, spec.id)
        root.mkdir(parents=True, exist_ok=True)
        (root / "task.md").write_text(_task_document(harness.schema, spec, resources))
        from .data import development_partition
        fit_rows, _probe = development_partition(list(harness.data.train))
        with (root / "examples.jsonl").open("w", encoding="utf-8") as fh:
            for row in fit_rows:
                fh.write(json.dumps(row, default=str) + "\n")
        for i, name in enumerate(names):
            (root / f"component_{i}.py").write_text(load_candidate(harness.workspace, name).source)
        from .portfolio import AvenueState
        composed = AvenueState(spec=spec)
        composition_worker = PiWorkerRunner(
            self.command,
            timeout=self.worker_timeout,
            remote_compute=(
                resources.search.remote_compute
                if use_remote_for_avenue(spec)
                else None
            ),
        )
        portfolio.avenues.append(composed)
        portfolio.write(portfolio_path)
        try:
            admission = RemoteAdmission(resources.search.remote_compute)
            self._run_with_admission(
                admission,
                spec,
                lambda: self._charged_worker_turn(
                    harness.workspace,
                    resources,
                    composition_worker,
                    root,
                    "Build the self-contained composition described in task.md "
                    "from the supplied component files. Create and syntax-check "
                    "solution.py.",
                    session_id=spec.id,
                    allowed_api_providers=tuple(
                        provider for provider in resources.runtime.api_providers
                        if provider in set(resources.search.candidate_api_providers or ())
                    ),
                    model=spec.worker_model,
                ),
            )
            solution = root / "solution.py"
            if not solution.exists():
                repaired = self._repair_worker_output(
                    harness,
                    composed,
                    resources,
                    composition_worker,
                    root,
                    ["composition worker did not create solution.py"],
                )
                if repaired is None:
                    portfolio.write(portfolio_path)
                    return
                _result, source = repaired
            else:
                source = solution.read_text(encoding="utf-8")
            source = self._ensure_adherent_solution(
                harness, composed, resources, composition_worker, root,
                initial_task=(
                    "Build the explicit composition in task.md from the supplied "
                    "components. Only this composition contract permits cross-family "
                    "routing; do not silently replace all components with a new family."
                ),
            )
            if source is None:
                portfolio.write(portfolio_path)
                return
            self._evaluate_solution_bundle(
                harness,
                composed,
                resources,
                composition_worker,
                root,
                source,
                portfolio,
                portfolio_path,
            )
        except BudgetExceededError:
            portfolio.write(portfolio_path)
            return
        except Exception as exc:
            composed.record_blocker(
                "composition-implementation-needs-human", [str(exc)]
            )
        portfolio.write(portfolio_path)

    def _orchestrator(self, ws, *, web_research: bool = False) -> PiRpcClient:
        extensions = ()
        tools = ()
        if web_research:
            extensions = (Path(__file__).parent / "pi" / "web-search.ts",)
            tools = ("web_search", "web_fetch")
        return PiRpcClient(
            self.command,
            cwd=ws.root,
            model=(
                self.orchestrator_model
                or (
                    self.resources.search.pi_models[0]
                    if self.resources is not None and self.resources.search.pi_models
                    else None
                )
            ),
            system_prompt=_ORCHESTRATOR_SYSTEM,
            timeout=self.orchestrator_timeout,
            extensions=extensions,
            tools=tools,
        )

    def _propose_metric_suite(self, harness, resources: Resources) -> bool:
        ws = harness.workspace
        BudgetLedger(ws.budget_json).check()
        sample = list(harness.data.train[: min(8, len(harness.data.train))])
        prompt = f"""Propose 2-4 independent quality lenses for this task. Include at
least one direct correctness lens and, where meaningful, one graded or robustness
lens. Do not use cost or latency; the controller supplies those. Return JSON:
{{"metric_code": "complete Python defining METRICS", "acceptance": [names],
  "diagnostic": [names], "preference_order": [acceptance names],
  "floors": {{}}, "rationale": "..."}}
Metric functions receive predicted and expected bare values for one output, or
dicts for multiple outputs. They must be deterministic and return numeric scores.
Every name in acceptance, diagnostic, preference_order, and floors must exactly
match a key in the returned METRICS mapping.
Schema:\n{harness.schema.describe()}\nResource policy:\n{json.dumps(resources.to_dict())}\nDevelopment examples:\n{json.dumps(sample, default=str)}"""
        with self._orchestrator(ws) as client:
            result = client.prompt(prompt)
        BudgetLedger(ws.budget_json).charge(dollars=result.usage.dollars, category="agent")
        proposal = _json_object(result.text)
        critic_feedback = ""
        critic_cost = 0.0
        try:
            BudgetLedger(ws.budget_json).check()
            critic_prompt = f"""Act as an adversarial metric critic. Find proxy
hacking, flat metrics, formatting blind spots, semantic blind spots, evaluator
self-preference, and missing robustness lenses in this proposal. Return the same
JSON schema with a corrected complete proposal, plus `critic_feedback`. Do not
remove a direct correctness lens. All role and policy names must exactly match
keys in the corrected METRICS mapping. Task schema:\n{harness.schema.describe()}\nExamples:
{json.dumps(sample, default=str)}\nProposal:\n{json.dumps(proposal)}"""
            with PiRpcClient(
                self.command,
                cwd=ws.root,
                model=(
                    self.orchestrator_model
                    or (
                        resources.search.pi_models[0]
                        if resources.search.pi_models else None
                    )
                ),
                system_prompt=(
                    "You are an independent evaluation-design critic. You never "
                    "implement the task. Return only requested JSON."
                ),
                timeout=self.orchestrator_timeout,
            ) as critic:
                critique = critic.prompt(critic_prompt)
            critic_cost = critique.usage.dollars
            BudgetLedger(ws.budget_json).charge(
                dollars=critic_cost, category="agent"
            )
            revised = _json_object(critique.text)
            if revised.get("metric_code"):
                proposal = revised
            critic_feedback = str(revised.get("critic_feedback") or "")
        except Exception as exc:
            critic_feedback = f"metric critic unavailable: {exc}"

        code = str(proposal.get("metric_code") or "")
        if "METRICS" not in code and "def metric" not in code:
            raise RunnerError("Pi metric proposal did not define METRICS or metric().")
        metric_mod.write_metric(ws, code)
        names = tuple(metric_mod.quality_metrics(ws))
        suite, adjustments = _normalize_metric_suite_proposal(proposal, names)
        proposed = {
            "suite": suite.to_dict(),
            "rationale": proposal.get("rationale", ""),
            "critic_feedback": critic_feedback,
            "proposal_adjustments": adjustments,
            "pi_usage_dollars": result.usage.dollars + critic_cost,
        }
        proposal_path = ws.root / "metric_proposal.json"
        proposal_path.write_text(json.dumps(proposed, indent=2) + "\n")
        if os.environ.get("AP_AUTO_APPROVE_METRIC", "").strip().lower() in ("1", "true", "yes"):
            approve_suite(ws, "auto (AP_AUTO_APPROVE_METRIC)", suite)
            return True
        print(
            "[autoprogramming] Pi proposed a metric suite and paused before search. "
            f"Review {ws.metric_py} and {proposal_path}; demonstrate it on real "
            "examples, then approve with:\n"
            "    import json\n"
            "    from autoprogramming.objectives import approve_suite, MetricSuite\n"
            f"    suite = MetricSuite.from_dict(json.load(open({str(proposal_path)!r}))['suite'])\n"
            "    approve_suite(prg.workspace, 'your name', suite)\n"
            "or use prg.approve_metric_suite(...), then resume optimize()."
        )
        return False

    def _create_portfolio(self, harness, resources: Resources) -> Portfolio:
        BudgetLedger(harness.workspace.budget_json).check()
        feasibility = resources.feasibility()
        prompt = f"""Before designing the portfolio, use web_search at least twice
with different task-specific queries and inspect current sources for modern
models, algorithms, libraries, and compound systems. Do not include private
examples in queries. Then design a diverse task-specific implementation
portfolio across EVERY feasible tier in the supplied feasibility map. Each
avenue must use a materially distinct mechanism and cite source URLs in
`research_sources`. Do not write code. Return JSON:
{{"avenues": [{{"id":"...", "tier":1, "title":"...",
"hypothesis":"...", "implementation_brief":"...", "mechanism":"...",
"runtime_requirements":[], "allowed_api_providers":[],
"required_capabilities":[], "required_mechanisms":["non-negotiable evidence"],
"forbidden_substitutions":["specific cross-family fallback"], "max_rounds":3,
"worker_model":"one exact Resources.search.pi_models pattern, or null",
"wildcard":false, "research_sources":["https://..."]}}],
"exclusions": {{"tier": "reason"}}}}
Every avenue is a hard mechanism experiment: missing packages, credentials, GPU,
models, or network may block it but must never justify another implementation
family. Specify what must be present and what substitutions are forbidden.
Schema:\n{harness.schema.describe()}\nResources:\n{json.dumps(resources.to_dict())}\nFeasibility:\n{json.dumps(feasibility)}"""
        try:
            with self._orchestrator(harness.workspace, web_research=True) as client:
                result = client.prompt(prompt)
            BudgetLedger(harness.workspace.budget_json).charge(
                dollars=result.usage.dollars, category="agent"
            )
            research = _web_research_records(result)
            if len(research) < 2:
                raise RunnerError(
                    "Pi portfolio planning did not complete the required two "
                    "current-source web searches. Planning from model memory was "
                    "refused; fix web access or let the human-facing orchestrator "
                    "perform prg.web_search(...)."
                )
            research_path = harness.workspace.research_json
            research_path.parent.mkdir(parents=True, exist_ok=True)
            sources: dict[str, dict] = {}
            searches = []
            for item in research:
                normalized = {
                    "query": item.get("query", ""),
                    "searched_at": item.get("searchedAt", ""),
                    "results": item.get("results", []),
                }
                searches.append(normalized)
                for source in normalized["results"]:
                    if source.get("url"):
                        sources[str(source["url"])] = source
            research_path.write_text(json.dumps({
                "searches": searches,
                "sources": list(sources.values()),
                "updated_at": searches[-1].get("searched_at", ""),
            }, indent=2) + "\n")
            from .research import WebResearchError, ensure_researched

            try:
                ensure_researched(harness.workspace)
            except WebResearchError as exc:
                raise RunnerError(str(exc)) from exc
            value = _json_object(result.text)
            research_urls = [
                str(source.get("url"))
                for search in research
                for source in search.get("results", [])
                if source.get("url")
            ]
            raw_specs = []
            known_research_urls = set(research_urls)
            for raw in value.get("avenues", []):
                item = dict(raw)
                cited = [
                    str(url) for url in item.get("research_sources", [])
                    if str(url) in known_research_urls
                ]
                item["research_sources"] = cited or research_urls
                raw_specs.append(item)
            specs = [
                ensure_avenue_contract(AvenueSpec.from_dict(v), resources)
                for v in raw_specs
            ]
            portfolio = Portfolio.create(
                resources,
                specs,
                exclusions={int(k): str(v) for k, v in value.get("exclusions", {}).items()},
                fill_missing=True,
            )
            from dataclasses import replace

            for avenue in portfolio.avenues:
                if not avenue.spec.research_sources:
                    avenue.spec = replace(
                        avenue.spec, research_sources=tuple(research_urls)
                    )
            return portfolio
        except RunnerError:
            raise
        except Exception as exc:
            print(
                f"[autoprogramming] Pi portfolio plan was structurally invalid "
                f"after web research ({exc}); filling missing tiers with the "
                "deterministic breadth policy while preserving the research record."
            )
            portfolio = Portfolio.create(resources, [], fill_missing=True)
            try:
                from dataclasses import replace

                urls = tuple(
                    str(source.get("url"))
                    for search in research
                    for source in search.get("results", [])
                    if source.get("url")
                )
                for avenue in portfolio.avenues:
                    avenue.spec = replace(
                        avenue.spec, research_sources=urls
                    )
            except Exception:
                pass
            return portfolio

    def _run_avenue(
        self, harness, spec: AvenueSpec, resources: Resources,
        worker: PiWorkerRunner, *, human_retry_confirmed: bool = False,
    ):
        root = _avenue_dir(harness.workspace, spec.id)
        root.mkdir(parents=True, exist_ok=True)
        # Generic filenames and prose intentionally reveal no optimizer, candidate,
        # metric, score, split, or competing-agent context.
        task_doc = _task_document(harness.schema, spec, resources)
        if human_retry_confirmed:
            task_doc += (
                "\n## Human-confirmed retry\nA human reviewed the previous setup "
                "blocker and confirmed that the capability was fixed or that this "
                "approach must be attempted again. Treat that confirmation as newer "
                "than any stale detected-resource field above. Preserve the assigned "
                "mechanism; do not introduce a fallback.\n"
            )
        (root / "task.md").write_text(task_doc)
        from .data import development_partition

        rows = list(harness.data.train)
        try:
            seed = int(json.loads(harness.workspace.split_json.read_text()).get("seed", 0))
        except (OSError, ValueError):
            seed = 0
        fit_rows, probe_rows = development_partition(rows, seed=seed)
        with (root / "examples.jsonl").open("w", encoding="utf-8") as fh:
            for row in fit_rows:
                fh.write(json.dumps(row, default=str) + "\n")
        task = (
            "Implement the function described in task.md through exactly its "
            "non-negotiable approach contract. Inspect examples.jsonl, create "
            "solution.py, and push that mechanism as far as possible. A working "
            "answer from another implementation family is invalid. If infrastructure "
            "is absent, preserve the mechanism and fail clearly rather than adding "
            "a fallback. Test what the environment permits and syntax-check the file."
        )
        model = self.worker_model or spec.worker_model or (
            resources.search.pi_models[0] if resources.search.pi_models else None
        )
        # Probe contents never enter the worker directory or prompt.
        _ = probe_rows
        return worker.run(
            root, task, model=model, session_id=spec.id,
            allowed_api_providers=spec.allowed_api_providers,
        ), root


def _orchestration_dir(workspace) -> Path:
    path = Path(workspace.root) / ".ap" / "controller"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_resources(workspace) -> Resources:
    path = getattr(workspace, "resources_json", Path(workspace.root) / "resources.json")
    if not Path(path).exists():
        raise RunnerError(
            "Pi orchestration requires a confirmed Resources profile. Pass "
            "resources=ap.Resources(...) to optimize(), or construct "
            "PiOrchestratorBackend(resources=...)."
        )
    return Resources.from_dict(json.loads(Path(path).read_text()))
