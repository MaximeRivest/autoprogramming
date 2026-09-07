"""`prg` — the agent-side handle to an optimization workspace.

Two names, one object, two trust levels: the user's Program runs the active
candidate; the AgentHarness here can create and score candidates, always
through the guards. finalize() is the single code path that reads the test
split — evaluated exactly once, at the end, on the top candidates only.
"""

from __future__ import annotations

import contextlib
import getpass
import json
import os
import sys
from dataclasses import dataclass, field

from . import candidates as candidates_mod
from . import data as data_mod
from . import guards
from . import metric as metric_mod
from . import scoring
from .budget import BudgetLedger
from .candidates import Candidate
from .errors import FinalizedError, MetricNotApprovedError, NotOptimizedError
from .runner import RunResult, run_candidate
from .schema import Schema
from .scoring import (
    COST_OBJECTIVES,
    CompareReport,
    EvalReport,
    TradeoffReport,
    _format_objective,
)
from .workspace import Workspace


def attach(workspace_path) -> "AgentHarness":
    """Attach to an existing workspace as the agent: ``prg = ap.attach(path)``."""
    return AgentHarness(workspace_path)


@dataclass
class TracedRun:
    """One traced run of a candidate on a single train row."""

    result: RunResult
    expected: dict
    score: float | None

    def __str__(self) -> str:
        lines = [self.result.trace().rstrip("\n")]
        lines.append(f"expected: {self.expected!r}")
        if self.score is None:
            lines.append("score: n/a (metric not approved yet — approve it to score traces)")
        else:
            lines.append(f"score: {self.score:.3f}")
        return "\n".join(lines)


@dataclass
class FrontierReport:
    """Pareto view over per-row train scores from scores.json."""

    rows: dict[str, dict]
    nondominated: list[str]
    missing: list[str]

    def __str__(self) -> str:
        if not self.rows:
            lines = [
                "no per-row train scores yet — run "
                "prg.eval(name, split='train', per_instance=True) to build the frontier"
            ]
        else:
            lines = ["Pareto frontier over per-row train scores:"]
            for rid, info in self.rows.items():
                names = ", ".join(info["candidates"])
                lines.append(f"  {rid}: {info['score']:.3f}  <- {names}")
            lines.append(f"nondominated: {', '.join(self.nondominated) or '(none)'}")
        if self.missing:
            lines.append(
                f"no per-row train scores yet: {', '.join(self.missing)} "
                f"(eval with split='train', per_instance=True)"
            )
        return "\n".join(lines)


@dataclass
class FinalReport:
    """The final report card — test scores, demotions, activation, tradeoffs.

    The primary quality metric drives the "test scores" block, the demotion
    logic, and the activated default (all as before). ``frontier`` names the
    Pareto-nondominated finalists over the full TEST objective vector (quality
    maximized, cost/latency minimized); each entry carries its ``objectives``
    means and a ``frontier`` flag so the tradeoff table can be rendered.
    """

    entries: list[dict]
    activated: str | None
    val_reliability: str
    frontier: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["test scores (evaluated once):"]
        for e in self.entries:
            lines.append(f"  {e['candidate']}: {e['test_mean']:.2f}   ({e['note']})")
        if self.entries and all(e["demoted"] for e in self.entries):
            lines.append(
                "every finalist scored far below its val score — that is "
                "overfitting to val; the best test score was activated anyway, "
                "treat these numbers with suspicion."
            )
        lines.append(f"activated: {self.activated if self.activated else '(none)'}")
        lines.extend(self._tradeoff_block())
        if self.val_reliability == "warn":
            lines.append(
                "note: val absorbed many selection decisions relative to its "
                "size; its scores are noisy — trust the test column above."
            )
        elif self.val_reliability == "unreliable":
            lines.append(
                "warning: val scores have lost meaning (too many selection "
                "decisions for the val size) and must not be quoted — the "
                "test scores above are the only report card."
            )
        return "\n".join(lines)

    def _tradeoff_block(self) -> list[str]:
        objective_entries = [e for e in self.entries if e.get("objectives")]
        if not objective_entries:
            return []
        obj_names = list(objective_entries[0]["objectives"].keys())
        header = ["", "candidate", *obj_names]
        table = [header]
        for e in objective_entries:
            mark = "*" if e["candidate"] in self.frontier else " "
            table.append([
                mark, e["candidate"],
                *[_format_objective(n, e["objectives"][n]) for n in obj_names],
            ])
        widths = [max(len(row[i]) for row in table) for i in range(len(header))]
        rendered = [
            "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
            for row in table
        ]
        lines = ["", "quality / cost tradeoffs:", *rendered]
        compute_targets = sorted({
            str(e.get("evaluation_compute"))
            for e in objective_entries if e.get("evaluation_compute")
        })
        if compute_targets:
            lines.append("evaluation compute: " + ", ".join(compute_targets))
        constrained = [
            e for e in objective_entries if e.get("deployment_notes")
        ]
        if constrained:
            lines.append("deployment requirements:")
            for entry in constrained:
                lines.append(
                    f"  {entry['candidate']}: "
                    + "; ".join(entry["deployment_notes"])
                )
        alts = [
            e for e in objective_entries
            if e["candidate"] in self.frontier and e["candidate"] != self.activated
        ]
        if alts:
            names = ", ".join(
                f"{e['candidate']} (ws.activate({e['candidate']!r}, {e['test_mean']:.4g}))"
                for e in alts
            )
            lines.append(
                f"activated {self.activated} under the precommitted frontier "
                f"preference; quality/cost frontier alternatives: {names}"
            )
        else:
            lines.append(
                f"activated {self.activated}; it is the only candidate on the "
                f"quality/cost frontier"
            )
        return lines

    def to_dict(self) -> dict:
        """JSON-serializable form (what final_report.json stores)."""
        return {
            "entries": self.entries,
            "activated": self.activated,
            "val_reliability": self.val_reliability,
            "frontier": self.frontier,
        }


class AgentHarness:
    """The `prg` object the coding agent holds during optimization."""

    def __init__(self, workspace):
        if isinstance(workspace, Workspace):
            self._workspace = workspace
        else:
            self._workspace = Workspace.load(workspace)
        self._data_view: data_mod.SplitView | None = None

    @property
    def workspace(self) -> Workspace:
        """The underlying workspace object."""
        return self._workspace

    @property
    def schema(self) -> Schema:
        """The program's immutable schema (inputs, outputs, docstrings)."""
        return self._workspace.schema

    @property
    def data(self):
        """Split view: .train is readable, .val is scoring-only, no .test."""
        if self._data_view is None:
            self._data_view = data_mod.SplitView(self._workspace)
        return self._data_view

    @property
    def budget(self) -> dict:
        """Remaining headroom per unit (dollars / eval_calls / minutes)."""
        return BudgetLedger(self._workspace.budget_json).remaining()

    def eval(self, candidate, split: str = "val", per_instance: bool = False,
             n_repeats: int | None = None) -> EvalReport:
        """Score a candidate: aggregate on val (selection), per-row on train."""
        return scoring.evaluate(
            self._workspace, candidate, split=split,
            per_instance=per_instance, n_repeats=n_repeats,
        )

    def run(self, candidate, split: str = "train", row: int = 0) -> TracedRun:
        """One traced run of a single train row — the reflection primitive."""
        guards.assert_trace_allowed(split)
        ws = self._workspace
        rows = data_mod.load_split(ws, split)
        if not 0 <= row < len(rows):
            raise IndexError(
                f"{split} has {len(rows)} rows (row_0 … row_{len(rows) - 1}); "
                f"there is no row {row}."
            )
        cand = candidates_mod.load_candidate(ws, candidate)
        ledger = BudgetLedger(ws.budget_json)
        ledger.check()
        schema = ws.schema
        row_dict = rows[row]
        inputs = schema.coerce_inputs(row_dict)
        from . import runner as runner_mod

        if (
            run_candidate is runner_mod.run_candidate
            and getattr(runner_mod, "candidate_session", None) is not None
        ):
            with runner_mod.candidate_session(ws, cand) as session:
                result = session.run(inputs)
        else:
            result = run_candidate(ws, cand, inputs)
        ledger.charge(
            eval_calls=1, dollars=result.cost_dollars or 0.0, category="candidate"
        )
        expected = schema.coerce_expected(row_dict)
        score: float | None = None
        if metric_mod.is_approved(ws):
            if result.ok:
                metric_fn = metric_mod.load_metric(ws)
                score, _ = metric_mod.score_pair(
                    metric_fn, schema, result.outputs, expected, _approval_weights(ws)
                )
            else:
                score = 0.0
        return TracedRun(result=result, expected=expected, score=score)

    def new_candidate(self, source: str | None = None, from_: str | None = None) -> Candidate:
        """Create the next candidate file, from source text or by copying."""
        return candidates_mod.new_candidate(self._workspace, source=source, from_=from_)

    def frontier(self) -> FrontierReport:
        """True Pareto frontier over per-row train scores in scores.json."""
        ws = self._workspace
        scores = scoring.load_scores(ws)
        per_cand: dict[str, dict[str, float]] = {}
        for name, entry in scores.get("candidates", {}).items():
            rows_map = (entry.get("train") or {}).get("rows") or {}
            if rows_map:
                per_cand[name] = {rid: float(v) for rid, v in rows_map.items()}

        all_names = set(candidates_mod.list_candidates(ws)) | set(scores.get("candidates", {}))
        missing = sorted(all_names - set(per_cand), key=_cand_key)

        rows_report: dict[str, dict] = {}
        all_row_ids = sorted({rid for m in per_cand.values() for rid in m}, key=_row_key)
        for rid in all_row_ids:
            having = {n: m[rid] for n, m in per_cand.items() if rid in m}
            best = max(having.values())
            winners = sorted((n for n, v in having.items() if v == best), key=_cand_key)
            rows_report[rid] = {"score": best, "candidates": winners}

        nondominated = [
            name for name in sorted(per_cand, key=_cand_key)
            if not any(
                _dominates(per_cand[other], per_cand[name])
                for other in per_cand if other != name
            )
        ]
        return FrontierReport(rows=rows_report, nondominated=nondominated, missing=missing)

    def compare(self, a: str, b: str, split: str = "val",
                objective: str | None = None) -> CompareReport:
        """Paired comparison of two candidates' stored per-row scores.

        ``objective=None`` compares the primary quality metric; pass an
        objective name (a quality metric, ``cost_dollars`` or ``latency_s``) to
        compare on that objective's stored per-row scores instead.
        """
        return scoring.compare(self._workspace, a, b, split=split, objective=objective)

    def tradeoffs(self, split: str = "val") -> TradeoffReport:
        """The quality/cost Pareto frontier over evaluated candidates."""
        return scoring.tradeoffs(self._workspace, split=split)

    def propose_metric(self, code: str, examples: list[tuple], note: str = "",
                       primary: str | None = None) -> bool | str:
        """Write a metric set, demonstrate it on examples, and seek sign-off.

        ``code`` may define a single ``metric`` or a ``METRICS`` dict; the demo
        table shows one column per quality metric so the user signs off on the
        whole set at once, with ``primary`` (the headline metric) marked.
        Returns True once approved; returns the user's feedback string when they
        push back (the agent iterates); raises MetricNotApprovedError when
        nobody can approve (no terminal, no AP_AUTO_APPROVE_METRIC).
        """
        ws = self._workspace
        metric_mod.write_metric(ws, code)
        demo = metric_mod.demonstrate(ws, examples)
        effective_primary = _effective_primary(ws, primary)
        table = _demo_table(demo, effective_primary)
        if _auto_approve_enabled():
            metric_mod.approve(ws, "auto (AP_AUTO_APPROVE_METRIC)", primary=primary)
            return True
        if sys.stdin.isatty():
            print(table)
            if note:
                print(note)
            answer = input("Do these scores match your intuition of 'good'? [y/N or feedback] ")
            if answer.strip().lower() in ("y", "yes"):
                metric_mod.approve(ws, getpass.getuser(), primary=primary)
                return True
            return answer
        raise MetricNotApprovedError(
            "The metric was written and demonstrated, but nobody signed off "
            "and stdin is not a terminal to ask. The entire search optimizes "
            "whatever metric.py says — a wrong metric produces a "
            "confidently-scored wrong program — so scoring is refused until a "
            "human approves. Review the demonstration below, then either call "
            "autoprogramming.metric.approve(workspace, 'your name') or set "
            "AP_AUTO_APPROVE_METRIC=1 to accept it as-is.\n" + table
        )

    def approve_metric_suite(self, approved_by: str, suite, *, weights=None) -> None:
        """Approve metric roles and the precommitted selection policy.

        This is the non-interactive counterpart to ``propose_metric``: the
        orchestrator may propose several lenses, but a user still signs off on
        which are acceptance criteria. Diagnostic lenses guide search only.
        """
        from .objectives import approve_suite

        approve_suite(self._workspace, approved_by, suite, weights=weights)

    @property
    def metric_suite(self):
        """The approved suite, with old metrics treated as all-acceptance."""
        from .objectives import metric_suite

        return metric_suite(self._workspace)

    def web_search(self, query: str, *, limit: int = 6):
        """Search current public sources and persist citations for planning.

        This method is intentionally called by the human-facing Pi agent: its
        conversation remains the strategy context. Task examples are never
        placed in the query automatically.
        """
        resources_path = self._workspace.resources_json
        if not resources_path.exists():
            from .research import WebResearchError

            raise WebResearchError(
                "Web research needs a confirmed Resources profile so data-egress "
                "permission is explicit."
            )
        from .resources import Resources
        from .research import WebResearchError, record_report, search_web

        resources = Resources.from_dict(json.loads(resources_path.read_text()))
        if resources.data.external_egress is not True:
            raise WebResearchError(
                "Web research would send a task-derived query off-machine, but "
                "data.external_egress is not permitted. Ask the human to permit "
                "abstract task research or explicitly resolve this blocker; do "
                "not plan from stale model memory."
            )
        normalized_query = str(query).casefold()
        # Research should describe the task, not publish demonstration rows.
        for row in data_mod.load_split(self._workspace, "dev"):
            for value in row.values():
                text = str(value).strip()
                if len(text) >= 16 and text.casefold() in normalized_query:
                    raise WebResearchError(
                        "The search query contains a verbatim development-example "
                        "value. Rewrite it as an abstract task/capability query."
                    )
        report = search_web(query, limit=limit)
        record_report(self._workspace, report)
        return report

    def plan_portfolio(self, specs, *, exclusions=None, policy=None):
        """Record a web-informed plan authored by this current Pi session."""
        from dataclasses import replace

        from .portfolio import AvenueSpec, Portfolio, PortfolioPolicy
        from .research import ensure_researched
        from .resources import Resources

        evidence = ensure_researched(self._workspace)
        metric_mod.ensure_approved(self._workspace)
        resources = Resources.from_dict(
            json.loads(self._workspace.resources_json.read_text())
        )
        sources = tuple(
            str(item.get("url"))
            for item in evidence.get("sources", [])
            if item.get("url")
        )
        known_sources = set(sources)
        normalized = []
        for raw in specs:
            spec = raw if isinstance(raw, AvenueSpec) else AvenueSpec.from_dict(raw)
            if not spec.research_sources:
                spec = replace(spec, research_sources=sources)
            elif not set(spec.research_sources).issubset(known_sources):
                from .research import WebResearchError

                raise WebResearchError(
                    f"Avenue {spec.id!r} cites a URL absent from the persisted "
                    "web-search evidence."
                )
            normalized.append(spec)
        effective_policy = policy
        if isinstance(effective_policy, dict):
            effective_policy = PortfolioPolicy(**effective_policy)
        elif effective_policy is not None and not isinstance(
            effective_policy, PortfolioPolicy
        ):
            raise TypeError("policy= must be PortfolioPolicy, a dict, or None.")
        if effective_policy is None and guards.is_bootstrap(self._workspace):
            effective_policy = PortfolioPolicy(min_configs_before_abandon=1)
        portfolio = Portfolio.create(
            resources,
            normalized,
            exclusions=exclusions,
            policy=effective_policy,
            fill_missing=True,
        )
        for avenue in portfolio.avenues:
            if not avenue.spec.research_sources:
                avenue.spec = replace(
                    avenue.spec, research_sources=sources
                )
        self._workspace.portfolio_json.parent.mkdir(parents=True, exist_ok=True)
        portfolio.write(self._workspace.portfolio_json)
        return portfolio.to_dict()

    def portfolio_status(self) -> dict:
        """Full implementation/audit/failure evidence for host strategy."""
        if not self._workspace.portfolio_json.exists():
            raise NotOptimizedError(
                "No host portfolio exists yet. Search the web and call "
                "prg.plan_portfolio([...]) first."
            )
        from .portfolio import Portfolio

        portfolio = Portfolio.load(self._workspace.portfolio_json)
        status = portfolio.to_dict()
        status["may_finalize"] = portfolio.may_finalize
        status["unresolved_blocker_ids"] = [
            avenue.spec.id for avenue in portfolio.unresolved_blockers
        ]
        return status

    def orchestrate_portfolio(
        self,
        phase: str = "breadth",
        *,
        avenue_ids=(),
        budget=None,
        backend=None,
    ) -> dict:
        """Run one trusted controller phase; strategy stays in this Pi session.

        ``breadth`` dispatches every planned family and its mandatory independent
        configuration, ``deepen`` works only the selected avenue ids, and
        ``compose`` builds from the measured frontier. Pass an explicit
        ``budget=ap.Budget(...)`` to start/resume a phase with new total limits;
        prior spend remains charged. This method launches implementation/audit
        subagents, never another strategy orchestrator.
        """
        if phase not in ("breadth", "deepen", "compose"):
            raise ValueError("phase must be 'breadth', 'deepen', or 'compose'.")
        if not self._workspace.portfolio_json.exists():
            raise NotOptimizedError(
                "No portfolio plan exists. Call prg.plan_portfolio([...]) after "
                "the required web research."
            )
        if budget is not None:
            from .budget import Budget, BudgetLedger

            if not isinstance(budget, Budget):
                raise TypeError("budget= must be an ap.Budget instance.")
            BudgetLedger.start(self._workspace.budget_json, budget)
        from .pi_backend import PiOrchestratorBackend
        from .resources import Resources

        resources = Resources.from_dict(
            json.loads(self._workspace.resources_json.read_text())
        )
        controller = backend or PiOrchestratorBackend(
            resources=resources,
            host_orchestrated=True,
        )
        controller.run(self, context={
            "mode": "optimize",
            "host_orchestrated": True,
            "host_phase": phase,
            "selected_ids": list(avenue_ids),
        })
        return self.portfolio_status()

    def resolve_blocker(
        self, avenue_id: str, action: str, *, confirmed_by: str
    ) -> None:
        """Resolve an environment-blocked approach after checking with a human.

        ``action='retry'`` means the capability was fixed (or the human wants it
        attempted again). ``action='exclude'`` confirms that this approach really
        is unavailable. The controller never makes that exclusion on a worker's
        word alone and never substitutes another mechanism under the avenue.
        """
        path = self._workspace.portfolio_json
        if not path.exists():
            raise NotOptimizedError(
                "There is no Pi portfolio state with an approach blocker to resolve."
            )
        from .portfolio import Portfolio

        portfolio = Portfolio.load(path)
        portfolio.resolve_blocker(avenue_id, action, confirmed_by)
        portfolio.write(path)

    def cleanup_search_cache(self, *, force: bool = False) -> dict:
        """Remove AutoProgramming-owned package caches and environments.

        Finalized workspaces are safe to clean. For an unfinished workspace,
        ``force=True`` removes only package caches and disposable virtual
        environments. Candidate source, artifacts, outputs, and inactive-candidate
        diagnostics are never removed.
        """
        from .pi_worker import cleanup_worker_cache

        return cleanup_worker_cache(self._workspace, force=force)

    def finalize(self, top_k: int | None = None) -> FinalReport:
        """The one-time test evaluation: score the top val candidates on test,
        demote overfitters, activate the winner, and seal the workspace.

        This is the only code path that reads the test split. It charges the
        budget but never checks it — the report card happens even on an
        exhausted budget. Running it a second time raises FinalizedError.
        """
        ws = self._workspace
        if ws.active.get("finalized"):
            raise FinalizedError(
                f"finalize() already ran for this workspace — test is "
                f"evaluated exactly once, because a second pass would let the "
                f"report card steer the search and turn test into another val "
                f"set. The recorded report is at {ws.final_report}; read that "
                f"instead. To keep improving, start a new workspace (fresh "
                f"data split) and optimize again."
            )
        if ws.portfolio_json.exists():
            from .portfolio import Portfolio

            portfolio = Portfolio.load(ws.portfolio_json)
            if not portfolio.may_finalize:
                blocked = [a.spec.id for a in portfolio.unresolved_blockers]
                raise NotOptimizedError(
                    "Cannot finalize: the controller-enforced portfolio lacks "
                    "conclusive breadth evidence or has unresolved blockers. "
                    f"Blocked avenues: {blocked}. Inspect "
                    "prg.portfolio_status(), repair/retry implementations, or "
                    "obtain explicit human exclusions first."
                )
        metric_mod.ensure_approved(ws)
        scores = scoring.load_scores(ws)
        val_scored = list(dict.fromkeys(scores.get("val_scored", [])))
        if not val_scored:
            raise NotOptimizedError(
                "Cannot finalize: no candidate has been evaluated on val yet, "
                "so there is nothing to promote to the one-time test "
                "evaluation. Selection happens on val — score at least one "
                "candidate first: prg.eval('candidate_0')."
            )
        # Scores select exact source, not a mutable filename. Also run the
        # source-only memorization rule even when an agent skipped train eval;
        # val-only candidates must not bypass lookup-table detection.
        stale = [
            n for n in val_scored
            if _val_stats(scores, n) is not None
            and not scoring.score_provenance_current(ws, n, "val")
        ]
        train_rows_for_check = data_mod.load_split(ws, "train")
        for name in val_scored:
            stats = _val_stats(scores, name)
            if stats is None or name in stale:
                continue
            cand = candidates_mod.load_candidate(ws, name)
            source_flags = guards.memorization_check(
                cand.source, float(stats["mean"]), float(stats["mean"]),
                train_rows_for_check, ws.schema,
            )
            if source_flags:
                scores.setdefault("flags", {}).setdefault(name, [])
                for flag in source_flags:
                    if flag not in scores["flags"][name]:
                        scores["flags"][name].append(flag)
        if stale or scores != scoring.load_scores(ws):
            scoring.save_scores(ws, scores)

        flags = scores.get("flags", {})
        flagged = {n: flags[n] for n in val_scored if flags.get(n)}
        human_excluded: set[str] = set()
        if ws.portfolio_json.exists():
            from .portfolio import AvenueStatus, Portfolio

            portfolio = Portfolio.load(ws.portfolio_json)
            human_excluded = {
                candidate
                for avenue in portfolio.avenues
                if avenue.status == AvenueStatus.INFEASIBLE
                and any("human blocker resolution" in note for note in avenue.notes)
                for candidate in avenue.candidates
            }
        eligible = [
            n for n in val_scored
            if n not in flagged
            and n not in stale
            and n not in human_excluded
            and _val_stats(scores, n) is not None
        ]
        if not eligible:
            if human_excluded:
                raise NotOptimizedError(
                    "Cannot finalize: every otherwise eligible candidate belongs "
                    "to an approach the human explicitly excluded. Retry/provision "
                    "a faithful family or evaluate another conclusive avenue."
                )
            if flagged:
                details = "; ".join(f"{n}: {'; '.join(fl)}" for n, fl in flagged.items())
                raise NotOptimizedError(
                    f"Cannot finalize: every val-scored candidate is flagged "
                    f"as a memorizer, and memorizers are excluded from "
                    f"selection because their scores measure recall of the "
                    f"training rows, not the task ({details}). Write a "
                    f"candidate that generalizes — reflect on train failures, "
                    f"avoid lookup tables over train inputs — then eval it on "
                    f"val and finalize."
                )
            raise NotOptimizedError(
                "Cannot finalize: the val-scored candidates have no stored "
                "val aggregates in scores.json. Re-run prg.eval(name) so "
                "selection has real numbers to promote from."
            )

        # New metric suites promote the VAL Pareto frontier rather than merely
        # the two highest points on one scalar. Legacy approvals preserve the
        # old top-2 behavior.
        from .objectives import (
            meets_floors,
            metric_suite,
            preference_key,
            selection_goals,
        )

        suite = metric_suite(ws)
        approval = json.loads(ws.metric_approval.read_text())
        suite_aware = isinstance(approval.get("suite"), dict)
        if suite_aware:
            val_vectors = {
                name: {
                    objective: float(stats["mean"])
                    for objective, stats in _val_stats(scores, name).get("objectives", {}).items()
                    if objective in selection_goals(suite)
                }
                for name in eligible
            }
            floor_ok = {
                name: vector for name, vector in val_vectors.items()
                if meets_floors(vector, suite)
            }
            pool = floor_ok or val_vectors
            frontier_names = scoring.pareto_nondominated(
                scoring._domination_points(pool), selection_goals(suite)
            )
            ordered = sorted(
                frontier_names,
                key=lambda n: (preference_key(pool[n], suite), _cand_key(n)),
            )
            limit = top_k if top_k is not None else suite.policy.max_test_finalists
            top = _diverse_finalists(ws, ordered, max(1, limit))
        else:
            limit = 2 if top_k is None else top_k
            top = sorted(
                eligible,
                key=lambda n: (-_val_stats(scores, n)["mean"], _cand_key(n)),
            )[: max(1, limit)]

        schema = ws.schema
        test_rows = data_mod.load_split(ws, "test")
        metrics = metric_mod.quality_metrics(ws)
        primary = metric_mod.primary_name(ws)
        weights = _approval_weights(ws)
        ledger = BudgetLedger(ws.budget_json) if ws.budget_json.exists() else None

        expected_by_row = {
            f"row_{i}": schema.coerce_expected(r) for i, r in enumerate(test_rows)
        }
        entries: list[dict] = []
        avenue_metadata = _candidate_avenue_metadata(ws)
        for name in top:
            cand = candidates_mod.load_candidate(ws, name)
            evaluation_compute = _evaluation_compute_label(ws, cand)
            n_repeats = 1 if cand.deterministic else scoring.DEFAULT_REPEATS
            cache_rows: dict[str, list[dict]] = {}
            from . import runner as runner_mod

            session_factory = (
                getattr(runner_mod, "candidate_session", None)
                if run_candidate is runner_mod.run_candidate
                else None
            )
            session_cls = (
                getattr(runner_mod, "CandidateSession", None)
                if run_candidate is runner_mod.run_candidate
                else None
            )
            session_context = (
                session_factory(ws, cand)
                if session_factory is not None
                else session_cls(ws, cand)
                if session_cls is not None
                else contextlib.nullcontext(None)
            )
            with session_context as candidate_session:
                run_one = (
                    candidate_session.run
                    if candidate_session is not None
                    else lambda inputs: run_candidate(ws, cand, inputs)
                )
                for i, row_dict in enumerate(test_rows):
                    inputs = schema.coerce_inputs(row_dict)
                    reps_list: list[dict] = []
                    for _ in range(n_repeats):
                        result = run_one(inputs)
                        if ledger is not None:
                            ledger.charge(
                                eval_calls=1,
                                dollars=result.cost_dollars or 0.0,
                                category="candidate",
                            )
                        reps_list.append(scoring._run_cache_entry(result))
                    cache_rows[f"row_{i}"] = reps_list
            scoring.write_output_cache(ws, cand, "test", cache_rows)
            sub = scoring._aggregate_from_cache(
                cache_rows, metrics, primary, schema, weights,
                expected_by_row, n_repeats,
            )
            sub["candidate_sha"] = scoring._source_sha(ws, cand)
            test_mean = sub["mean"]  # compatibility headline
            objectives = {o: v["mean"] for o, v in sub["objectives"].items()}

            stats = _val_stats(scores, name)
            val_mean = float(stats["mean"])
            ci = stats.get("ci95") or [val_mean, val_mean]
            half_width = (float(ci[1]) - float(ci[0])) / 2.0
            gap = val_mean - test_mean
            demoted = gap > max(0.05, half_width)
            note = f"val was {val_mean:.2f} — " + (
                "overfit to val, demoted" if demoted else "healthy gap"
            )
            metadata = avenue_metadata.get(name, {})
            deployment_notes = list(metadata.get("deployment_notes", []))
            if cand.pi_runtime:
                pi_requirement = (
                    "requires the Pi CLI and an authenticated model subscription "
                    "at deployment; no raw provider API key is required"
                )
                if pi_requirement not in deployment_notes:
                    deployment_notes.append(pi_requirement)
            entries.append({
                "candidate": name,
                "val_mean": val_mean,
                "test_mean": test_mean,
                "gap": gap,
                "demoted": demoted,
                "note": note,
                "objectives": objectives,
                "frontier": False,
                "cold_start_s": sub.get("cold_start_s"),
                "approach_tier": metadata.get("tier"),
                "approach": metadata.get("title"),
                "runtime_requirements": metadata.get("runtime_requirements", []),
                "deployment_notes": deployment_notes,
                "evaluation_compute": evaluation_compute,
            })

        entries.sort(key=lambda e: (-e["test_mean"], _cand_key(e["candidate"])))

        # Pareto frontier over the finalists' full TEST objective vectors
        # (quality maximized, cost/latency minimized). Cost coordinates are
        # rounded for domination so wall-clock latency jitter cannot flip which
        # finalists are reported on the frontier run to run.
        points = {e["candidate"]: e["objectives"] for e in entries}
        goals = (
            selection_goals(suite)
            if suite_aware
            else ({**{name: "max" for name in metrics}, **COST_OBJECTIVES})
        )
        frontier = set(scoring.pareto_nondominated(
            scoring._domination_points(points), goals
        ))
        for e in entries:
            e["frontier"] = e["candidate"] in frontier

        # Activate the best-primary candidate that is on the frontier and not
        # demoted; fall back to best-primary on the frontier, then to best
        # primary overall (loud note) — always exactly one activated.
        survivors = [e for e in entries if not e["demoted"]]
        frontier_survivors = [e for e in survivors if e["frontier"]]
        frontier_entries = [e for e in entries if e["frontier"]]
        pool = frontier_survivors or frontier_entries or entries
        if suite_aware:
            winner = min(
                pool,
                key=lambda e: (
                    preference_key(e["objectives"], suite),
                    _cand_key(e["candidate"]),
                ),
            )
        else:
            winner = min(pool, key=lambda e: (-e["test_mean"], _cand_key(e["candidate"])))
        if not survivors:
            winner["note"] += (
                "; every finalist overfit to val — activated on best test "
                "score anyway"
            )

        val_size = int(
            json.loads(ws.split_json.read_text()).get("counts", {}).get("val", 0)
        )
        val_reliability = guards.pressure_status(len(set(val_scored)), val_size)

        report = FinalReport(
            entries=entries,
            activated=winner["candidate"],
            val_reliability=val_reliability,
            frontier=sorted(frontier, key=_cand_key),
        )
        ws.activate(winner["candidate"], winner["test_mean"])
        ws.mark_finalized(report.to_dict())
        if ws.portfolio_json.exists():
            try:
                cleanup = self.cleanup_search_cache()
                cleanup_errors = cleanup["remote_errors"]
            except Exception as exc:
                cleanup_errors = [str(exc)]
            if cleanup_errors:
                print(
                    "[autoprogramming] finalized successfully, but worker cache "
                    "cleanup needs attention: " + "; ".join(cleanup_errors),
                    file=sys.stderr,
                )
        return report


def _val_stats(scores: dict, name: str) -> dict | None:
    return scores.get("candidates", {}).get(name, {}).get("val")


def _approval_weights(workspace) -> dict | None:
    path = workspace.metric_approval
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("weights")


def _auto_approve_enabled() -> bool:
    return os.environ.get("AP_AUTO_APPROVE_METRIC", "").strip().lower() in ("1", "true", "yes")


def _effective_primary(workspace, primary: str | None) -> str | None:
    """The metric name that WILL be primary once approved (for the demo table)."""
    try:
        metrics = metric_mod.quality_metrics(workspace)
    except Exception:
        return primary
    if primary in metrics:
        return primary
    return metric_mod.primary_name(workspace)


def _fmt_metric_score(value) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else repr(value)


def _demo_table(demo: list[dict], primary: str | None = None) -> str:
    """Render the metric demonstration: one column per quality metric.

    Single-metric proposals keep the one-column shape; multi-metric proposals
    show every metric's score on each example, with the primary metric marked,
    so the user signs off on the whole set at once.
    """
    lines = ["proposed metric on real examples:"]
    names = list(demo[0].get("scores", {})) if demo else []
    for entry in demo:
        lines.append(f"  expected:  {entry['expected']!r}")
        scores = entry.get("scores")
        if scores and len(names) > 1:
            lines.append(f"  predicted: {entry['predicted']!r}")
            for name in names:
                tag = "  <- primary" if name == primary else ""
                lines.append(f"      {name}: {_fmt_metric_score(scores[name])}{tag}")
        else:
            shown = _fmt_metric_score(entry.get("score"))
            lines.append(f"  predicted: {entry['predicted']!r}   -> {shown}")
    return "\n".join(lines)


def _dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    """True Pareto dominance: a >= b on every row b has, > on at least one."""
    strict = False
    for rid, bv in b.items():
        av = a.get(rid)
        if av is None or av < bv:
            return False
        if av > bv:
            strict = True
    return strict


def _cand_key(name: str):
    prefix, _, idx = name.rpartition("_")
    if prefix and idx.isdigit():
        return (0, int(idx), name)
    return (1, 0, name)


def _row_key(row_id: str):
    return _cand_key(row_id)


def _evaluation_compute_label(workspace, candidate: Candidate | None = None) -> str:
    path = getattr(workspace, "resources_json", None)
    if path is None or not path.exists():
        return "local"
    try:
        resources = json.loads(path.read_text())
        remote = (resources.get("search") or {}).get("remote_compute")
        if isinstance(remote, dict) and remote.get("endpoint"):
            if candidate is not None:
                if candidate.pi_runtime:
                    return "local:authenticated-pi-host"
                if candidate.network_required or candidate.api_providers:
                    return "local:authenticated-network-host"
                from .remote import candidate_placement

                planned = candidate_placement(workspace, candidate.name)
                if planned == "local" or (
                    planned is None and not candidate.compute_heavy
                ):
                    return "local:lightweight-host"
            return f"remote:{remote['endpoint']}"
    except (OSError, ValueError, TypeError):
        pass
    return "local"


def _candidate_avenue_metadata(workspace) -> dict[str, dict]:
    path = getattr(workspace, "portfolio_json", None)
    if path is None or not path.exists():
        return {}
    try:
        portfolio = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    result: dict[str, dict] = {}
    for avenue in portfolio.get("avenues", []):
        spec = avenue.get("spec") or {}
        metadata = {
            "tier": spec.get("tier"),
            "title": spec.get("title"),
            "runtime_requirements": list(spec.get("runtime_requirements") or []),
            "deployment_notes": list(spec.get("deployment_notes") or []),
        }
        for candidate in avenue.get("candidates", []):
            result[str(candidate)] = metadata
    return result


def _diverse_finalists(workspace, ordered: list[str], limit: int) -> list[str]:
    """Prefer one val-frontier representative per approach tier before repeats."""
    path = getattr(workspace, "portfolio_json", None)
    if path is None or not path.exists():
        return ordered[:limit]
    try:
        portfolio = json.loads(path.read_text())
        tiers = {
            candidate: int(avenue["spec"]["tier"])
            for avenue in portfolio.get("avenues", [])
            for candidate in avenue.get("candidates", [])
        }
    except (OSError, ValueError, KeyError, TypeError):
        return ordered[:limit]
    selected: list[str] = []
    seen_tiers: set[int] = set()
    for name in ordered:
        tier = tiers.get(name)
        if tier is not None and tier not in seen_tiers:
            selected.append(name)
            seen_tiers.add(tier)
            if len(selected) >= limit:
                return selected
    for name in ordered:
        if name not in selected:
            selected.append(name)
            if len(selected) >= limit:
                break
    return selected
