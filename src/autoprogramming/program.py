"""User-facing programs: the ``@program`` decorator and the Program object.

A Program can only run its active candidate. Creating and scoring candidates
belongs to the agent-side handle (``autoprogramming.attach``) — two names,
one workspace, two trust levels.
"""

from __future__ import annotations

import functools
import inspect
import json
import shutil
from pathlib import Path

from .budget import Budget, BudgetLedger
from .errors import (
    BudgetError,
    BudgetExceededError,
    DataDisciplineError,
    NotOptimizedError,
    RunnerError,
    WorkspaceError,
)
from .schema import Schema


def program(fn) -> "Program":
    """Turn a typed, documented function stub into a Program (``ap.program``).

    The schema is extracted immediately, so schema problems (missing
    annotations, duplicate output types, ``*args``, defaults) surface at
    decoration time, not deep inside optimize().
    """
    return Program(fn)


class Program:
    """A program defined by its schema and implemented by whatever wins.

    ``optimize()`` launches a coding agent that searches over complete
    candidate implementations inside a workspace; ``__call__`` runs the
    active candidate like a normal function; ``save()`` relocates the
    workspace; ``distill()`` compresses the current behavior into a cheaper
    imitation trained on the program's own logs.
    """

    def __init__(self, fn):
        functools.update_wrapper(self, fn)
        self._fn = fn
        self.schema = Schema.from_function(fn)
        self._workspace = None
        self._logging = False

    # ---------------------------------------------------------------- state

    @property
    def workspace(self):
        """The bound Workspace, or None before optimize()/use()."""
        return self._workspace

    def use(self, path) -> "Program":
        """Bind an existing workspace directory to this program."""
        from .workspace import Workspace

        ws = Workspace.load(path)
        if ws.program_name != self.schema.name:
            raise WorkspaceError(
                f"Refused to bind workspace {ws.root}: it was built for program "
                f"{ws.program_name!r}, not {self.schema.name!r}. Binding it would "
                f"run another program's candidates against this schema. Point "
                f".use() at a workspace created for {self.schema.name!r}, or run "
                f".optimize() to create one."
            )
        self._workspace = ws
        return self

    def __repr__(self) -> str:
        params = ", ".join(f"{f.name}: {f.type_name}" for f in self.schema.inputs)
        outs = ", ".join(self.schema.output_names)
        state = str(self._workspace.root) if self._workspace is not None else "unoptimized"
        return f"<Program {self.schema.name}({params}) -> {outs} [{state}]>"

    # ------------------------------------------------------------- optimize

    def optimize(self, data, budget: Budget | None = None, *, workspace=None,
                 seed: int = 0, ratios=None, backend=None, resources=None,
                 _context=None):
        """Launch the optimization agent on this program.

        ``data`` is a list of dict rows, a duck-typed DataFrame, a .csv/.jsonl
        path, or ``"logs:reviewed"``. ``budget`` is required — there is no
        default. ``ratios`` of None means ``data.DEFAULT_RATIOS``. Returns a
        FinalReport when candidates were scored (or the run was already
        finalized), else None.
        """
        if not isinstance(budget, Budget):
            raise BudgetError(
                "optimize() was refused: no budget was given, and there is no "
                "default — evaluation spends real money and time (LLM candidates "
                "cost money to score, not just to mutate), so you must say what "
                "you're willing to spend. Pass budget=ap.Budget(dollars=20), "
                "ap.Budget(eval_calls=2000), or ap.Budget(minutes=30) — combinable."
            )
        rows = self._resolve_rows(data)
        from_reviewed_logs = isinstance(data, str) and data == "logs:reviewed"

        from . import data as data_mod
        from .harness import AgentHarness
        from .workspace import Workspace

        sha = data_mod.data_sha(rows)
        if workspace is not None:
            root = Path(workspace)
        elif self._workspace is not None:
            if from_reviewed_logs:
                root = self._reopt_root(sha)
                print(
                    f"[autoprogramming] re-optimizing from reviewed logs into "
                    f"{root} — the split in {self._workspace.root} is fixed (the "
                    f"data is split once), so the reviewed entries get their own "
                    f"workspace. Pass workspace=... to choose the path yourself."
                )
            else:
                root = self._workspace.root
        else:
            root = Path(f"{self.schema.name}_ap")

        if root.exists():
            ws = Workspace.load(root)
            if ws.program_name != self.schema.name:
                raise WorkspaceError(
                    f"Refused to optimize into {root}: that workspace belongs to "
                    f"program {ws.program_name!r}, not {self.schema.name!r}. Use a "
                    f"different workspace path for this program."
                )
            split_info = json.loads(ws.split_json.read_text())
            if sha != split_info.get("data_sha"):
                if from_reviewed_logs:
                    raise DataDisciplineError(
                        f'optimize(data="logs:reviewed") was refused for workspace '
                        f"{root}: the reviewed log entries are new data, and that "
                        f"workspace's data was split once into train/val/test and "
                        f"must stay fixed — re-splitting would silently leak val "
                        f"and test rows into training. Omit workspace= to let "
                        f"optimize() start a fresh re-optimization workspace "
                        f"automatically, or pass a new workspace path."
                    )
                raise DataDisciplineError(
                    f"Refused to re-split data in workspace {root}: the data was "
                    f"split once into train/val/test and must stay fixed, otherwise "
                    f"val and test rows silently leak into training and every score "
                    f"becomes a lie. The rows you passed differ from the original "
                    f"split (data_sha mismatch). Pass the original data to continue "
                    f"this run, or point optimize() at a new workspace path to "
                    f"restart with the new data."
                )
        else:
            if ratios is None:
                ratios = data_mod.DEFAULT_RATIOS
            from .guards import BOOTSTRAP_MIN

            splits = data_mod.split_rows(rows, seed=seed, ratios=ratios)
            ws = Workspace.create(
                root, self.schema, splits,
                seed=seed, ratios=ratios, data_sha=sha,
                bootstrap=len(rows) < BOOTSTRAP_MIN,
                secure_data=resources is not None,
            )

        if resources is not None:
            from dataclasses import replace

            from .resources import Resources

            if not isinstance(resources, Resources):
                raise TypeError(
                    "resources= must be an ap.Resources profile separating "
                    "search capabilities, deployment resources, and data policy."
                )
            # A live Pi session exposes its selected provider/model/thinking
            # tuple to bash children. Persist that capability so workers and
            # Pi-backed candidates inherit subscription OAuth without looking
            # for raw API-key environment variables. On resume, reuse the pinned
            # registry rather than letting a catalog refresh mutate the contract.
            if ws.resources_json.exists() and not resources.search.pi_models:
                try:
                    pinned = Resources.from_dict(
                        json.loads(ws.resources_json.read_text())
                    ).search.pi_models
                except (OSError, ValueError, TypeError):
                    pinned = ()
                if pinned:
                    resources = replace(
                        resources,
                        search=replace(resources.search, pi_models=pinned),
                    )
            resources = resources.with_current_pi_model()
            resources.ensure_confirmed()
            ws.secure_splits()
            serialized = json.dumps(resources.to_dict(), indent=2) + "\n"
            if ws.resources_json.exists() and ws.resources_json.read_text() != serialized:
                existing_scores = json.loads(ws.scores_json.read_text())
                if existing_scores.get("candidates"):
                    raise DataDisciplineError(
                        f"Refused to change resources for an active search in {ws.root}: "
                        "resource availability determines which approach families and "
                        "data-egress paths are legal. Start a new workspace for a new "
                        "resource contract."
                    )
                # No implementation has been scored, so a resource correction
                # may replace an unspent plan rather than wedging resume.
                if ws.portfolio_json.exists():
                    ws.portfolio_json.unlink()
            ws.resources_json.write_text(serialized)

        self._workspace = ws
        if resources is not None and ws.active.get("finalized"):
            return self._load_final_report(ws)

        BudgetLedger.start(ws.budget_json, budget)

        from . import guards

        if guards.is_bootstrap(ws):
            print(
                f"[autoprogramming] bootstrap mode: {ws.root} has fewer than "
                f"{guards.BOOTSTRAP_MIN} examples. The agent will build and "
                f"compare baseline candidates but fine-grained mutation loops "
                f"are refused — a 0.92-vs-0.86 difference on 5 val rows is one "
                f"row of noise. Provide 30+ examples for full optimization, or "
                f"ask the agent to generate synthetic examples for you to validate."
            )

        from .backend import default_backend

        if backend is not None:
            be = backend
        elif resources is not None:
            from .host_backend import HostOrchestratorBackend, in_live_pi_session

            if in_live_pi_session():
                # The model already speaking with the human owns strategy. Do
                # not create a second, stateless orchestrator behind its back.
                be = HostOrchestratorBackend(resources=resources)
            else:
                # Headless Pi strategy is an explicit opt-in because it creates
                # a separate context. Pass PiOrchestratorBackend(...) when that
                # is truly desired; otherwise preserve the normal/manual backend.
                be = default_backend()
        else:
            be = default_backend()
        harness = AgentHarness(ws)
        try:
            be.run(harness, context=_context or {"mode": "optimize"})
        except BudgetExceededError:
            pass  # the ledger stopped the loop; the one-time test eval below still runs

        if ws.active.get("finalized"):
            return self._load_final_report(ws)
        if getattr(be, "defer_finalization", False):
            # The current Pi session must inspect worker evidence and choose the
            # next controller phase. Automatic finalization here would bypass
            # that same-session orchestration boundary.
            return None
        if ws.portfolio_json.exists():
            from .portfolio import Portfolio

            portfolio = Portfolio.load(ws.portfolio_json)
            if portfolio.unresolved_blockers:
                print(
                    "[autoprogramming] search is paused for human confirmation of "
                    "blocked approaches. Fix and retry them, or explicitly confirm "
                    "their exclusion with prg.resolve_blocker(...); no fallback "
                    "implementation was accepted."
                )
                return None
        scores = json.loads(ws.scores_json.read_text())
        if scores.get("val_scored"):
            if ws.portfolio_json.exists():
                from .portfolio import Portfolio

                portfolio = Portfolio.load(ws.portfolio_json)
                if not portfolio.may_finalize:
                    print(
                        "[autoprogramming] Pi search paused before finalization: "
                        "portfolio breadth or a pending candidate journal is "
                        "incomplete. Resume optimize() with a new explicit budget."
                    )
                    return None
            return harness.finalize()
        if (_context or {}).get("prepare_only") or (_context or {}).get("mode") == "prepare":
            return None
        print(
            f"[autoprogramming] no candidates were scored in {ws.root}; the "
            f"workspace is ready for a manual session. Attach and continue:\n"
            f"    import autoprogramming as ap\n"
            f'    prg = ap.attach("{ws.root}")\n'
            f"then propose the metric, create candidates, eval, and finalize."
        )
        return None

    def prepare(
        self,
        data,
        budget: Budget | None = None,
        *,
        resources,
        workspace=None,
        seed: int = 0,
        ratios=None,
        backend=None,
    ) -> "PreparedRun":
        """Prepare resources and a metric proposal without dispatching workers.

        Preparation has an explicit budget because the Pi metric/portfolio
        analysis is itself an agent call. The returned object resumes against
        the exact normalized rows and fixed workspace split after sign-off.
        """
        rows = self._resolve_rows(data)
        if backend is not None:
            selected_backend = backend
        else:
            from .host_backend import HostOrchestratorBackend, in_live_pi_session

            if in_live_pi_session():
                selected_backend = HostOrchestratorBackend(resources=resources)
            else:
                # A separate headless Pi strategist must be requested explicitly.
                from .backend import default_backend

                selected_backend = default_backend()
        self.optimize(
            rows,
            budget,
            workspace=workspace,
            seed=seed,
            ratios=ratios,
            backend=selected_backend,
            resources=resources,
            _context={"mode": "prepare", "prepare_only": True},
        )
        return PreparedRun(
            program=self,
            rows=rows,
            resources=resources,
            backend=selected_backend,
        )

    def _reopt_root(self, sha: str) -> Path:
        """A workspace path for re-optimizing on reviewed logs.

        The bound workspace's split is fixed forever, so reviewed-log rows
        get a sibling workspace: the first ``<name>_reopt*_ap`` path that is
        either free or already holds exactly this reviewed data — repeating
        the same call resumes that run instead of piling up directories.
        """
        parent = self._workspace.root.parent
        k = 1
        while True:
            suffix = "_reopt_ap" if k == 1 else f"_reopt{k}_ap"
            candidate = parent / f"{self.schema.name}{suffix}"
            if not candidate.exists() or self._holds_run(candidate, sha):
                return candidate
            k += 1

    def _holds_run(self, root: Path, sha: str) -> bool:
        """Whether ``root`` is this program's workspace for exactly this data."""
        try:
            active = json.loads((root / "active.json").read_text())
            split = json.loads((root / "data" / "split.json").read_text())
        except (OSError, ValueError):
            return False
        return (
            active.get("program") == self.schema.name
            and split.get("data_sha") == sha
        )

    def _resolve_rows(self, source) -> list[dict]:
        """Turn optimize()'s data argument into normalized rows."""
        if isinstance(source, str) and source == "logs":
            raise DataDisciplineError(
                'optimize(data="logs") was refused: logs record what the current '
                "program predicted, and optimizing toward your own outputs "
                "reinforces your own errors — re-optimization needs a correction "
                "signal. Run .review_logs() to accept/correct/reject sampled "
                'entries, then optimize(data="logs:reviewed") to train on only '
                "the corrected entries. (To compress the current behavior into "
                "something cheaper, use .distill() — imitation is the one job "
                "raw logs are perfect for.)"
            )
        if isinstance(source, str) and source == "logs:reviewed":
            if self._workspace is None:
                raise NotOptimizedError(
                    'optimize(data="logs:reviewed") was refused: this program has '
                    "no workspace bound, so there are no logs to read. Bind the "
                    'workspace that holds the logs with .use("<name>_ap") (or '
                    "optimize() with real data first)."
                )
            from . import logs as logs_mod

            entries = logs_mod.read_reviewed(self._workspace)
            if not entries:
                raise DataDisciplineError(
                    'optimize(data="logs:reviewed") was refused: no reviewed log '
                    "entries exist yet, and unreviewed logs cannot train a better "
                    "program (they only echo the current one). Run .review_logs() "
                    "to accept or correct sampled entries first."
                )
            return logs_mod.logs_to_rows(entries, self.schema)
        from . import data as data_mod

        return data_mod.normalize_rows(source, self.schema)

    @staticmethod
    def _load_final_report(ws):
        """Reconstruct the FinalReport persisted by a finished run."""
        from .harness import FinalReport

        d = json.loads(ws.final_report.read_text())
        return FinalReport(
            entries=d["entries"],
            activated=d.get("activated"),
            val_reliability=d.get("val_reliability", "ok"),
            frontier=d.get("frontier", []),
        )

    # ----------------------------------------------------------------- save

    def save(self, path) -> Path:
        """Move the workspace to ``path`` and repoint this program at it.

        Saving to the directory the workspace already lives in (e.g. the
        README's ``optimize()`` then ``save("<name>_ap")``) is a no-op —
        the workspace is already exactly there.
        """
        if self._workspace is None:
            raise NotOptimizedError(
                "save() was refused: this program has no workspace yet, so there "
                "is nothing to save. optimize() creates a workspace (or .use() "
                "binds an existing one)."
            )
        target = Path(path)
        if target.expanduser().resolve() == self._workspace.root:
            return target  # already saved there — the workspace IS this directory
        if target.exists():
            raise WorkspaceError(
                f"save() was refused: {target} already exists, and overwriting it "
                f"could destroy another workspace's candidates and scores. Choose "
                f"a fresh directory name or remove the existing one first."
            )
        if not target.name.isidentifier():
            raise WorkspaceError(
                f"save() was refused: {target.name!r} is not a valid Python "
                f"identifier, and the workspace is an importable package — name "
                f'it like "{self.schema.name}_ap" (underscores, no dashes).'
            )
        from .workspace import Workspace

        shutil.move(str(self._workspace.root), str(target))
        self._workspace = Workspace.load(target)
        return target

    # ----------------------------------------------------------------- call

    def __call__(self, *args, **kwargs):
        """Run the ACTIVE candidate on the given inputs, like a normal call."""
        if self._workspace is None:
            raise NotOptimizedError(
                f"{self.schema.name}() has no implementation yet — nothing has "
                f"been optimized. Run .optimize(data=..., budget=ap.Budget(...)) "
                f'first, or bind an existing workspace with .use("<name>_ap").'
            )
        active = self._workspace.active.get("active")
        if not active:
            raise NotOptimizedError(
                f"{self.schema.name}() cannot run: workspace "
                f"{self._workspace.root} has no active candidate — candidates "
                f"may exist, but none was activated. Run finalize() via the "
                f"agent harness (ap.attach) or optimize() again to completion."
            )
        bound = self._bind_inputs(args, kwargs)
        inputs = self.schema.coerce_inputs(bound)

        from . import candidates as candidates_mod
        from . import runner as runner_mod

        cand = candidates_mod.load_candidate(self._workspace, active)
        result = runner_mod.run_candidate(self._workspace, cand, inputs)
        if not result.ok:
            raise RunnerError(
                f"Active candidate {active!r} failed on this input. The shipped "
                f"program should never do this — inspect the trace below, then "
                f"re-optimize or activate a different candidate.\n{result.trace()}"
            )
        if self._logging:
            from . import logs as logs_mod

            logs_mod.append_log(
                self._workspace, inputs=inputs, outputs=result.outputs,
                candidate=active, n_repeat=1,
            )
        return self.schema.dict_to_outputs(result.outputs)

    def _bind_inputs(self, args, kwargs) -> dict:
        """Bind call arguments to the schema's input names like a plain function."""
        params = [
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in self.schema.input_names
        ]
        try:
            bound = inspect.Signature(params).bind(*args, **kwargs)
        except TypeError as exc:
            raise TypeError(f"{self.schema.name}() {exc}") from None
        return dict(bound.arguments)

    # -------------------------------------------------------------- logging

    def enable_logging(self) -> "Program":
        """Append every production call to <workspace>/logs/<UTC date>.jsonl."""
        self._logging = True
        return self

    def disable_logging(self) -> "Program":
        """Stop logging production calls."""
        self._logging = False
        return self

    def review_logs(self, sample: int | None = None) -> dict:
        """Review sampled log entries: accept / correct / reject each one."""
        if self._workspace is None:
            raise NotOptimizedError(
                "review_logs() was refused: this program has no workspace bound, "
                "so there are no logs to review. optimize() first (or .use() an "
                "existing workspace), enable_logging(), and gather traffic."
            )
        from . import logs as logs_mod

        return logs_mod.review_logs(self._workspace, sample=sample)

    # -------------------------------------------------------------- distill

    def distill(self, model: str, data: str = "logs", output=None,
                budget: Budget | None = None, *, backend=None):
        """Compress the current program into a cheaper imitation.

        Raw, UNREVIEWED logs are the right data here on purpose: the goal is
        imitation, so the program's own outputs are the training target. The
        distilled program lands in a new workspace (default
        ``<name>_distilled_ap``) and this program stays untouched.
        """
        if self._workspace is None:
            raise NotOptimizedError(
                "distill() was refused: this program has no workspace, so there "
                "is no logged behavior to imitate. optimize() first, "
                "enable_logging(), gather production traffic, then distill."
            )
        if not isinstance(budget, Budget):
            raise BudgetError(
                "distill() was refused: no budget was given. Distillation runs "
                "the same optimizer — candidates still cost money and time to "
                "score — so say what you're willing to spend, e.g. "
                "budget=ap.Budget(dollars=5)."
            )
        if data == "logs":
            from . import logs as logs_mod

            entries = logs_mod.read_logs(self._workspace)
            hint = "call .enable_logging() and gather production traffic first"
        elif data == "logs:reviewed":
            from . import logs as logs_mod

            entries = logs_mod.read_reviewed(self._workspace)
            hint = "run .review_logs() to accept or correct sampled entries first"
        else:
            raise DataDisciplineError(
                f"distill(data={data!r}) was refused: distillation imitates the "
                f'program\'s own logged outputs, so it accepts only "logs" (the '
                f'default) or "logs:reviewed". To optimize against other data, '
                f"use optimize() on a program instead."
            )
        if not entries:
            raise DataDisciplineError(
                f"distill() was refused: there are no log entries to imitate — "
                f"{hint}."
            )
        from . import logs as logs_mod

        rows = logs_mod.logs_to_rows(entries, self.schema)
        target = output if output is not None else f"{self.schema.name}_distilled_ap"
        clone = Program(self._fn)
        return clone.optimize(
            rows, budget, workspace=target, backend=backend,
            _context={
                "mode": "distill",
                "target_model": model,
                "parent": str(self._workspace.root),
            },
        )


class PreparedRun:
    """A fixed-data run paused between proposal and implementation search."""

    def __init__(self, *, program: Program, rows: list[dict], resources, backend):
        self.program = program
        self.rows = list(rows)
        self.resources = resources
        self.backend = backend

    @property
    def workspace(self):
        return self.program.workspace

    @property
    def metric_proposal(self) -> dict | None:
        if self.workspace is None:
            return None
        path = self.workspace.root / "metric_proposal.json"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def show_metric_suite(self) -> dict | None:
        """Return and print the orchestrator's proposal for user review."""
        proposal = self.metric_proposal
        if proposal is None:
            print("[autoprogramming] no metric proposal is recorded yet")
        else:
            print(json.dumps(proposal, indent=2))
        return proposal

    def demonstrate_metrics(self, examples: list[tuple]) -> list[dict]:
        """Run every proposed lens on user-chosen predicted/expected pairs."""
        if self.workspace is None:
            raise WorkspaceError("The prepared run has no workspace.")
        from .metric import demonstrate

        rows = demonstrate(self.workspace, examples)
        print(json.dumps(rows, indent=2, default=str))
        return rows

    def approve_metrics(self, approved_by: str, *, weights=None):
        """Approve the recorded metric roles and precommitted policy."""
        proposal = self.metric_proposal
        if proposal is None:
            raise WorkspaceError("There is no metric_proposal.json to approve.")
        from .objectives import MetricSuite, approve_suite

        suite = MetricSuite.from_dict(proposal["suite"])
        approve_suite(self.workspace, approved_by, suite, weights=weights)
        return self

    def optimize(self, budget: Budget):
        """Dispatch the portfolio against the already-fixed rows and workspace."""
        return self.program.optimize(
            self.rows,
            budget,
            workspace=self.workspace.root,
            backend=self.backend,
            resources=self.resources,
            _context={"mode": "optimize"},
        )
