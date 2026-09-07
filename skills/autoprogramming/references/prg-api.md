# `prg` — the agent-side API for driving an optimization workspace

Two names, one workspace, two trust levels: the user's `Program` can only run
the active candidate; `prg` (an `AgentHarness`) can create and score
candidates — always through harness-enforced guards. Every rule below is
enforced by the library (violations raise, with the reasoning in the message);
none of it depends on your good manners.

## Attach

```py
import autoprogramming as ap
prg = ap.attach("translate_ap")     # any workspace path; returns an AgentHarness
```

Raises `WorkspaceError` when the directory does not exist, has no
`active.json`, or has no `schema.py`.

## Inspect

```py
prg.schema        # Schema (immutable): .describe() prints inputs/outputs + docstrings,
                  # .input_names, .output_names, .expected_columns
prg.workspace     # the Workspace object (.root, .candidates_dir, ...)
prg.budget        # dict of remaining headroom: {"dollars": ..., "eval_calls": ..., "minutes": ...}
                  # None where no limit was set; floored at 0
prg.data          # SplitView(train=<n>, val=<n>)
```

### `prg.data` semantics

- `prg.data.train` — fully readable `Rows`: `len()`, iteration, indexing and
  slicing (returns dict copies), `.columns`, `.sample(k, seed=0)`
  (deterministic). Reflect here, freely.
- `prg.data.val` — `GuardedRows`: **`len()` only**. Iterating or indexing
  raises `DataDisciplineError`. Val exists for selection; you never see why a
  val row scored low, only aggregates, so you cannot edit candidates to fix
  specific val examples.
- There is **no `prg.data.test`** attribute at all, and no
  `prg.eval(split="test")`. Test belongs to `finalize()` — evaluated exactly
  once, at the end. Resource-confirmed runs store val/test in controller-private
  storage outside the coding-agent workspace; legacy/manual workspaces retain
  their CSV layout.

## Same-session Pi host orchestration

When optimization starts from a live Pi conversation, that human-facing session
is the sole strategy orchestrator. The library does not launch another strategy
process. After metric approval, the current agent performs and records research:

```py
prg.web_search(query: str, *, limit: int = 6) -> SearchReport
```

Task-derived web queries require confirmed external egress. Run at least two
distinct queries yielding at least two sources; private examples are never added
automatically. Evidence is persisted under `.ap/controller/research.json`.
Planning before that gate raises `WebResearchError`:

```py
prg.plan_portfolio(
    specs: list[ap.AvenueSpec | dict], *, exclusions=None, policy=None
) -> dict
```

The host-authored plan is validated/fill-completed by the deterministic breadth
policy. Then drive one controller phase at a time:

```py
prg.orchestrate_portfolio("breadth", budget=ap.Budget(dollars=20))
state = prg.portfolio_status()  # complete source/audit/failure/objective evidence
prg.orchestrate_portfolio("deepen", avenue_ids=["specialized-model"])
prg.orchestrate_portfolio("compose")
```

These calls may launch implementation workers and independent auditors, never a
strategy orchestrator. A supplied phase budget replaces the total limits while
preserving all prior spend; later phases may pass a larger explicit total budget
or use remaining headroom. Inspect `state` in the same conversation, ask the human
about unresolved blockers, and call `finalize()` explicitly only when
`state["may_finalize"]` is true.

Worker crashes, malformed source, cross-family substitutions, all-run errors,
and suspicious zero outputs are implementation evidence—not family outcomes.
They trigger bounded same-mechanism repairs and a materially independent worker
configuration. `FAILED`/`NONCOMPLIANT` never satisfy breadth; only evaluated
faithful implementations or human-confirmed infeasibility do.

Pi workers default to the exact host provider/model/thinking level. The host can
set an avenue's `worker_model` to another exact pattern in the discovered
`resources.search.pi_models` registry when a stronger specialist is appropriate.
Stored Pi OAuth is resolved by Pi itself, so no raw SDK key is required. A model-call candidate
may invoke persistent Pi CLI/RPC and is reported as requiring Pi plus an
authenticated subscription at deployment.

If the user supplied `SearchResources.remote_compute`, heavy worker operations
and candidate evaluation are staged there. Pi-runtime candidates remain beside
the authenticated host, so OAuth credentials are never copied remotely.
GPU-heavy avenues are admitted under a per-target concurrency/VRAM lease (one
concurrent GPU job by default). Remote compute is optional and is never inferred
or hard-coded by the skill.

## Metric first: `propose_metric`

No scoring happens until the workspace's `metric.py` is approved. Metrics are a
**set of lenses**: every candidate is scored on all of them from a single run,
so extra quality metrics cost ~no model calls.

```py
prg.propose_metric(code: str, examples: list[tuple],
                   note: str = "", primary: str | None = None) -> bool | str
```

- `code` — full source of `metric.py`, in either form:
  - **single** quality metric — `def metric(predicted, expected)`. For a
    single-output program it returns a float (called with the bare output
    values); for a multi-output program a dict keyed by output type names — one
    score per output, no field may be missing. Its objective name is `"quality"`.
  - **multi** — `METRICS = {"<name>": fn, ...}`, a dict of named quality metrics,
    each following the same per-metric contract. Names must be non-empty, unique,
    and must not collide with the cost objectives `cost_dollars` / `latency_s`
    (else `SchemaError`). If both `metric` and `METRICS` are present, `METRICS`
    wins.
- `examples` — `(predicted, expected)` pairs (bare values for single-output,
  dicts for multi-output) scored under **every** quality metric and shown as a
  demonstration table (one column per metric, primary marked). Pick 3+ that show
  their judgment: an exact match, a near miss (synonym/rounding), a clear failure.
- `primary` — the compatibility headline used by top-level report fields and
  the default `compare()` objective. For legacy approvals it also drives val
  ranking and activation. Approved metric suites instead select from their
  acceptance frontier under the precommitted policy. `None` lets the library
  pick (the sole metric, else the first `METRICS` key).
- Returns `True` once approved (interactive terminal sign-off, or the
  `AP_AUTO_APPROVE_METRIC` env var — see below). Returns the user's feedback
  **string** when they push back: revise the code and propose again. Raises
  `MetricNotApprovedError` when stdin is not a terminal and auto-approve is
  off — in that case surface the demonstration to the user and record their
  sign-off with:

```py
import autoprogramming as apm
# Legacy/all-acceptance approval:
apm.metric.approve(prg.workspace, "user name", weights=None, primary="chrF")

# Suite-aware approval (preferred for orchestrated search):
suite = apm.MetricSuite(
    acceptance=("chrF",),
    diagnostic=("exact",),
    policy=apm.SelectionPolicy(
        floors={"chrF": 0.75}, preference_order=("chrF",)
    ),
)
apm.approve_suite(prg.workspace, "user name", suite, weights=None)

# primary — compatibility headline; must name a defined metric (or None).
# weights (multi-output only): {"Answer": 2.0, "Confidence": 1.0} — the per-field
#   aggregate weighting, part of the sign-off; must sum positive. Aggregate score
#   is the weighted mean of the per-field scores.
```

Facts that matter:

- Approval stamps line 1 of `metric.py`
  (`# metric.py — approved by <who> on <date>`) and pins the file's sha in
  `metric_approval.json`. The stamp line is excluded from the sha, so
  re-approving never looks like a metric change.
- **Editing a metric's code re-scores from cached outputs, it does not wipe.**
  On a `metric.py` code change, every candidate whose outputs are still cached is
  re-scored under the new metric for **free** (no re-runs, no budget charge); only
  candidates whose own code changed (stale cache) or that were never cached are
  archived to `scores.archive/<k>.json` and warned about. Cost/latency objectives
  survive untouched. The edit still voids the approval (`MetricChangedError` on the
  next scoring attempt until re-approved). Re-proposing byte-identical code is a
  no-op that keeps everything.
- **Headline primary and weights are config, not code.** Changing either never
  changes `metric_sha` and so never invalidates cached outputs. Suite acceptance
  roles, floors, and preference order are additionally frozen once val selection
  starts; changing those requires a fresh workspace.
- `AP_AUTO_APPROVE_METRIC=1` (or `true`/`yes`) auto-approves an unapproved
  metric for unattended runs. It never re-approves a metric that changed after
  sign-off. Do not set it to dodge the sign-off conversation with a present user.

## Create candidates: `new_candidate`

```py
prg.new_candidate(source: str | None = None, from_: str | None = None) -> Candidate
```

Exactly one of `source` (fresh file contents) or `from_` (name of an existing
candidate to copy) — both or neither is a `CandidateError`. Writes
`candidates/candidate_<max+1>.py` (starting at `candidate_0`); metadata is
validated before the file lands. Candidates are plain files — after creating
via `from_`, edit the new file directly like any Python file.

### Candidate file conventions (PEP 723 single-file scripts)

Each candidate defines `predict(<input parameter names>)` returning:
- the bare output value (single-output program), or
- a tuple in schema order (multi-output), or a dict keyed by output names.

Values are coerced to the builtin bases of the schema types on the way out.

```py
# candidates/candidate_0.py
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.0", "translate-ap"]
#
# [tool.uv.sources]
# translate-ap = { path = "..", editable = true }
#
# [tool.ap]
# cost_per_call = 0.0002
# ///
from openai import OpenAI
from translate_ap.schema import French

_client = None

def predict(english: str) -> French:
    global _client
    if _client is None:
        _client = OpenAI()
    ...
```

- **No work at import time.** Clients and models load lazily inside
  `predict()`; importing the package must never need an API key or network.
- **Dependencies** go in the `# /// script` block (at most one per file, valid
  TOML). A stdlib-only candidate needs no block at all — and runs on the fast
  path without uv. Dep-bearing candidates run under `uv run --no-project` in
  their own ephemeral environment, so conflicting deps across candidates
  coexist; `uv` must be on PATH or the run raises `RunnerError`.
- **The self-reference** (`"translate-ap"` plus its `[tool.uv.sources]` entry)
  lets the candidate import `<pkg>.schema` / `<pkg>.paths` when run standalone
  (`uv run candidates/candidate_0.py`). At harness eval time it is stripped
  and the package is injected via `sys.path` instead, so evals stay hermetic.
- **`[tool.ap]` keys** (all optional):
  - `deterministic = true` — the candidate makes no stochastic calls; it is
    scored with 1 repeat instead of 3, saving budget. Default false.
  - `cost_per_call = <float>` — flat dollar cost charged per run when the run
    does not report its own cost.
  - `pi_runtime = true` — `predict()` invokes Pi CLI/RPC and therefore must run
    beside the authenticated host rather than copying OAuth to remote compute.
  - `fetch = ["huggingface:...", ...]` — declared artifact download steps
    (recorded candidate metadata for heavyweight models).
- **`AP_COST_DOLLARS`** — a candidate that spends money per call should set
  this module global after each `predict()` (e.g.
  `AP_COST_DOLLARS = tokens * rate`); the runner reads it after the call and
  charges the ledger with it, overriding `cost_per_call`.
- **Artifacts** (weights, pickles, lookup tables) belong in the workspace's
  `artifacts/` directory, resolved as
  `from <package>.paths import artifacts` (a `pathlib.Path`).
- Each run executes in a subprocess with the env var `AP_WORKSPACE` set to the
  workspace root, a default timeout of 120 s (a timeout kills the whole
  process group and scores 0), and results reported through a file — stdout
  cannot forge an outcome. A `predict()` exception is a failed run scoring
  0.0, never a crash of the harness.

## Score: `eval`

```py
prg.eval(candidate, split="val", per_instance=False, n_repeats=None) -> EvalReport
```

- `split="val"` (default): selection. Aggregate only — `per_instance=True` on
  val raises `DataDisciplineError`. Each distinct candidate scored on val is
  registered as selection pressure (see guards below).
- `split="train"`: reflection. `per_instance=True` returns `per_row`
  (`{"row_0": score, ...}`) so you can find the worst rows.
- `split="test"`: always refused.
- `n_repeats`: default 1 for `deterministic = true` candidates, else 3; the
  score per row is the mean over repeats. Must be >= 1 when given.
- The budget is checked before every row; a mid-eval `BudgetExceededError`
  propagates with nothing persisted for the partial eval. Every run charges
  `eval_calls` and dollars to the ledger.
- Metric must be approved or this raises before any spend.

Every eval is **multi-objective**: the single run per (row, repeat) is scored
under every quality metric, and `cost_dollars` + `latency_s` are read off the
same run (lower is better). Extra quality metrics therefore add no runs.

`EvalReport` fields — for compatibility, the top-level ones are the headline
**primary** quality metric's: `candidate`, `split`, `mean`, `std`,
`ci95: (lo, hi)` (seeded bootstrap over per-row means), `n_rows`, `n_repeats`,
`repeat_variance`, `per_row` (train + `per_instance=True` only, else `None`),
`per_field` (multi-output aggregates or `None`), `errors` (failed-run
summaries; failed runs score 0.0 on every quality metric), `flags` (memorization
flags, see below). Plus:

- `primary` — the name of the headline quality metric.
- `objectives: dict[str, dict]` — every objective (each quality metric, plus
  `cost_dollars` and `latency_s`), each `{"mean", "std", "ci95", "per_field"}`.

`print(report)` shows the primary headline line, then one indented line per other
objective with its goal (`max` for quality, `min` for cost/latency) and CI —
cost as dollars, latency in seconds.

## Trace: `run`

```py
prg.run(candidate, split="train", row=0) -> TracedRun
```

The reflection primitive: one full traced run of a single **train** row —
inputs, outputs or traceback, stdout/stderr, timing, cost, expected values,
and the metric score (`None` if the metric is not yet approved; 0.0 for a
failed run). Any split other than `"train"` raises `DataDisciplineError`; an
out-of-range row raises `IndexError`. Charges 1 eval call.

## Compare: `compare`

```py
prg.compare(a, b, split="val", objective=None) -> CompareReport
```

Paired comparison over per-row scores **already stored** by `eval()` — never
triggers new spend. `a` is the baseline, `b` the challenger; `diff_mean` is
the mean of `b - a` per row; `ci95` is a seeded paired-bootstrap CI;
`improved` is direction-aware — `True` only when the CI excludes zero on the
better side: entirely **above** zero for a maximized quality metric, entirely
**below** zero for a minimized `cost_dollars` / `latency_s` objective (a
reliably cheaper or faster challenger). `CompareReport.goal` records which. Raises
`DataDisciplineError` when either candidate has no stored per-row scores for
that split, or their stored rows share no ids — eval both first. (Per-row val
scores are persisted internally for exactly this pairing; they are never
returned to you.)

- `objective=None` (default) compares the **primary** quality metric — the
  back-compat default.
- `objective="<name>"` compares any other objective's stored per-row scores: a
  named quality metric, or `"cost_dollars"` / `"latency_s"`. The refusal message
  names the objective if either candidate lacks stored scores for it.

**"Improved" means the CI excludes zero.** A point estimate going up is not a
win; do not keep a mutation `compare()` cannot distinguish from noise. For a
minimized cost objective, `improved=True` means the CI of `b - a` is entirely
**below** zero: the challenger is reliably cheaper/faster. Unknown cost cannot
be compared and is never treated as zero.

## Tradeoffs: `tradeoffs`

```py
prg.tradeoffs(split="val") -> TradeoffReport
```

The quality/cost Pareto frontier over every candidate with stored objective
vectors for the split. `rows` is one dict per candidate
(`{"candidate", "objectives": {name: mean}, "dominated": bool}`), sorted by the
primary metric descending; `nondominated` lists the frontier — candidates no
other candidate beats on **every** objective (quality maximized, cost minimized).
`print(report)` renders a compact table, one column per objective, a `*` on
frontier members. Read it during the loop to see where each candidate sits on the
quality/cost curve and drop the dominated dead ends. Cost as dollars, latency in
seconds.

## Survey: `frontier`

```py
prg.frontier() -> FrontierReport
```

True Pareto view over stored per-row **train** scores: `rows` maps each row id
to its best score and the candidates achieving it; `nondominated` lists
candidates no other candidate dominates (>= on every shared row, > on one);
`missing` lists candidates with no per-row train scores yet (eval them with
`split="train", per_instance=True` to include them). Complementary candidates
that win different rows are prime material for a merged/pipeline candidate.

## Resolve an environment or implementation blocker: `resolve_blocker`

```py
prg.resolve_blocker(avenue_id, action, confirmed_by="user")
```

After bounded same-family diagnosis/repair and an independent configuration, an
ambiguous implementation or environment failure (missing access, package, model,
GPU, network, remote transport, etc.) pauses the portfolio. It does **not**
accept a cross-family fallback, count broken code as breadth, or silently discard
the approach. Check with the human first: they may be able to provision/fix it.

- `action="retry"` — the human fixed the capability or explicitly requests
  another attempt. This grants one retry even when conservative preflight still
  reflects the old environment snapshot.
- `action="exclude"` — only after the human confirms the approach really is
  unavailable. The avenue becomes an explicit infeasibility, not a fake result.
- `confirmed_by` is required so the controller records who made the decision.

Then resume the original `optimize()` call with a new explicit budget.

## Finish: `finalize`

```py
prg.finalize(top_k=None) -> FinalReport
```

The **only** code path that reads the test split, run once per workspace:

1. Requires an approved metric and at least one val-scored candidate
   (else `NotOptimizedError`). Memorization-flagged candidates are excluded;
   if every val-scored candidate is flagged, finalize refuses.
2. Legacy metrics take the top two eligible candidates by val mean. Approved
   metric suites instead take up to `policy.max_test_finalists` candidates from
   the acceptance/cost val Pareto frontier, preferring distinct approach tiers;
   explicit `top_k` overrides the cap. Evaluates each on every test row (1 repeat
   if deterministic, else 3). Charges the budget but never checks it.
3. Scores every finalist on **all objectives** from the shared test runs
   (quality metrics + `cost_dollars` + `latency_s`) and computes the Pareto
   frontier over those TEST objective vectors.
4. Demotes any finalist whose val-to-test drop on the compatibility headline
   exceeds `max(0.05, half the width of its val CI)` — that is overfitting to
   val, reported loudly, never hidden.
5. For suites, activates the non-demoted test-frontier point chosen by the
   precommitted acceptance preference, then cost/latency tie-breakers. Legacy
   workspaces retain best-primary activation. If every finalist is demoted, the
   report uses the corresponding frontier fallback with a loud note. Writes
   `final_report.json` and rewrites `active.json` + `pyproject.toml` so the
   shipped package runs the winner with exactly its runtime deps.

`FinalReport`: `entries` (per finalist: `candidate`, `val_mean`, `test_mean`,
`gap`, `demoted`, `note` — all **primary**, back-compat — plus
`objectives: {name: mean}` and `frontier: bool`), `activated`,
`val_reliability` (`"ok"` / `"warn"` / `"unreliable"`), and `frontier`
(the list of nondominated finalists). `print(report)` keeps the legacy
"test scores (evaluated once):" and "activated:" lines, then appends a
"quality / cost tradeoffs:" table (a `*` on frontier members) and a closing line
naming the activated default and the cheaper/faster frontier alternatives with
the exact `ws.activate("<name>", <headline test mean>)` call to switch. Present
that frontier to the user — do not hand them only the single most-accurate
candidate.

Calling `finalize()` a second time raises `FinalizedError` — a second pass
would let the report card steer the search and turn test into another val
set. Do not skip finalize either: an un-finalized workspace ships nothing.
Finalization also removes the orchestrated run's owned local and remote UV
caches and disposable `.venv*` package environments. It preserves all worker
source, artifacts, outputs, and inactive-candidate diagnostics.

## Remove worker package caches: `cleanup_search_cache`

```py
prg.cleanup_search_cache()            # finalized workspace
prg.cleanup_search_cache(force=True)  # cache-only cleanup for a paused workspace
```

Paused and blocked runs keep their package environments by default. Forced
cleanup removes only owned UV caches and task-local `.venv*` directories; package
setup can rebuild them later. It never removes worker files, candidates, declared
artifacts, cached outputs, or inactive-candidate diagnostics. UV cache created by
versions before this lifecycle was introduced can be pruned once with
`uv cache prune`.

## Guards you will hit (by design)

| Action | Result |
|---|---|
| `eval(split="test")`, or any unknown split | `DataDisciplineError` |
| `eval(split="val", per_instance=True)` | `DataDisciplineError` |
| `run()` on a val/test row | `DataDisciplineError` |
| Iterating/indexing `prg.data.val` | `DataDisciplineError` |
| Scoring with unapproved / changed metric | `MetricNotApprovedError` / `MetricChangedError` |
| 6th distinct candidate on val in bootstrap mode (< 30 examples) | `BootstrapModeError`, raised before any spend |
| Any charge once a limit is hit | `BudgetExceededError` (finalize still runs) |
| Environment-blocked avenue | Portfolio pauses for human `retry` / confirmed `exclude`; no fallback is accepted |
| Second `finalize()` | `FinalizedError` |

Soft signals (warnings, not errors):

- **Val selection pressure** — once more distinct candidates than val rows
  have been compared, a `ValReliabilityWarning` fires (`warn`); past 3x the
  val size, val scores are declared `unreliable` and will not be reported as
  final. Prefer few, genuinely different candidates over many small tweaks.
- **Memorization** — once a candidate has both train and val aggregates, it is
  flagged (`MemorizationWarning`, recorded in `scores.json` flags and
  excluded from selection) when train mean exceeds val mean by more than 0.2
  with train mean above 0.5, or when at least `max(3, 10% of train rows)`
  distinct train string outputs (length >= 8) appear verbatim in the candidate
  source. A regex/rules candidate may legitimately win — but only on data it
  never saw.

## The loop

0. **Metric-suite sign-off** — propose and demonstrate every lens before any
   eval, classify acceptance vs diagnostic roles, and precommit floors and
   preference order. `propose_metric(..., primary=...)` remains the legacy
   all-acceptance path. Add a graded diagnostic if a strict lens is flat.
1. **Survey the ladder, seed a diverse portfolio.** Check current tools (do not
   trust remembered "latest" models), then create several genuinely distinct
   candidates across cost tiers — a model call, a classical head, a rules
   baseline, a pretrained-model-as-a-stage pipeline — breadth before depth.
2. **Baseline the portfolio on val**, then read `prg.tradeoffs()` to see where
   each lands on the quality/cost frontier. Drop dominated dead ends; record why
   a family was dropped as a scored result, not a hunch.
3. **Deepen the promising tiers.** Compose (cascade, ensemble, router,
   learned-feature + classical-head), tune, and try a second config before
   dismissing a family. Reflect on train: `prg.eval(name, split="train",
   per_instance=True)` for the worst rows, `prg.run(name, split="train", row=i)`
   for their traces, `prg.frontier()` for which candidate wins which rows. Keep a
   mutation only if `prg.compare(best, new).improved` is `True`.
4. **Watch the frontier, not one number.** Check `prg.budget` between iterations
   and stop when the quality/cost frontier stops advancing (a new candidate no
   longer joins or displaces it), not merely when a single metric plateaus.
5. **`prg.finalize()`** — one-time test eval, activation of the precommitted
   frontier default (best-primary only for legacy approvals), sealed report.
   Present the tradeoff frontier and the one-liner to switch alternatives.
