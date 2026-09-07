---
name: autoprogramming
description: Build and optimize a typed Python program or prompt from examples or labeled input/output pairs with the autoprogramming library — DSPy/GEPA-style prompt optimization generalized to whole implementations. Use when the user wants to turn example pairs into an optimized program, optimize or evolve a prompt against a metric, distill a program into a cheaper model, or improve a deployed program from production logs. Covers defining @ap.program schemas, gathering data, proposing and getting sign-off on the evaluation metric, setting an explicit ap.Budget, running optimize()/distill(), driving the candidate loop yourself via ap.attach, and reading results honestly (confidence intervals, overfit demotion, train/val/test discipline).
---

# AutoProgramming — conversational front-end

You are the front-end for the `autoprogramming` library inside the user's own
project. The user describes a task ("I want a translator", "classify these
tickets"); your job is to turn the conversation into four concrete artifacts —
a typed `@ap.program` definition, the data, a user-approved metric, and an
explicit `ap.Budget` — and then call `optimize()` exactly like any library
user would. A coding-agent search then evolves complete candidate
implementations (plain `.py` files: LLM calls, regex, scikit-learn, local
models, pipelines) under a strict train/val/test discipline, and the output is
a portable Python package with zero runtime dependence on the optimizer.

## Install

```sh
uv add "git+https://github.com/lillyguisnet/autoprogramming.git"
# or: pip install "git+https://github.com/lillyguisnet/autoprogramming.git"
```

- Requires Python >= 3.12.
- `uv` must be on PATH at evaluation time: candidates that declare third-party
  dependencies run under `uv run --no-project` in their own ephemeral
  environments (conflicting deps coexist). Stdlib-only candidates run without uv.

## The conversation: six questions

Ask these, in order, before writing any code. Everything else the library and
the optimizer figure out.

1. **What goes in, what comes out?** "English in, French out" is enough.
   Becomes the `@ap.program` definition (below).
2. **Do you have examples?** 30+ pairs unlocks full optimization; 5–29 runs in
   bootstrap mode (baselines only, at most 5 candidates compared — the
   library enforces this); fewer than 5 is refused outright. If none: offer to
   generate synthetic pairs the user validates.
3. **What does "good" mean?** Exact match? Same meaning? Latency? Cost per
   call? Usually more than one of these — that is fine, they become a *set* of
   metrics (below). This shapes the metric(s) you will propose.
4. **Sign off on the metric set.** THE critical step — the entire search
   optimizes whatever the metric says, so a wrong metric produces a
   confidently-scored wrong program. Never skip; never approve on the user's
   behalf. You may propose **several quality metrics at once** (a strict one and
   a graded one, say); every candidate is scored on all of them from a single
   run, so extra metrics cost ~nothing. Classify lenses as **acceptance**
   (user-approved and eligible to select the final program) or **diagnostic**
   (search guidance only). Precommit acceptance floors and preference order;
   do not average everything into one gameable scalar. Demonstrate every lens
   on real examples and iterate until the scores match the user's intuition:

   ```
   I propose chrF as an acceptance lens and exact-match as a strict diagnostic.
     expected:  "Où est la gare ?"
     predicted: "Où se trouve la gare ?"
         chrF: 0.81  <- acceptance   exact: 0.00  <- diagnostic
     predicted: "Où est la gare ?"
         chrF: 1.00                  exact: 1.00
     predicted: "La gare est fermée."
         chrF: 0.34                  exact: 0.00
   Does chrF 0.81 for the synonymous phrasing match your intuition of "good"?
   ```

   For multi-output programs the per-field aggregate weighting is also part of
   sign-off. Acceptance roles, floors, and preference order are frozen before
   val selection; diagnostic lenses may evolve and re-score cached outputs.
5. **What resources exist while SEARCHING?** CPU/RAM/disk, GPU + VRAM,
   package/model-download permission, fine-tuning services, authenticated Pi
   subscription models, candidate API providers, maximum parallel workers, and
   a conservative maximum dollars per agent call. Ask whether the user offers
   optional remote compute; never assume or hard-code one. If supplied, record
   its endpoint/capabilities with `ap.RemoteCompute` and prefer it for heavy
   installs, training, model loading, and evaluation while keeping lightweight
   orchestration local. Pi-runtime candidates stay beside the authenticated
   host—never copy OAuth credentials to remote compute. Default remote GPU
   concurrency is one to avoid VRAM contention.
6. **What may the SHIPPED PROGRAM require?** Runtime CPU/GPU/RAM/network,
   Pi plus a logged-in subscription, candidate API providers,
   latency/cost/artifact limits, and whether task data may leave the machine.
   Search may still measure Pi-backed approaches outside the preferred runtime
   envelope; label their deployment requirements clearly and let the user choose
   the frontier point. Never persist credentials—Pi OAuth remains in Pi's auth
   store and remote authentication remains in its transport configuration.

## Define the program

Types are subclasses of builtins; docstrings become descriptions the optimizer
uses. Schema errors (missing annotations, defaults, `*args`, generics/unions)
surface at decoration time.

```py
import autoprogramming as ap

class French(str):
    """Natural, idiomatic French. Formal register unless the input is casual."""

@ap.program
def translate(english: str) -> French:
    """Translate English text to French."""
```

Multiple outputs are a tuple of DISTINCT types — output names come from their
types, so two outputs of the same type is a `SchemaError`:

```py
class Answer(str):
    """Direct answer to the question, one sentence."""

class Confidence(float):
    """Calibrated probability that the answer is correct, 0.0-1.0."""

@ap.program
def qa(question: str) -> tuple[Answer, Confidence]:
    """Answer a factual question with a calibrated confidence."""
```

Data rows must cover every input (by parameter name) and every expected output
(by output type name): here `question`, `Answer`, `Confidence`. Accepted
sources: a list of dicts, a DataFrame (anything with `.to_dict("records")`),
or a path to a `.csv`/`.jsonl` file. Extra columns are dropped with a warning;
missing columns are a `SchemaError`.

## Budget — explicit, no default

`optimize()` and `distill()` refuse to run without one, because evaluation
spends real money and time (LLM candidates cost money to *score*, not just to
mutate). Ask the user what they are willing to spend:

```py
ap.Budget(dollars=20)                    # or eval_calls=2000, or minutes=30
ap.Budget(dollars=20, minutes=60)        # combinable; first limit hit stops the run
```

## Prepare, approve, research, then run the Pi portfolio

**In a live Pi conversation, the model speaking with the human is the one and
only strategy orchestrator. Never launch `PiOrchestratorBackend` from that
session:** doing so creates a second, context-poor orchestrator. The default
`HostOrchestratorBackend` detects the live session and pauses for this agent to
do metric design, web research, planning, failure inspection, and finalization.
The trusted Python controller—not another strategist—dispatches workers and
enforces policy.

```py
resources = ap.Resources(
    search=ap.SearchResources(
        max_parallel_agents=4,
        max_dollars_per_agent_call=0.05,
        # The active Pi provider/model/thinking level is captured automatically.
        candidate_api_providers=(),
        allow_package_installs=True,
        allow_model_downloads=True,
        # ONLY if supplied: ap.RemoteCompute(endpoint=..., transport="ssh", ...).
        # Ask for the transport; never assume that remote compute means SSH.
    ),
    runtime=ap.RuntimeResources(network=False),
    data=ap.DataPolicy(external_egress=True),  # required for task-derived web queries
    confirmed=True,
)
prepared = translate.prepare(
    pairs, resources=resources, budget=ap.Budget(dollars=2)
)
prg = ap.attach(prepared.workspace.root)
```

Propose metric code yourself in this conversation, demonstrate every lens on
real examples, obtain explicit user approval, then write/approve the suite. Next,
**search before planning**:

```py
print(prg.web_search("latest efficient models and methods for <abstract task>"))
print(prg.web_search("2026 open source <task> benchmark best approaches"))
# Inspect sources; never put private examples in a query.

prg.plan_portfolio(web_informed_avenue_specs)
status = prg.orchestrate_portfolio(
    "breadth", budget=ap.Budget(dollars=20)
)
# Inspect status: source/audits/setup/failures/objectives for every avenue.
status = prg.orchestrate_portfolio("deepen", avenue_ids=[...])
status = prg.orchestrate_portfolio("compose")
report = prg.finalize()  # only when status says may_finalize
```

`web_informed_avenue_specs` is the current orchestrator's task-specific list of
`ap.AvenueSpec` objects; each may cite `research_sources`. `plan_portfolio()`
refuses until at least two searches and two distinct sources are recorded.
Outside a live Pi session, a caller may explicitly choose the legacy headless
`PiOrchestratorBackend`; that fallback owns its own web-enabled strategy process.

The deterministic Python controller enforces breadth and dispatches parallel,
implementation-only Pi workers. Workers default to the exact active Pi model and
thinking level; an avenue may name another exact discovered `pi_models` pattern
when a stronger specialist is useful. Pi resolves stored OAuth/subscription authentication; absence
of `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` does not invalidate Pi access. A
Pi-backed candidate calls Pi CLI/RPC and is reported as requiring Pi plus the
logged-in subscription at deployment, so the user can accept or reject that
frontier point.

Workers never receive optimizer identity, metric names/code/weights, leaderboard
scores, other workers, val, or test. They see a generic function task, dev-fit
examples, one assigned mechanism, permitted resources, and their own prior files.
Worker `uv` operations use an AutoProgramming-owned cache under
`~/.cache/ap-work/<run>/`; do not create a new virtual environment for each
experiment. Reuse one task-local `.venv` only when package execution is necessary,
and keep required runtime files in the declared artifact namespace instead.

**Mechanism fidelity is absolute.** An avenue may fail, be blocked, or score
poorly, but it may never substitute another approach family to keep the function
working. Missing Torch, a model, a genuinely required raw API key, GPU, network,
or package is a setup failure—not permission to replace a deep/API avenue with
classical CV, rules, or
a lookup. Workers must declare dependencies in PEP 723 and fail clearly when a
required capability is truly unavailable. Before import, the controller performs
deterministic plus independent semantic adherence checks; rejected source is
repaired in-session and then clean-restarted, never scored as that avenue.

What happens mechanically:

- The data is normalized and **split exactly once** into train/val/test
  (default ratios 0.6/0.2/0.2, deterministic by `seed=0`). The split is pinned
  by a content hash (`data_sha`) and is never redone for that workspace.
- Fewer than 30 rows puts the workspace in **bootstrap mode**; fewer than 5 is
  refused.
- A workspace directory is created — default `<program name>_ap/` (e.g.
  `translate_ap/`) — which is itself a valid, installable Python package:
  `pyproject.toml`, `__init__.py` (runs the active candidate), `schema.py`
  (SHA-pinned), `active.json`, `data/` (dev train data; resource-confirmed
  runs keep val/test in controller-private storage outside the agent workspace),
  `candidates/` (one PEP 723 `.py` file
  each), `artifacts/` (weights, pickles), and `scores.json`. `metric.py` is
  NOT among the created files — it appears only after the metric is proposed
  and approved (the sign-off step below); until then any scoring attempt
  raises `MetricNotApprovedError`.
- With a confirmed `Resources` profile inside Pi, the current human-facing
  session remains the host orchestrator and uses the staged `prg` methods above;
  it never creates a second strategy agent. Only implementation workers and
  independent auditors get isolated Pi contexts. Outside Pi, headless strategy
  orchestration is explicit. Read [references/prg-api.md](references/prg-api.md)
  for the complete host/controller API.
- The metric set must be proposed, demonstrated on real examples, and approved
  before any scoring happens. Editing a metric's **code** later re-scores every
  candidate from its cached outputs (free — no re-runs) and archives only
  candidates whose own code/artifact bundle changed and can't be recovered.
  Diagnostic roles may evolve, but acceptance roles/floors/preference are
  precommitted before val and cannot be changed afterward.
- The controller loop is: require one avenue in every resource-feasible tier,
  dispatch those avenues to parallel Pi workers, baseline the portfolio on
  **val** and read the
  quality/cost frontier (`prg.tradeoffs()`), deepen and compose the promising
  tiers by reflecting on **train** failures (full traces allowed) and selecting
  on **val** (aggregate scores only), all within budget. **test** is
  untouchable until `finalize()` evaluates it once, at the end, on the top
  candidates, and activates the winner. See
  [references/prg-api.md](references/prg-api.md) for the full loop.
- A worker crash, malformed output, noncompliant source, dependency mistake, or
  suspicious zero result never satisfies breadth and never proves the approach
  failed. The controller diagnoses and repairs within the same mechanism, then
  uses a fresh Pi context for a materially independent configuration. Remote GPU
  work is admitted under VRAM/concurrency leases; contention/OOM is paused and
  retried exclusively rather than scored against the approach. Only evaluated
  faithful implementations or a human-confirmed infeasibility satisfy breadth.
- If an approach still appears blocked after bounded investigation, the
  controller pauses instead of accepting a fallback or discarding the tier.
  Ask the human whether they can fix it. After they do, use
  `prg.resolve_blocker("<avenue>", "retry", confirmed_by="user")`; only after
  explicit human agreement may you use `"exclude"`. Then resume `optimize()`.
- Returns a `FinalReport` when candidates were scored; `None` when the
  workspace is awaiting metric approval, blocker resolution, or a manual session.
- Successful finalization automatically removes only owned local/remote UV
  caches and disposable worker `.venv*` directories. Paused, blocked, and
  budget-exhausted runs retain them unless the user requests
  `prg.cleanup_search_cache(force=True)`. Candidate source, artifacts, outputs,
  and inactive-candidate diagnostics always remain available for downstream
  evaluation.

Afterwards the program is a normal function and a normal package:

```py
translate("Hello, how are you?")   # => French('Bonjour, comment allez-vous ?')
translate.save("translate_ap")     # relocate the workspace (no-op if already there)
# and: pip install ./translate_ap  — deps of the active candidate install automatically
```

## What the optimizer explores: the approach ladder

A candidate is any `.py` file that satisfies the schema, so the search space is
a whole cost/capability spectrum — from most expensive/capable to cheapest:

1. Generalist harness (a coding agent / reasoning model doing the task live)
2. A graph of several model calls (plan, act, critique, retrieve)
3. A single model call with an optimized prompt
4. A finetuned small model
5. A specialized deep net (a task-specific pretrained model)
6. Classical ML (a fitted scikit-learn head, gradient boosting, nearest neighbor)
7. Hand-written features, rules, regex, or lookup logic

The optimizer does not seed one idea and mutate it forever (the single-step
trap). It seeds a **diverse portfolio across tiers first**, baselines all of
them, then deepens the ones the data rewards — **breadth before depth**. It
**composes across tiers**: the best solution is usually a compound system, using
a heavy pretrained model as ONE STAGE (pipeline/decomposition, cascade, ensemble,
router, learned-feature + classical-head), not only end to end. It does not
dismiss a family on a single failed config, and it picks the **cheapest tier that
can plausibly clear the bar**, climbing only when the data shows that tier's
ceiling. Steer the user toward this: a rules candidate at 0.95 beats a model call
at 0.90 when the CI separates and it never memorized.

**Use current tools, not remembered ones.** Do not trust training memory for
"the latest" model or library — it is stale. Before fetching a model or picking a
package, check what is current (model hub, package index, a quick web search) and
prefer the current best-for-cost; verify a model actually loads before building a
candidate around it. The classic failure is reaching for an old version you
remember (SAM 1 when a newer, smaller, faster SAM exists) instead of today's.

## Cost is an objective — present the frontier, not one number

`latency_s` is measured on every run. `cost_dollars` is taken from the
candidate's `AP_COST_DOLLARS` report or declared `cost_per_call`; missing cost is
**unknown**, never silently `$0` (lower is better). Every eval reports them next to the
quality metrics, and `prg.tradeoffs()` shows the quality/cost Pareto frontier —
the candidates where no other beats them on every objective. The goal is best
quality PER cost, not perfection at any price.

`finalize()` activates the frontier point selected by the precommitted
acceptance preference and lists quality/cost frontier alternatives with the one-liner
to switch. When you report to the user, present the **frontier**: "here is the
cheap-good one, the mid one, and the expensive-best one" — and let them pick the
point on the curve, rather than only handing them the single most-accurate
candidate.

## Read results honestly

- **The test column is the report card.** `FinalReport` lists, per finalist,
  `val_mean`, `test_mean`, a note, and its per-objective means with a frontier
  flag. Test was evaluated exactly once; quote the test number, never the val
  number.
- **Overfit demotion.** A finalist whose test score drops far below its val
  score (gap beyond the val CI half-width, min 0.05) is marked
  "overfit to val, demoted". If every finalist is demoted, the report says so
  and activates the best test score anyway — tell the user to treat the
  numbers with suspicion.
- **Val reliability.** When val absorbed many selection decisions relative to
  its size, the report carries `val_reliability` of `"warn"` (val scores are
  noisy) or `"unreliable"` (val scores must not be quoted at all).
- **Every eval carries uncertainty.** `EvalReport` shows mean, std, a 95%
  bootstrap CI, and repeat variance (stochastic candidates are scored with 3
  repeats by default). "Improved" means a paired-comparison CI **excludes
  zero** — not that the point estimate went up. Report wins to the user in
  those terms.

## Production lifecycle

```py
translate.enable_logging()
# every call appends one JSONL entry to <workspace>/logs/<UTC date>.jsonl:
# {"inputs": {...}, "outputs": {...}, "candidate": "...", "n_repeat": 1, "timestamp": "..."}
```

Two different improvement paths — do not mix them up:

- **Distill** (make it cheaper, same behavior). Raw unreviewed logs are the
  RIGHT data here on purpose: the goal is imitation, so the program's own
  outputs are the training target. Budget is still required. The result lands
  in a new workspace (default `<name>_distilled_ap`); the original is untouched.

  ```py
  translate.distill(model="gpt-4.1-nano", data="logs", budget=ap.Budget(dollars=5))
  ```

- **Re-optimize** (make it better). Logs alone cannot do this — optimizing
  toward your own outputs reinforces your own errors, so
  `optimize(data="logs")` is refused by design. A human must add the
  correction signal first:

  ```py
  translate.review_logs()   # interactive: accept / correct / reject sampled entries
  translate.optimize(data="logs:reviewed", budget=ap.Budget(dollars=10))
  ```

  Only accepted/corrected entries become training data. Because the original
  workspace's split is fixed forever, re-optimization goes to a **sibling
  workspace** (`<name>_reopt_ap`, then `_reopt2_ap`, ...) automatically — and
  the call re-binds the program to that sibling. To resume the run later,
  first re-bind the ORIGINAL workspace (the one holding the reviewed logs),
  then repeat the call; it finds the sibling that already holds exactly that
  reviewed data and resumes it instead of creating another directory:

  ```py
  translate.use("translate_ap")   # the workspace with the reviewed logs
  translate.optimize(data="logs:reviewed", budget=ap.Budget(dollars=10))
  ```

  Repeating the call WITHOUT re-binding raises a misleading
  `DataDisciplineError` ("no reviewed log entries exist yet") — the program
  is then reading logs from the empty sibling, not from the original
  workspace. Re-bind with `.use()` as above; do not re-review logs.

## Refusals to expect — explain them, don't fight them

Every refusal in this library is deliberate and self-explaining. When one
fires, relay the reason to the user and take the suggested path; do not look
for workarounds.

| Refusal | Why | What to do |
|---|---|---|
| `optimize()`/`distill()` without a budget (`BudgetError`) | Evaluation spends real money; there is no default | Ask the user for `dollars`/`eval_calls`/`minutes` |
| Fewer than 5 rows (`DataDisciplineError`) | Nothing to select on, even in bootstrap mode | Gather more examples, or generate synthetic ones the user validates |
| `data_sha` mismatch when re-optimizing an existing workspace | The data was split once; re-splitting leaks val/test into train | Pass the original data, or point at a new workspace path |
| `optimize(data="logs")` (`DataDisciplineError`) | Unreviewed logs only echo the current program | `review_logs()` first, then `data="logs:reviewed"`; or `distill()` if the goal is compression |
| Scoring before metric sign-off (`MetricNotApprovedError`) | The search optimizes whatever the metric says | Run the demonstration + sign-off conversation with the user |
| Metric **code** edited after approval (`MetricChangedError`) | The old sign-off is void until re-approved; scores re-compute from cached outputs (only candidates whose own code changed are archived) | Re-demonstrate and re-approve — it is not a scary wipe |
| Adding a diagnostic metric | Re-scores cached outputs without candidate calls | Re-demonstrate/re-approve code; acceptance policy stays fixed |
| Changing acceptance roles/floors/preference after val began | Would choose policy after seeing selection results | Start a fresh workspace; the policy is precommitted |
| Bootstrap cap: 6th candidate on val (`BootstrapModeError`) | Differences on a tiny val set are one row of noise | Pick the best baseline already scored, or get to 30+ examples |
| Reading val rows, tracing val/test, eval on test, per-row val scores (`DataDisciplineError`) | Reflection is train-only; selection is aggregate-only; test belongs to `finalize()` | Reflect on train; read val as aggregates; finalize at the end |
| Second `finalize()` (`FinalizedError`) | Test is evaluated exactly once | Read `final_report.json`; start a fresh workspace to keep improving |

## Driving the loop yourself

When you (rather than a launched optimizer agent) create candidates, score
them, and finalize — via `prg = ap.attach("<workspace>")` — read
[references/prg-api.md](references/prg-api.md) first. It documents every
`prg` method signature, the candidate PEP 723 file conventions, and the
reflect-on-train / select-on-val / finalize loop.
