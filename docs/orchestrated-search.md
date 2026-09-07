# Orchestrated portfolio search

Status: initial implementation, July 2026.

## Decision

The Pi model already speaking with the human is the strategy orchestrator. It
does not write candidate implementations, and a live session never spawns a
second strategy process. A trusted Python controller dispatches independent Pi
workers, one implementation avenue per isolated context, evaluates their
solution bundles, and returns aggregate vectors plus sanitized implementation,
audit, setup, and failure evidence to that same host session.

This replaces prompt-only encouragement to "try diverse approaches" with a
controller-enforced breadth policy.

## Roles and trust boundaries

| Role | Sees | Does not see |
|---|---|---|
| User | resource proposal, metric demonstrations, final report | hidden rows |
| Python controller | all run state, expected outputs, metrics, budget | n/a |
| Human-facing Pi host orchestrator | conversation, schema, confirmed resources, web research, portfolio state, aggregate vectors and implementation diagnostics | hidden rows; candidate-authoring tools |
| Pi mechanism auditor | one avenue contract and its proposed source/dependency metadata | examples, metrics, scores, other candidates, val/test |
| Pi implementation worker | generic task brief, development examples, its own files and assigned mechanism | optimizer identity, metrics, scores, other workers, parent workspace, val/test |
| Candidate runtime | one input at a time, schema/runtime artifacts | expected val/test outputs |

Resource-confirmed workspaces store val and test outside the coding-agent
workspace. Pi workers additionally load only a cooperative root-guard extension.
That guard prevents accidental traversal but is **not** an OS security boundary.
`strict_isolation=True` therefore refuses until an OS sandbox adapter is
configured; the project must never describe path filtering as adversarial
sandboxing.

## Resources

`Resources` separates three contracts:

1. `SearchResources`: build hardware, Pi models, parallelism, package/model
   download permission, fine-tuning access.
2. `RuntimeResources`: the CPU/GPU/network/API/cost/latency envelope of the
   shipped package.
3. `DataPolicy`: whether task data may leave the machine and to which domains.

Hardware may be detected. Egress, network, downloads, package installation, and
remote compute are never inferred as consent. If a user supplies
`RemoteCompute`, they explicitly select its transport (the built-in adapter is
SSH); lightweight orchestration/bookkeeping stays local while heavy worker
commands, setup/training/model loads, and candidate evaluation are staged on the
target. Pi-runtime candidates remain beside the authenticated host (they are
network-bound and OAuth is never copied remotely). GPU-heavy avenues hold
per-target leases (one by default), use a configurable free-VRAM floor (80% when
total VRAM is supplied and no floor is chosen), and queue
rather than contend. OOM caused by contention is retried exclusively and is not
approach evidence.

The active Pi provider/model/thinking tuple may be detected as a capability. Pi
resolves stored OAuth/subscription authentication itself; raw API-key variables
are not required. Pi-backed runtime candidates remain in the measured portfolio
and are labelled as requiring Pi plus an authenticated subscription, allowing
the user to decide whether that deployment restriction is acceptable. Profiles
contain model/provider names and capability limits, never credentials.

A dollar-limited run serializes Pi calls when no per-call agent bound is known.
If `SearchResources.max_dollars_per_agent_call` is confirmed, the controller
reserves that much headroom for each in-flight call, settles actual reported
usage on completion, and submits additional calls only when committed headroom
allows it. This prevents several ordinary parallel calls from independently
consuming the same remaining dollars. A provider call that exceeds its declared
bound can still overshoot by that single call, after which no new call launches.

## Approach ladder and portfolio gates

The controller tracks eight tiers:

1. runtime generalist/coding agent
2. graph of model calls
3. single model call
4. fine-tuned model
5. specialized pretrained/deep model
6. classical machine learning
7. algorithms, features, and rules
8. cross-tier composition

Every feasible tier must produce faithful evaluated evidence or carry a
human-confirmed infeasibility reason. Each avenue is a hard mechanism experiment:
workers may not replace a blocked API/deep/classical/rules mechanism with another
family merely to return a valid answer. The implementation plan within that
boundary is flexible—dependency/model variants, setup, batching, preprocessing,
parsing, and device placement should adapt when the first plan hits a wall.
Dependencies are resolved from PEP 723 rather than inferred from the shell.

Before import, deterministic checks and an independent Pi mechanism audit reject
cross-tier fallbacks. Worker crashes, missing/malformed files, noncompliance,
all-run errors, and suspicious zero outputs are recorded as implementation
evidence and trigger bounded repair plus a fresh independent configuration.
`FAILED` and `NONCOMPLIANT` do not satisfy breadth. After those attempts,
ambiguous implementation/environment failures become human blockers rather than
family exclusions. Default budget allocation remains 40% breadth, 40% deepening,
and 20% composition/wildcards.

Workers use persistent Pi session IDs per avenue, but their only durable task
context is their isolated directory and their own prior implementation.

## Metric roles

A metric suite classifies every quality lens as one of:

- **acceptance**: user-approved definition of task success; final selection may
  use it;
- **diagnostic**: orchestrator-selected search lens; useful for plateaus and
  blind spots but unable to silently choose the winner;
- **operational**: harness-owned cost and latency objectives.

There is no required weighted scalar for suite-aware search. Acceptance floors
filter invalid operating points; Pareto dominance preserves quality/cost
tradeoffs; a lexicographic preference policy is committed before test is opened.
The legacy `primary` field remains only as a compatibility headline for reports
and old workspaces.

## Data flow

The public train split is deterministically partitioned for Pi work:

- `dev_fit`: written to an avenue's generic `examples.jsonl`;
- controller probe: withheld from implementation workers;
- `val`: controller-only selection;
- `test`: controller-only, once at finalization.

Resource-confirmed runs place val/test under the controller-private data root
(`$AP_PRIVATE_DATA_DIR` or `~/.cache/autoprogramming/private`). They are not
written under the agent workspace.

## Pi integration

The implementation is split by responsibility: `pi_rpc.py` owns JSONL/RPC
framing and usage collection, `pi_worker.py` owns implementation task bundles,
environment scrubbing, run-owned UV cache placement, worker process launch, and
scratch cleanup, while `pi_backend.py` is the trusted portfolio controller.
Worker state remains under `$AP_WORKER_DIR` (default `~/.cache/ap-work`) for
repair, diagnostics, and downstream evaluation. Finalization removes only owned
UV caches and disposable `.venv*` directories. Remote transfer excludes those
package environments, and remote evaluation uses a workspace-owned UV cache so
finalization can remove it without deleting staged source, artifacts, outputs,
or inactive-candidate diagnostics.

Inside a live Pi conversation, Python does not launch strategy RPC at all. The
host uses `prg.web_search`, `plan_portfolio`, `orchestrate_portfolio`, and
`portfolio_status` across turns. Python uses Pi's documented process APIs for:

- independent source-vs-mechanism and suspicious-implementation reviews;
- parallel `--mode json --print` implementation workers with skills, context,
  prompts, themes, and user extensions disabled; workers default to the exact
  host Pi model/thinking level or use a host-assigned discovered registry model,
  always through Pi's stored OAuth login;
- an explicitly loaded root-guard extension;
- per-avenue session IDs and session directories;
- assistant usage/cost collected from Pi message events and charged to the same
  dollar ledger as candidate evaluation.

The Python controller validates host-authored JSON plans, fills missing feasible
tiers, schedules remote/GPU work, serializes candidate import/evaluation,
propagates abort/failure state, and leaves finalization to the host after it has
inspected evidence. An explicitly headless `PiOrchestratorBackend` remains
available and is web-enabled before planning. Terminal provider errors are
rejected even when Pi itself exits with status zero.

### Live integration validation

On 2026-07-12 the complete staged path was exercised against Pi 0.80.6 and a
real `openai-codex/gpt-5.4-mini:low` model: metric proposal and independent
critique, user-side suite approval, portfolio planning, four persistent-session
workers behind the root guard, private val/test evaluation, budget attribution,
finalization, activation, and a production call all completed. The smoke run
used 30 candidate evaluations and $0.0425 of reported agent usage. This is a
compatibility smoke test, not a security claim; strict OS isolation remains the
limitation below.

## Score integrity

Every new split score records the candidate source SHA that produced it.
Comparison, tradeoff calculation, and finalization reject stale source. The
schema is also pinned at workspace creation. Unknown candidate cost is represented
as unknown/conservatively infinite for Pareto selection, never as zero dollars.
Candidate import is journaled in portfolio state before source/artifact creation;
on resume the controller finishes an already-scored or partially evaluated
candidate rather than dispatching a duplicate. Orphan artifact namespaces from
a crash before candidate creation are replaced only when no candidate file can
reference them.

## Human confirmation for blocked approaches

Only after implementation diagnosis, bounded same-family repair, an independent
configuration, and resource verification does an unresolved failure become
`blocked`; it never satisfies breadth. The controller shows the evidence instead
of accepting fallback code or silently excluding the family. A human may
fix/provision it and choose `retry`, or explicitly confirm `exclude`, via
`prg.resolve_blocker(...)`. The decision and approver are persisted. A retry
bypasses one stale preflight snapshot so newly provisioned access can be tested.

## User flow

```python
resources = ap.Resources(
    search=ap.SearchResources(
        max_parallel_agents=4,
        # Active Pi subscription model is captured from the host session.
        candidate_api_providers=(),
        # only if supplied: ap.RemoteCompute(endpoint=..., transport="ssh", ...),
        # with transport explicitly confirmed rather than assumed,
        allow_package_installs=True,
        allow_model_downloads=True,
    ),
    runtime=ap.RuntimeResources(network=False),
    data=ap.DataPolicy(external_egress=True),
    confirmed=True,
)

prepared = translate.prepare(
    pairs,
    resources=resources,
    budget=ap.Budget(dollars=2),
)
prg = ap.attach(prepared.workspace.root)
# Current host proposes/demonstrates/approves metrics with the user.
prg.web_search("latest efficient approaches for <abstract task>")
prg.web_search("2026 open source <task> benchmark")
prg.plan_portfolio(web_informed_avenue_specs)
prg.orchestrate_portfolio("breadth", budget=ap.Budget(dollars=20))
state = prg.portfolio_status()
prg.orchestrate_portfolio("deepen", avenue_ids=[...])
prg.orchestrate_portfolio("compose")
report = prg.finalize()
```

`prepare()` fixes the data split. In a live Pi session the current host proposes
the multi-lens suite and keeps all strategy context; no second orchestrator is
created. Search resumes only after sign-off and recorded web research.

## Remaining hardening

The controller-private directory and Pi root guard prevent ordinary accidental
leakage. Adversarial same-user isolation still requires an OS-level worker and
candidate sandbox, so strict isolation currently refuses. Candidate evaluation
already reuses one process per candidate/split: lazy clients and models persist,
`latency_s` is warm repeated-call latency when warm samples exist, and
`cold_start_s` is reported separately.
