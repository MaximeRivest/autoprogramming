---
name: candidate-optimizer
description: Optimize candidate implementations inside an autoprogramming workspace. Use when the working directory (or a nearby one, typically named like translate_ap or another *_ap package) contains active.json, schema.py, candidates/, and scores.json, or when asked to optimize, improve, evaluate, or add candidates for an autoprogramming program. Covers attaching via ap.attach, surveying the approach ladder and seeding a diverse portfolio across cost tiers, composing pretrained models as pipeline stages, acceptance/diagnostic metric-suite sign-off, the quality/cost tradeoff frontier, the train/val/test data discipline, PEP 723 candidate file conventions, and honest budget accounting.
---

# Candidate optimizer for autoprogramming workspaces

## Role boundary

The Pi session speaking with the human is the **only strategy orchestrator**:
it plans and allocates avenues but never implements candidates itself. Never
launch another `PiOrchestratorBackend` from this session; that creates a
context-poor duplicate. The trusted Python controller launches only
implementation workers and independent auditors with generic task briefs and
isolated directories. Do not send workers `prg`, optimizer terminology, metric
code/names/weights, scores, other candidates, val, or test. They know only their
function contract, dev-fit examples, assigned mechanism, resource envelope, and
their own prior files.

Before planning, call `prg.web_search(...)` at least twice with task-specific,
non-private queries and inspect current sources. Submit the resulting
`ap.AvenueSpec` list through `prg.plan_portfolio(...)`, dispatch with
`prg.orchestrate_portfolio("breadth")`, inspect `prg.portfolio_status()`, then
choose deepen/compose phases in this same conversation.

Do not finalize merely because one avenue is acceptable. Controller policy
requires every resource-feasible tier to be attempted or explicitly excluded,
a second configuration per successful family, then a score-informed deepening
and cross-tier composition round. Metric suites separate user-approved
acceptance lenses from search-only diagnostics; approach novelty is a portfolio
gate, not a quality metric.

An avenue is a **hard mechanism experiment**, not a request to solve the task by
any means. It may fail, block, or score badly, but it must never switch families
to avoid an error. However, implementation plans are flexible: change package
versions, model variants within the family, batching, preprocessing, parsing,
device placement, and setup until the assigned mechanism gets a fair test.
Worker crashes, malformed output, noncompliance, and suspicious zero results do
not satisfy breadth. They trigger diagnosis, same-family repair, and a fresh
independent configuration; ambiguity goes to the human.

Authenticated Pi model access does not require a raw SDK key. Pi workers default
to the host provider/model/thinking level; an avenue may select another exact
pattern from the discovered `pi_models` registry. Pi resolves OAuth from its auth store.
A Pi-backed candidate should call persistent Pi CLI/RPC and is labelled as
requiring Pi plus that login at deployment. Missing Torch/GPU still does not
permit classical CV. Dependencies belong in PEP 723. Only explicit composition
avenues may route across families.

You are optimizing one typed program by writing, evaluating, and evolving
complete candidate implementations — plain Python files — under a strict data
discipline and an explicit budget. Your job is not to polish one idea; it is to
survey a spectrum of approaches, baseline a diverse portfolio, then deepen the
ones the data rewards — watching quality AND cost.

## Find the workspace and attach

The workspace root is the directory containing `active.json`, `schema.py`, and
`candidates/` — usually your current working directory. (When this skill has
been copied into a workspace, this file itself sits below that root, under
`.agents/skills/candidate-optimizer/` or `.claude/skills/candidate-optimizer/`;
if the cwd is somewhere else, walk up from this file to find the root.) The
workspace is itself a valid, installable Python package. Work from the root
and attach:

    import autoprogramming as ap
    prg = ap.attach("<workspace root>")

`prg` is your handle. Everything else is file operations on `candidates/*.py`.

Read per-workspace facts live from the workspace, not from memory:

- `prg.schema` — the program's inputs, outputs, and docstrings. schema.py is
  immutable: you satisfy it, you never edit it.
- `prg.data` — `prg.data.train` rows are readable; `prg.data.val` supports
  len() only.
- `prg.budget` — remaining dollars / eval_calls / minutes.
- `data/split.json` — split sizes and the bootstrap flag. `"bootstrap": true`
  means fewer than ~30 examples: build and compare baseline candidates only.
  Fine-grained mutation loops are refused because score differences at that
  data size are one row of noise, and at most 5 distinct candidates may be
  scored on val. Offer to generate synthetic examples for the user to
  validate.

## The prg API

    prg.schema                                   # inputs/outputs & docstrings
    prg.data.train                               # readable rows: iterate, sample, inspect
    prg.data.val                                 # len() only — val rows are never readable
    prg.budget                                   # remaining dollars / eval_calls / minutes
    prg.propose_metric(code, examples, primary=..)  # metric sign-off — FIRST, before any eval
    prg.web_search(query)                        # current sources; run 2+ before planning
    prg.plan_portfolio(specs)                    # plan authored by THIS Pi session
    prg.orchestrate_portfolio("breadth", budget=ap.Budget(dollars=20))
                                                   # trusted controller + isolated workers
    prg.portfolio_status()                       # source/audits/failures/objectives
    prg.orchestrate_portfolio("deepen", avenue_ids=[...])
    prg.orchestrate_portfolio("compose")
    prg.new_candidate(source=...)                # manual/legacy path
    prg.eval("candidate_1")                      # val: every objective scored from one run
    prg.eval("candidate_1", split="train", per_instance=True)   # per-row — train only
    prg.run("candidate_1", split="train", row=17)               # full trace — train only
    prg.compare("candidate_0", "candidate_1")    # paired bootstrap diff on val (primary)
    prg.compare("candidate_0", "candidate_1", objective="cost_dollars")  # any objective
    prg.tradeoffs()                              # the quality/cost Pareto frontier
    prg.frontier()                               # Pareto frontier over train rows
    prg.resolve_blocker(id, "retry", confirmed_by="user")  # after human check
    prg.finalize()                               # suite policy chooses finalists — LAST

## Survey the approach ladder before you commit

A candidate is any Python that satisfies the schema, so the search space spans a
whole cost/capability spectrum. From most expensive/capable to cheapest:

1. Generalist harness — a coding agent or reasoning model doing the task live.
2. A graph of several model calls (plan, act, critique, retrieve).
3. A single model call with an optimized prompt.
4. A finetuned small model.
5. A specialized deep net (a task-specific pretrained model).
6. Classical ML (a fitted scikit-learn head, gradient boosting, nearest neighbor).
7. Hand-written features, rules, regex, or lookup logic.

Rules to follow, not options:

- **Breadth before depth.** Do NOT seed one candidate and mutate it forever —
  that single-step trap wastes budget refining a local idea while the winning
  family goes untried. First seed a PORTFOLIO of genuinely distinct candidates
  spanning several tiers (e.g. a model call, a classical head, a rules baseline,
  a pretrained-model pipeline) and baseline all of them. THEN deepen the ones the
  data rewards.
- **Compose across tiers.** The best solution is usually a compound system, not a
  single tier. Use a heavy pretrained model as ONE STAGE, not only end to end.
  Patterns worth naming and trying: pipeline/decomposition (segment, then
  measure; extract, then classify); cascade (a cheap model handles the easy bulk,
  an expensive one handles the hard tail); ensemble/vote; router (pick a path per
  input); learned-feature + classical-head (embed with a pretrained model, fit a
  small classifier on top).
- **Do not dismiss a family on one try.** A tier or model family is ruled out only
  after a fair attempt and at least two materially independent faithful configs.
  A worker/implementation failure is not a mechanism result and cannot satisfy
  breadth. Inspect `portfolio_status()` evidence, repair creatively within the
  family, and use a fresh worker context for the second config. If the family
  still appears blocked by credentials, GPU, packages, downloads, or network,
  pause and check with the human. Retry
  with `prg.resolve_blocker(id, "retry", confirmed_by="user")`; exclude only
  after the human explicitly confirms the capability is unavailable.
- **Pick the cheapest tier that can plausibly clear the bar**, then climb only when
  the data shows that tier's ceiling. Cost is an objective, not an afterthought
  (see below): a rules candidate at 0.95 beats a model call at 0.90 when the CI
  separates and it never memorized.

## Use current tools, not remembered ones

Do NOT trust training memory for "the latest" model, checkpoint, or library — it
is stale by construction. `plan_portfolio()` enforces a current-source gate:
run at least two `prg.web_search(...)` queries, inspect/cite the results, then
plan. Before you fetch a model or pick a package, check what is current and prefer
the current best-for-cost. The classic failure is reaching for a version you
remember (an old SAM when a newer, smaller, faster SAM exists; a superseded
checkpoint) instead of the one that ships today. Verify a model actually exists
and LOADS before you build a candidate around it — a candidate wired to a model
that no longer resolves is a wasted iteration.

## Step 0: metric suite sign-off before ANY scoring

A wrong metric produces a confidently wrong program. Propose several independent
lenses from one shared run, then classify each as **acceptance** (user-approved,
eligible for final selection) or **diagnostic** (search guidance only). Never
collapse them into one convenient scalar merely because it is easy to optimize.

1. Write `metric.py` with `METRICS = {"<name>": fn, ...}` (or the legacy single
   `metric`). Names must not collide with `cost_dollars` / `latency_s`.
2. Have an independent metric critic attack proxy hacking, flat exact-match,
   format blindness, semantic blindness, robustness gaps, and evaluator
   self-preference.
3. Demonstrate every lens on 3+ real exact/near-miss/failure pairs. Iterate with
   the user until judgments match.
4. Construct `ap.MetricSuite(acceptance=(...), diagnostic=(...),
   policy=ap.SelectionPolicy(floors=..., preference_order=...))` and call
   `prg.approve_metric_suite("<approver>", suite, weights=...)` only after the
   user approves. AP_AUTO_APPROVE_METRIC is for deliberately unattended runs,
   never a way to impersonate a reachable user.

Acceptance roles, floors, and preference order are precommitted before val and
cannot change after selection starts. Diagnostic lenses may evolve; editing
metric code re-scores unchanged candidate bundles from cached outputs. A flat
strict metric should become a diagnostic beside a graded lens, not force every
implementation worker to game exact match. Implementation workers see neither
kind of metric.

## Cost is an objective

`latency_s` is measured automatically. `cost_dollars` comes from
`AP_COST_DOLLARS` or declared `cost_per_call`; an omitted report is unknown and
never treated as free (lower is better) — no metric.py involvement. Every
`prg.eval` reports these alongside quality metrics. The goal is the best quality
per known cost, not perfection at any price.

- During the loop, read `prg.tradeoffs()` to see the quality/cost Pareto frontier
  over evaluated candidates — the set where no other candidate beats it on every
  objective. Candidates off the frontier are strictly dominated; drop them.
- Use `prg.compare(a, b, objective="cost_dollars")` (or `latency_s`) to test a
  cost/latency difference the same paired-bootstrap way you test quality. It is
  direction-aware: for these minimized objectives "improved" means the challenger
  is reliably CHEAPER or FASTER (lower is better), not higher.
- Present the USER the frontier, not just the single most-accurate candidate:
  "here is the cheap-good one, the mid one, and the expensive-best one." Let them
  choose the point on the curve. `prg.finalize()` activates a sensible default
  (the point chosen by the precommitted acceptance preference) and reports the
  remaining frontier alternatives.

## Data discipline (harness-enforced; violations raise, they don't warn)

- Reflect on TRAIN failures only. prg.run() and per-instance scores are refused
  on val and test. You never learn why a val row scored low — only the aggregate —
  so you cannot edit candidates to fix specific val rows.
- VAL is for selection only, and selection pressure is capped. Every eval scores
  the identical val set; the harness counts how many distinct candidates val has
  judged, and past its capacity val scores lose meaning and are not reported as
  final. Prefer few, genuinely different candidates over many small tweaks.
- TEST is off-limits to you entirely. prg.finalize() — the harness, not you —
  evaluates it once, at the end, on the top candidates, then activates the winner.
- "Improved" means the paired-bootstrap CI excludes zero, not that the point
  estimate went up. Use prg.compare() before believing a win.
- Memorizers are flagged and excluded from selection: train score far above val,
  or verbatim training outputs pasted into candidate source. A lookup table over
  train inputs cannot win; a candidate only wins on data it never saw.

## Optional remote search compute

Remote compute and its transport are never assumed or hard-coded. The user must
explicitly choose the available adapter (currently `transport="ssh"`). Only when
the user's confirmed `Resources` contains `remote_compute` does the controller stage worker tools,
installs, training, model loads, and evaluation there. Lightweight orchestration
and bookkeeping remain local. Pi-runtime candidates also remain beside the
authenticated host; OAuth is never copied to the target. GPU-heavy avenues hold
target-specific leases (default one at a time), wait for free VRAM, and retry contention/OOM
exclusively; they must not fall back to local compute or count contention as an
approach failure. If remote access breaks, consult the user.

## Candidate conventions: PEP 723 single-file scripts in candidates/

- Each candidate is candidates/candidate_<n>.py defining `predict(<input names>)`
  that returns the output value (or a tuple in schema order for multi-output
  programs). Create files with prg.new_candidate; edit them like any Python file.
- Dependencies go in a PEP 723 block at the top of the file:

      # /// script
      # requires-python = ">=3.11"
      # dependencies = ["openai>=1.0"]
      # ///

  stdlib-only candidates need no block at all. Candidates with conflicting
  dependencies can coexist — each runs in its own environment. The harness uses
  stable dependency-keyed drivers so UV reuses those environments. Do not create
  a `.venv` per candidate or experiment. For a direct package test, use the
  candidate's stable script path
  with `uv run candidates/candidate_<n>.py`; remove any throwaway environment
  before finishing.
- No work at import time: clients and models load lazily inside predict(), so
  importing the package never needs an API key or network access.
- **Never add a cross-family safety fallback.** Provider/model retries and robust
  parsing are valid. If task.md lists an authenticated Pi model, use Pi CLI/RPC
  instead of checking for an `*_API_KEY`; `if no key: use sklearn`,
  `except ImportError: use cv2`, or
  `if no GPU: use rules` are invalid. Raise a precise setup error instead. A
  suspicious zero result triggers implementation diagnosis and an independent
  configuration; it does not by itself establish that the mechanism is blocked.
  A high-scoring substitute corrupts the experiment.
- A candidate that invokes Pi CLI/RPC must declare `[tool.ap]` with
  `pi_runtime = true`, so evaluation retains the host OAuth boundary.
- A candidate that makes no stochastic calls should declare `[tool.ap]` with
  `deterministic = true` in its block — it is then scored with 1 repeat instead of
  3, which saves budget.
- A candidate that spends money per call should set the module global
  AP_COST_DOLLARS after each predict() so the ledger and the cost objective stay
  honest (or declare a flat `cost_per_call` in `[tool.ap]`).
- Model weights, pickles, and lookup tables belong in artifacts/, resolved via
  `from <package>.paths import artifacts`. Pretrained-model candidates may declare
  a `fetch = ["huggingface:..."]` download step in `[tool.ap]`.
- Explore genuinely different approaches AND compositions of them — the metric
  does not care how a candidate works, only that it wins honestly.

## The loop

0. **Metric suite sign-off** — acceptance vs diagnostic roles, floors, and
   preference committed before any eval. Add a graded diagnostic if strict is flat.
1. **Survey the ladder, seed a diverse portfolio.** Check current tools, then
   create several genuinely distinct candidates across tiers (a model call, a
   classical head, a rules baseline, a pretrained-model-as-a-stage pipeline) —
   breadth before depth.
2. **Baseline the portfolio on val**, then read `prg.tradeoffs()` to see where
   each lands on the quality/cost frontier. Drop dominated dead ends; record why.
3. **Deepen the promising tiers.** Compose (cascade, ensemble, router,
   learned-feature + classical-head), tune, and try a second config before
   dismissing a family. Reflect on train: `prg.eval(name, split="train",
   per_instance=True)` to find the worst rows, `prg.run(name, split="train",
   row=i)` to read their traces, `prg.frontier()` to see which candidate wins
   which rows. Keep a mutation only when `prg.compare(best, new)` says the CI
   excludes zero.
4. **Watch the frontier, not one number.** Check `prg.budget` between iterations;
   stop when the quality/cost frontier stops advancing (a new candidate no longer
   joins or displaces the frontier), not merely when a single metric plateaus.
5. **prg.finalize()** — one-time test eval, activation under the precommitted
   frontier preference, sealed report. Present the tradeoff frontier to the user and
   the one-liner to switch to a cheaper/faster alternative. Do not skip finalize:
   an un-finalized workspace ships nothing. Orchestrated runs clean only their
   owned UV caches and disposable package environments at this point; candidate
   diagnostics remain available, and paused runs retain package caches for resume.

## Budget honesty

Every evaluation call is charged against the budget — model-calling candidates
cost money to SCORE, not just to write. Re-scoring from cached outputs (after a
metric edit) is free and never charges the budget; only actual runs do. Check
prg.budget between iterations and stop cleanly before it runs dry; there is no
free retry once the ledger hits zero.
