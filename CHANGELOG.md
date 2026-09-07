# Changelog

## Unreleased

- Bounded PEP 723 evaluation environments with stable dependency-keyed drivers,
  moved worker and remote-evaluation UV data into run-owned caches, and added
  guarded cache-only cleanup that preserves all candidate diagnostics.
- Made the human-facing Pi session the sole strategy orchestrator; live runs use
  staged host APIs instead of spawning a second context-poor strategy process.
- Added mandatory, persisted web research before host/headless portfolio plans.
- Added active/default plus same-provider Pi registry model assignment for
  workers, a mandatory measured subscription-runtime avenue when feasible, and explicit
  deployment labels without raw SDK API keys.
- Added optional user-provided remote search compute with explicit transport,
  selective heavy-work placement, remote candidate sessions, and per-target
  GPU device/concurrency/VRAM admission with exclusive retries.
- Added implementation-failure diagnosis and repair loops, independent second
  configurations, suspicious-output review, and evidence-rich human blockers;
  broken/noncompliant attempts no longer satisfy portfolio breadth.
- Made portfolio avenues hard mechanism contracts: implementation workers may
  not replace a blocked API, deep-model, classical, or rules approach with
  another family merely to avoid an error.
- Added deterministic and independent Pi mechanism-adherence audits before
  candidate import, with bounded in-session repair and clean-session restart.
- Added search-time candidate API access to the confirmed resource contract.
- Added environment blocker state and human-confirmed retry/exclusion; blocked
  approaches pause breadth instead of silently falling back or being discarded.

## 0.2.0 — 2026-07-12

- Added a resource-aware Pi portfolio backend with one strategy orchestrator and
  parallel, implementation-only avenue workers.
- Added explicit search, deployment, and data-egress resource contracts.
- Added acceptance/diagnostic metric suites, acceptance floors, and
  precommitted Pareto selection policies.
- Added breadth, second-configuration, stagnation, deepening, and composition
  policy state.
- Added the staged `Program.prepare()` / `PreparedRun` workflow.
- Added controller-private val/test storage for resource-confirmed runs and a
  deterministic dev-fit/controller-probe partition.
- Pinned schemas, split scores, and declared artifact bundles by content hash.
- Unknown candidate cost is no longer represented as free.
- Added Pi RPC/JSON usage accounting, worker context isolation flags, persistent
  avenue sessions, and a cooperative root-guard extension.
- Added in-flight agent dollar reservations; dollar-limited Pi work serializes
  when no confirmed per-call bound is available.
- Added pre-import candidate journaling and pending-evaluation recovery to avoid
  duplicate candidates across controller crashes.
- Added persistent candidate processes so lazy clients/models survive across
  rows, with warm latency and cold start reported separately.
- Split Pi RPC, worker isolation/bundling, and trusted portfolio control into
  focused modules.
- Hardened Pi integration against zero-exit provider failures and stale metric
  role names after critic rewrites; exposed separate worker/orchestrator timeouts.
- Validated the complete staged workflow against Pi 0.80.6 with real orchestrator,
  critic, and implementation-worker model calls.
- Added CI, an MIT license, architecture documentation, and adversarial tests.

### Known limitation

The worker root guard and controller-private files prevent ordinary accidental
leakage but are not an adversarial same-user OS sandbox. Strict isolation
currently refuses rather than silently weakening that guarantee.

## 0.1.0 — 2026-07-12

- Initial complete library API, Agent Skills, guarded train/val/test workflow,
  portable workspaces, metric approval, output caching, and multi-objective
  quality/cost/latency reporting.
