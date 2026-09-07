"""Honest numbers: seeded bootstrap statistics and the evaluate/compare loop.

Every aggregate carries a bootstrap confidence interval, stochastic
candidates are scored across repeats with the variance reported, and
comparisons are paired per row — "improved" means the interval excludes
zero, not that the point estimate went up.

Scoring is multi-objective: a single shared run per (row, repeat) is scored
under every quality metric and its cost/latency read off the same RunResult,
so extra metrics cost ~no extra model calls. The primary quality metric is
mirrored to the top level of every record for back-compat; the full objective
vector lives alongside under ``["objectives"]``.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import math
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from statistics import fmean, pstdev, pvariance
from typing import TYPE_CHECKING, Sequence

from . import metric
from .budget import BudgetLedger
from .errors import DataDisciplineError, MemorizationWarning

if TYPE_CHECKING:
    from .workspace import Workspace

DEFAULT_REPEATS = 3
UNKNOWN_COST = float.fromhex("0x1.fffffffffffffp+1023")

#: Cost objectives are properties of the run (lower is better), captured for
#: free on every candidate and independent of metric.py.
COST_OBJECTIVES: dict[str, str] = {"cost_dollars": "min", "latency_s": "min"}


def _sibling(name: str):
    """Import a sibling module at call time (unit tests stub it in sys.modules)."""
    return importlib.import_module(f"autoprogramming.{name}")


@dataclass
class EvalReport:
    """One candidate's evaluation on one split, with uncertainty attached.

    The top-level fields (``mean/std/ci95/per_field`` …) are the PRIMARY quality
    metric's, exactly as before. ``objectives`` adds every objective — each
    quality metric plus ``cost_dollars`` and ``latency_s`` — as a small dict
    with ``mean/std/ci95/per_field``. ``per_row`` is populated only for
    ``split="train"`` with ``per_instance=True``.
    """

    candidate: str
    split: str
    mean: float
    std: float
    ci95: tuple[float, float]
    n_rows: int
    n_repeats: int
    repeat_variance: float
    per_row: dict[str, float] | None
    per_field: dict[str, float] | None
    errors: list[str]
    flags: list[str]
    objectives: dict[str, dict] = field(default_factory=dict)
    primary: str | None = None
    cold_start_s: float | None = None

    def __str__(self) -> str:
        headline = (
            f"{self.candidate} on {self.split}: {self.mean:.3f} ± {self.std:.3f} "
            f"(n={self.n_repeats} repeats), 95% CI "
            f"[{self.ci95[0]:.3f}, {self.ci95[1]:.3f}]"
        )
        if self.objectives and self.primary:
            headline += f"  [primary: {self.primary}]"
        lines = [headline]
        lines.extend(self.flags)
        if self.cold_start_s is not None:
            lines.append(f"  cold_start_s: {self.cold_start_s:.3f}s  (reported separately)")
        for name, obj in self.objectives.items():
            if name == self.primary:
                continue
            goal = "min" if name in COST_OBJECTIVES else "max"
            ci = obj.get("ci95") or [obj["mean"], obj["mean"]]
            if name == "cost_dollars" and float(obj["mean"]) >= UNKNOWN_COST:
                lines.append(
                    "  cost_dollars: unknown  (min; declare cost_per_call or "
                    "AP_COST_DOLLARS)"
                )
            else:
                lines.append(
                    f"  {name}: {_format_objective(name, obj['mean'])}  ({goal})  "
                    f"95% CI [{float(ci[0]):.4g}, {float(ci[1]):.4g}]"
                )
        return "\n".join(lines)


@dataclass
class CompareReport:
    """A paired comparison of two candidates on stored per-row scores.

    ``a`` is the baseline, ``b`` the challenger; ``diff_mean`` is the mean of
    ``b - a`` per row. ``improved`` is True only when the 95% CI of that
    difference lies entirely above zero — the CI excludes zero in the
    challenger's favor. ``objective`` names which objective was compared
    (None = the primary quality metric, the back-compat default).
    """

    a: str
    b: str
    split: str
    diff_mean: float
    ci95: tuple[float, float]
    improved: bool
    objective: str | None = None
    goal: str = "max"

    def __str__(self) -> str:
        if self.improved:
            direction = "lower is better" if self.goal == "min" else "higher is better"
            verdict = f"improved — the CI excludes 0 in the challenger's favor ({direction})"
        else:
            verdict = "not improved — the CI does not exclude 0; the difference may be noise"
        obj = f" [{self.objective}]" if self.objective else ""
        return (
            f"{self.b} vs {self.a} on {self.split}{obj}: diff {self.diff_mean:+.3f}, "
            f"95% CI [{self.ci95[0]:.3f}, {self.ci95[1]:.3f}] — {verdict}"
        )


@dataclass
class TradeoffReport:
    """The quality/cost frontier over stored candidate objective vectors."""

    rows: list[dict]
    nondominated: list[str]
    split: str

    def __str__(self) -> str:
        if not self.rows:
            return (
                f"no objective vectors stored for {self.split!r} yet — eval some "
                f"candidates, then read prg.tradeoffs() to see the quality/cost frontier"
            )
        obj_names = list(self.rows[0]["objectives"].keys())
        header = ["", "candidate", *obj_names]
        table = [header]
        for r in self.rows:
            mark = "*" if r["candidate"] in self.nondominated else " "
            table.append([
                mark, r["candidate"],
                *[_format_objective(n, r["objectives"][n]) for n in obj_names],
            ])
        widths = [max(len(row[i]) for row in table) for i in range(len(header))]
        rendered = [
            "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
            for row in table
        ]
        return "\n".join([
            f"quality / cost tradeoffs on {self.split}:",
            *rendered,
            "frontier (*) = no other candidate beats it on every objective",
        ])


def _format_objective(name: str, value: float) -> str:
    if name == "cost_dollars":
        return "unknown" if value >= UNKNOWN_COST else f"${value:.4g}"
    if name == "latency_s":
        return f"{value:.3f}s"
    return f"{value:.3f}"


def _percentile(sorted_vals: list[float], q: float) -> float:
    idx = q * (len(sorted_vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean. Pure and seeded.

    Degenerate inputs degenerate honestly: an empty list gives ``(0.0, 0.0)``,
    a single value or a constant list gives ``(v, v)``.
    """
    vals = [float(v) for v in values]
    if not vals:
        return (0.0, 0.0)
    if len(vals) == 1 or min(vals) == max(vals):
        return (vals[0], vals[0])
    rng = random.Random(seed)
    n = len(vals)
    means: list[float] = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += vals[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return (_percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2))


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap CI of the mean paired difference ``b - a``. Pure and seeded.

    Resampling the per-row differences is equivalent to resampling the pairs,
    which keeps row-level correlation intact.
    """
    if len(a) != len(b):
        raise ValueError(
            f"paired bootstrap needs equal-length paired samples, got "
            f"{len(a)} and {len(b)}."
        )
    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    return bootstrap_ci(diffs, n_boot=n_boot, seed=seed, alpha=alpha)


# ------------------------------------------------------------- objective goals


def _objective_directions(quality_names) -> dict[str, str]:
    """Goal per objective: quality metrics maximize, cost objectives minimize."""
    directions = {name: "max" for name in quality_names}
    directions.update(COST_OBJECTIVES)
    return directions


def _directions_from_points(points: dict[str, dict]) -> dict[str, str]:
    names: set[str] = set()
    for p in points.values():
        names.update(p)
    return {n: ("min" if n in COST_OBJECTIVES else "max") for n in names}


#: Cost objectives (notably latency_s) are wall-clock and jitter run to run.
#: Rounding them to this many significant figures for Pareto DOMINATION (never
#: for display) keeps sub-resolution noise from flipping frontier membership,
#: while real differences (a 0.01s heuristic vs a 2s model) still separate.
_COST_PARETO_SIGFIGS = 3


def _round_sig(x: float, sig: int = _COST_PARETO_SIGFIGS) -> float:
    if not math.isfinite(x) or x >= UNKNOWN_COST:
        return x  # unknown cost stays worst; it must never round into "free"
    if x == 0:
        return 0.0
    digits = -int(math.floor(math.log10(abs(x)))) + (sig - 1)
    return round(x, digits)


def _domination_points(points: dict[str, dict]) -> dict[str, dict]:
    """Copy of ``points`` with cost objectives rounded for stable domination."""
    return {
        name: {o: (_round_sig(v) if o in COST_OBJECTIVES else v) for o, v in vec.items()}
        for name, vec in points.items()
    }


def pareto_nondominated(
    points: dict[str, dict], goals: dict[str, str]
) -> list[str]:
    """Names whose objective vectors are Pareto-nondominated (pure, seed-free).

    A candidate is dominated when another is at-least-as-good on every objective
    (respecting each objective's goal — ``max`` for quality, ``min`` for cost)
    and strictly better on at least one. Ties on every objective leave both on
    the frontier. Order of the result follows ``points`` insertion order.
    """
    names = list(points)
    return [
        a
        for a in names
        if not any(
            b != a and _dominates(points[b], points[a], goals) for b in names
        )
    ]


def _dominates(p: dict, q: dict, goals: dict[str, str]) -> bool:
    strictly_better = False
    for obj, goal in goals.items():
        if obj not in p or obj not in q:
            continue
        pv, qv = p[obj], q[obj]
        if goal == "min":
            if pv > qv:
                return False
            if pv < qv:
                strictly_better = True
        else:
            if pv < qv:
                return False
            if pv > qv:
                strictly_better = True
    return strictly_better


# --------------------------------------------------------------- outputs cache


def cache_path(workspace, candidate_name: str, split: str) -> Path:
    """Path of a candidate/split's cached run outputs under ``.ap/outputs/``."""
    base = getattr(workspace, "outputs_dir", None)
    if base is None:
        base = Path(workspace.root) / ".ap" / "outputs"
    return Path(base) / f"{candidate_name}__{split}.json"


def _source_sha(workspace, candidate) -> str:
    """Pinned candidate bundle identity (source plus declared artifacts)."""
    candidates_mod = _sibling("candidates")
    bundle = getattr(candidates_mod, "bundle_sha", None)
    if bundle is not None:
        return bundle(workspace, candidate)
    return hashlib.sha256(candidate.source.encode("utf-8")).hexdigest()


def read_output_cache(workspace, candidate_name: str, split: str) -> dict | None:
    """The candidate/split output cache, or None when missing or stale.

    Stale means the candidate's source sha no longer matches what produced the
    cached outputs — its code changed, so the cached predictions cannot be
    trusted and the candidate must be re-run. Missing caches short-circuit
    before the candidate is even loaded, so this is cheap on a cold workspace.
    """
    path = cache_path(workspace, candidate_name, split)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    try:
        candidate = _sibling("candidates").load_candidate(workspace, candidate_name)
    except Exception:
        return None
    if data.get("source_sha") != _source_sha(workspace, candidate):
        return None
    return data


def write_output_cache(workspace, candidate, split: str, cache_rows: dict) -> None:
    """Persist a candidate/split's per-repeat outputs + cost + latency."""
    path = cache_path(workspace, candidate.name, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"source_sha": _source_sha(workspace, candidate), "rows": cache_rows}
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _run_cache_entry(run) -> dict:
    """One repeat's cacheable outcome: outputs (if ok) plus cost and latency."""
    ok = bool(getattr(run, "ok", False))
    return {
        "ok": ok,
        "outputs": getattr(run, "outputs", None) if ok else None,
        "cost_dollars": (
            None
            if getattr(run, "cost_dollars", None) is None
            else float(getattr(run, "cost_dollars"))
        ),
        "latency_s": float(getattr(run, "duration_s", 0.0) or 0.0),
        "cold_start": bool(getattr(run, "cold_start", False)),
    }


def _reps_in_cache(cache_rows: dict) -> int:
    for reps in cache_rows.values():
        return len(reps)
    return 1


# ------------------------------------------------------------------ aggregation


def _aggregate_stats(row_means: dict, per_field_lists=None) -> dict:
    """Mean/std/ci95 over per-row means (rows ordered by index for determinism)."""
    ordered = sorted(row_means, key=_row_index)
    means = [row_means[rid] for rid in ordered]
    if means and any((not math.isfinite(v)) or v >= UNKNOWN_COST for v in means):
        # A finite JSON-safe sentinel keeps strict parsers happy while remaining
        # conservative for a minimized Pareto objective.
        mean = UNKNOWN_COST
        std = 0.0
        lo, hi = (UNKNOWN_COST, UNKNOWN_COST)
    else:
        mean = fmean(means) if means else 0.0
        std = pstdev(means) if means else 0.0
        lo, hi = bootstrap_ci(means)
    per_field = (
        {name: fmean(vals) for name, vals in per_field_lists.items()}
        if per_field_lists
        else None
    )
    return {
        "rows": dict(row_means),
        "mean": mean,
        "std": std,
        "ci95": [lo, hi],
        "per_field": per_field,
    }


def _aggregate_from_cache(
    cache_rows: dict, metrics, primary, schema, weights, expected_by_row, n_repeats
) -> dict:
    """The full per-split record from cached outputs: primary top-level + objectives.

    Reproduces the legacy single-metric aggregate exactly for the primary
    metric (top level), and computes every quality metric plus cost/latency
    under ``["objectives"]`` from the same cached runs — no model calls.
    """
    ordered = sorted(cache_rows, key=_row_index)
    q_row_means = {name: {} for name in metrics}
    q_field_lists = {name: {} for name in metrics}
    cost_row_means: dict = {}
    latency_row_means: dict = {}
    cold_start_durations: list[float] = []
    cold_only_latency_rows: set[str] = set()
    primary_row_variances: list[float] = []

    for rid in ordered:
        reps = cache_rows[rid]
        expected = expected_by_row[rid]
        for name, fn in metrics.items():
            repeat_scores: list[float] = []
            for rep in reps:
                if rep.get("ok"):
                    score, per_field = metric.score_pair(
                        fn, schema, rep["outputs"], expected, weights
                    )
                    if per_field:
                        for fname, fscore in per_field.items():
                            q_field_lists[name].setdefault(fname, []).append(fscore)
                else:
                    score = 0.0
                    # A failed repeat scores 0.0 in the headline mean; the
                    # per-field breakdown must count it as 0.0 too (for
                    # multi-output programs), or per_field would silently
                    # average over successes only and overstate quality.
                    if len(schema.output_names) > 1:
                        for fname in schema.output_names:
                            q_field_lists[name].setdefault(fname, []).append(0.0)
                repeat_scores.append(score)
            q_row_means[name][rid] = fmean(repeat_scores) if repeat_scores else 0.0
            if name == primary:
                primary_row_variances.append(
                    pvariance(repeat_scores) if repeat_scores else 0.0
                )
        reported_costs = [r.get("cost_dollars") for r in reps]
        cost_row_means[rid] = (
            UNKNOWN_COST
            if any(value is None for value in reported_costs)
            else fmean([float(value) for value in reported_costs])
            if reported_costs
            else UNKNOWN_COST
        )
        cold = [
            float(r.get("latency_s") or 0.0)
            for r in reps if r.get("cold_start")
        ]
        cold_start_durations.extend(cold)
        warm = [
            float(r.get("latency_s") or 0.0)
            for r in reps if not r.get("cold_start")
        ]
        # A one-call evaluation has no warm sample; retain its end-to-end
        # duration rather than fabricating zero.
        latency_row_means[rid] = fmean(warm or [
            float(r.get("latency_s") or 0.0) for r in reps
        ]) if reps else 0.0
        if cold and not warm:
            cold_only_latency_rows.add(rid)

    # Once a warm sample exists, remove the one cold-only row from the warm
    # latency objective. Cold start remains separately reported.
    if set(latency_row_means) - cold_only_latency_rows:
        for rid in cold_only_latency_rows:
            latency_row_means.pop(rid, None)

    objectives: dict[str, dict] = {}
    for name in metrics:
        objectives[name] = _aggregate_stats(q_row_means[name], q_field_lists[name] or None)
    objectives["cost_dollars"] = _aggregate_stats(cost_row_means)
    objectives["latency_s"] = _aggregate_stats(latency_row_means)

    prim = objectives[primary]
    return {
        "rows": dict(prim["rows"]),
        "mean": prim["mean"],
        "std": prim["std"],
        "ci95": list(prim["ci95"]),
        "n_repeats": n_repeats,
        "repeat_variance": fmean(primary_row_variances) if primary_row_variances else 0.0,
        "per_field": prim["per_field"],
        "cold_start_s": (
            fmean(cold_start_durations) if cold_start_durations else None
        ),
        "objectives": objectives,
    }


def _objectives_report(sub: dict) -> dict[str, dict]:
    return {
        name: {k: obj[k] for k in ("mean", "std", "ci95", "per_field")}
        for name, obj in sub.get("objectives", {}).items()
    }


# ------------------------------------------------------------- scores.json I/O


def _approval_weights(workspace) -> dict | None:
    path = Path(workspace.metric_approval)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get("weights")
    except json.JSONDecodeError:
        return None


def _weights_fingerprint(weights) -> str:
    """Stable digest of the approval weights, "" when unweighted.

    Weights are config, not code — re-approving with new per-field weights
    changes the aggregate score without changing metric.py's sha, so scores
    recorded under the old weights must be reconciled (not silently trusted).
    """
    if not weights:
        return ""
    blob = json.dumps(weights, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _recompute_flags(workspace, scores: dict) -> None:
    """Refresh memorization flags for candidates that have both train and val.

    The flag depends on the primary train/val means, so it must be recomputed
    whenever those means change (metric edit, primary switch, re-weighting).
    """
    data_mod = _sibling("data")
    guards = _sibling("guards")
    candidates_mod = _sibling("candidates")
    schema = workspace.schema
    train_rows = None
    for cand_name, splits in scores.get("candidates", {}).items():
        if not (isinstance(splits, dict) and "train" in splits and "val" in splits):
            continue
        if train_rows is None:
            train_rows = data_mod.load_split(workspace, "train")
        candidate = candidates_mod.load_candidate(workspace, cand_name)
        flags = list(guards.memorization_check(
            candidate.source, splits["train"]["mean"], splits["val"]["mean"],
            train_rows, schema,
        ))
        if flags:
            scores.setdefault("flags", {})[cand_name] = flags
        else:
            scores.get("flags", {}).pop(cand_name, None)


def reconcile_config(workspace, scores: dict) -> dict:
    """Reconcile stored scores to the current primary/weights (config, not code).

    The top-level of every record mirrors the *primary* metric, and multi-output
    aggregates fold in the approval *weights*. Both are config that can change
    without touching metric.py's sha, so a change here would otherwise leave the
    mirror pointing at the old primary (making finalize compare an old-primary
    val mean against a new-primary test mean — a fabricated overfit demotion).

    On a change, each candidate/split is recomputed from its outputs cache when
    available (exact under the new primary AND weights, free — no runs), else the
    primary mirror is re-derived from the already-stored per-objective vectors
    (exact for a primary switch; weights stay as last scored when uncached).
    Never charges the budget. A no-op when primary and weights are unchanged.
    """
    if not scores.get("candidates"):
        return scores
    if metric.metric_sha(workspace) is None:
        # No metric.py — there is no primary/weights config to reconcile against
        # (this also keeps load_scores usable for hand-built score fixtures).
        return scores
    current_primary = metric.primary_name(workspace)
    current_wfp = _weights_fingerprint(_approval_weights(workspace))
    # Only reconcile when a RECORDED config actually differs. Absent primary /
    # weights_fp (None) means the scores predate config tracking (or are a
    # hand-built fixture); leave those untouched rather than rewriting a mirror
    # whose provenance we don't know.
    stored_primary = scores.get("primary")
    stored_wfp = scores.get("weights_fp")
    primary_changed = stored_primary is not None and stored_primary != current_primary
    weights_changed = stored_wfp is not None and stored_wfp != current_wfp
    if not (primary_changed or weights_changed):
        return scores

    metrics = schema = weights = data_mod = None
    changed = False
    for cand_name, entry in scores["candidates"].items():
        if not isinstance(entry, dict):
            continue
        for split in ("train", "val"):
            sub = entry.get(split)
            if not isinstance(sub, dict):
                continue
            cached = read_output_cache(workspace, cand_name, split)
            if cached is not None:
                if metrics is None:
                    metrics = metric.quality_metrics(workspace)
                    schema = workspace.schema
                    weights = _approval_weights(workspace)
                    data_mod = _sibling("data")
                rows = data_mod.load_split(workspace, split)
                expected_by_row = {
                    f"row_{i}": schema.coerce_expected(r) for i, r in enumerate(rows)
                }
                entry[split] = _aggregate_from_cache(
                    cached["rows"], metrics, current_primary, schema, weights,
                    expected_by_row, _reps_in_cache(cached["rows"]),
                )
                entry[split]["candidate_sha"] = cached["source_sha"]
                changed = True
            else:
                objs = sub.get("objectives") or {}
                prim = objs.get(current_primary)
                if prim is not None and "mean" in prim:
                    sub["rows"] = dict(prim.get("rows", {}))
                    sub["mean"] = prim["mean"]
                    sub["std"] = prim.get("std", sub.get("std", 0.0))
                    sub["ci95"] = list(prim.get("ci95", sub.get("ci95", [prim["mean"], prim["mean"]])))
                    sub["per_field"] = prim.get("per_field")
                    changed = True

    scores["primary"] = current_primary
    scores["weights_fp"] = current_wfp
    if changed:
        _recompute_flags(workspace, scores)
        save_scores(workspace, scores)
    return scores


def load_scores(workspace) -> dict:
    """Read scores.json, reconciling it first if the metric changed underneath it.

    Scores under different metrics are never comparable, so a stored
    ``metric_sha`` that no longer matches re-scores every candidate whose
    outputs are still cached (free) and archives only those that cannot be
    recovered — see :func:`reconcile_metric_change`.
    """
    current = metric.metric_sha(workspace)
    path = Path(workspace.scores_json)
    if not path.exists():
        return metric.scores_skeleton(current)
    scores = json.loads(path.read_text())
    if scores.get("metric_sha") != current:
        if scores.get("candidates"):
            reconcile_metric_change(workspace, current)
            scores = json.loads(path.read_text())
        else:
            scores["metric_sha"] = current
    scores.setdefault("candidates", {})
    scores.setdefault("val_scored", [])
    scores.setdefault("flags", {})
    scores.setdefault("objectives", {})
    scores.setdefault("primary", None)
    # Metric code is now current; reconcile primary/weights config too.
    scores = reconcile_config(workspace, scores)
    return scores


def save_scores(workspace, scores: dict) -> None:
    """Write scores.json."""
    Path(workspace.scores_json).write_text(json.dumps(scores, indent=2) + "\n")


def _archive_scores(workspace, old: dict) -> Path:
    archive_dir = Path(workspace.root) / "scores.archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    k = 0
    while (archive_dir / f"{k}.json").exists():
        k += 1
    dest = archive_dir / f"{k}.json"
    dest.write_text(json.dumps(old, indent=2) + "\n")
    return dest


def rescore(workspace, candidate_name: str, split: str) -> bool:
    """Recompute a candidate/split's objectives from cached outputs. No runs.

    Returns False (caller must re-run) when the outputs cache is stale or
    absent. Never charges the budget — the whole point is that adding or
    editing a metric re-scores cached predictions instead of re-executing.
    """
    cached = read_output_cache(workspace, candidate_name, split)
    if cached is None:
        return False
    metrics = metric.quality_metrics(workspace)
    primary = metric.primary_name(workspace)
    schema = workspace.schema
    weights = _approval_weights(workspace)
    rows = _sibling("data").load_split(workspace, split)
    expected_by_row = {
        f"row_{i}": schema.coerce_expected(r) for i, r in enumerate(rows)
    }
    sub = _aggregate_from_cache(
        cached["rows"], metrics, primary, schema, weights,
        expected_by_row, _reps_in_cache(cached["rows"]),
    )
    sub["candidate_sha"] = cached["source_sha"]
    scores = load_scores(workspace)
    scores["candidates"].setdefault(candidate_name, {})[split] = sub
    scores["objectives"] = _objective_directions(metrics)
    scores["primary"] = primary
    scores["weights_fp"] = _weights_fingerprint(weights)
    scores["metric_sha"] = metric.metric_sha(workspace)
    save_scores(workspace, scores)
    return True


def reconcile_metric_change(workspace, new_sha: str | None) -> None:
    """Reconcile scores.json to a changed metric — re-score from cache, else archive.

    For every candidate/split with a *fresh* outputs cache, recompute the
    quality objectives under the new metric for free (cost/latency are
    metric-independent and survive). Candidates whose cache is stale or absent
    are unrecoverable: their old scores are archived and a loud warning fires,
    but only those — everything recoverable stays live. This is the single
    reconciliation point used by both ``metric.write_metric`` and
    :func:`load_scores`.
    """
    path = Path(workspace.scores_json)
    old: dict | None = None
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except json.JSONDecodeError:
            old = None
    candidates = old.get("candidates", {}) if isinstance(old, dict) else {}

    fresh = metric.scores_skeleton(new_sha)
    if isinstance(old, dict):
        fresh["val_scored"] = list(dict.fromkeys(old.get("val_scored", [])))
        fresh["flags"] = dict(old.get("flags", {}))
    if not candidates:
        save_scores(workspace, fresh)
        return

    recover: list[tuple[str, str, dict]] = []
    lost = False
    for cand_name, entry in candidates.items():
        if not isinstance(entry, dict):
            continue
        for split in ("train", "val"):
            if split not in entry:
                continue
            cached = read_output_cache(workspace, cand_name, split)
            if cached is not None:
                recover.append((cand_name, split, cached))
            else:
                lost = True

    if recover:
        metrics = metric.quality_metrics(workspace)
        primary = metric.primary_name(workspace)
        schema = workspace.schema
        weights = _approval_weights(workspace)
        data_mod = _sibling("data")
        for cand_name, split, cached in recover:
            rows = data_mod.load_split(workspace, split)
            expected_by_row = {
                f"row_{i}": schema.coerce_expected(r) for i, r in enumerate(rows)
            }
            sub = _aggregate_from_cache(
                cached["rows"], metrics, primary, schema, weights,
                expected_by_row, _reps_in_cache(cached["rows"]),
            )
            sub["candidate_sha"] = cached["source_sha"]
            fresh["candidates"].setdefault(cand_name, {})[split] = sub
        fresh["objectives"] = _objective_directions(metrics)
        fresh["primary"] = primary
        fresh["weights_fp"] = _weights_fingerprint(weights)
        # Recompute memorization where both splits recovered — the flag depends
        # on the (now re-scored) train/val means under the new metric.
        guards = _sibling("guards")
        candidates_mod = _sibling("candidates")
        for cand_name, splits in fresh["candidates"].items():
            if "train" in splits and "val" in splits:
                candidate = candidates_mod.load_candidate(workspace, cand_name)
                train_rows = data_mod.load_split(workspace, "train")
                flags = list(guards.memorization_check(
                    candidate.source, splits["train"]["mean"],
                    splits["val"]["mean"], train_rows, schema,
                ))
                if flags:
                    fresh["flags"][cand_name] = flags
                else:
                    fresh["flags"].pop(cand_name, None)

    fresh["val_scored"] = [n for n in fresh["val_scored"] if n in fresh["candidates"]]
    fresh["flags"] = {k: v for k, v in fresh["flags"].items() if k in fresh["candidates"]}

    if lost:
        dest = _archive_scores(workspace, old)
        n_lost = len(candidates) - len(fresh["candidates"])
        msg = (
            f"metric.py changed: {n_lost} candidate score set(s) could not be "
            f"recovered from cached outputs (their code changed, or was never "
            f"cached) and were archived to {dest}. Scores under different metrics "
            f"are never comparable — re-evaluate those candidates under the new "
            f"metric. Candidates whose outputs were still cached were re-scored "
            f"from cache for free."
        )
        print(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)

    save_scores(workspace, fresh)


# ------------------------------------------------------------------ evaluate


def evaluate(
    workspace,
    candidate_name: str,
    split: str = "val",
    per_instance: bool = False,
    n_repeats: int | None = None,
) -> EvalReport:
    """Score one candidate on one split under every objective and persist it.

    Order is load-bearing: data-discipline guards run before anything else,
    metric approval before any candidate is loaded, val selection pressure is
    registered before any spend, and the budget is checked before every row —
    a mid-eval BudgetExceededError propagates with nothing persisted for the
    partial eval (neither scores nor the outputs cache). One run per (row,
    repeat) is scored under all quality metrics and its cost/latency read from
    the same RunResult, so extra metrics add no runs. Per-row score maps are
    persisted for both train and val, but ``per_row`` is only returned for
    train with ``per_instance=True``.
    """
    guards = _sibling("guards")
    guards.assert_eval_allowed(split, per_instance)

    metric.ensure_approved(workspace)
    metrics = metric.quality_metrics(workspace)
    primary = metric.primary_name(workspace)
    approval = json.loads(Path(workspace.metric_approval).read_text())
    weights = approval.get("weights")

    candidates_mod = _sibling("candidates")
    data_mod = _sibling("data")
    candidate = candidates_mod.load_candidate(workspace, candidate_name)
    rows = data_mod.load_split(workspace, split)

    if split == "val":
        # A metric change must be reconciled (recover from cache, or archive +
        # reset on disk) BEFORE the val registration is written: guards appends
        # to scores.json as it stands, and a stale metric_sha there would make
        # the later load_scores() reconcile the registration away.
        load_scores(workspace)
        guards.register_val_candidate(workspace, candidate.name)

    if n_repeats is not None:
        if n_repeats < 1:
            raise ValueError(f"n_repeats must be at least 1, got {n_repeats!r}.")
        reps = n_repeats
    else:
        reps = 1 if candidate.deterministic else DEFAULT_REPEATS

    runner_mod = _sibling("runner")
    ledger = BudgetLedger(workspace.budget_json)
    schema = workspace.schema

    cache_rows: dict[str, list[dict]] = {}
    expected_by_row: dict[str, dict] = {}
    run_errors: list[str] = []

    session_factory = getattr(runner_mod, "candidate_session", None)
    session_cls = getattr(runner_mod, "CandidateSession", None)
    session_context = (
        session_factory(workspace, candidate)
        if session_factory is not None
        else session_cls(workspace, candidate)
        if session_cls is not None
        else contextlib.nullcontext(None)
    )
    with session_context as candidate_session:
        run_one = (
            candidate_session.run
            if candidate_session is not None
            else lambda inputs: runner_mod.run_candidate(workspace, candidate, inputs)
        )
        for i, row in enumerate(rows):
            row_id = f"row_{i}"
            ledger.check()
            inputs = schema.coerce_inputs(row)
            expected_by_row[row_id] = schema.coerce_expected(row)
            reps_list: list[dict] = []
            for rep in range(1, reps + 1):
                run = run_one(inputs)
                ledger.charge(
                    eval_calls=1,
                    dollars=getattr(run, "cost_dollars", None) or 0.0,
                    category="candidate",
                )
                reps_list.append(_run_cache_entry(run))
                if not run.ok:
                    lines = (
                        getattr(run, "error", None) or "candidate failed"
                    ).strip().splitlines()
                    # The final traceback line carries the actionable exception
                    # (missing credential/module/CUDA, etc.); preserving it lets
                    # the portfolio distinguish an environmental blocker from a
                    # bad prediction and ask a human before excluding the avenue.
                    summary = lines[-1] if lines else "candidate failed"
                    run_errors.append(f"{row_id} repeat {rep}: {summary}")
            cache_rows[row_id] = reps_list

    sub = _aggregate_from_cache(
        cache_rows, metrics, primary, schema, weights, expected_by_row, reps
    )
    sub["candidate_sha"] = _source_sha(workspace, candidate)

    scores = load_scores(workspace)
    entry = scores["candidates"].setdefault(candidate.name, {})
    entry[split] = sub
    scores["objectives"] = _objective_directions(metrics)
    scores["primary"] = primary
    scores["weights_fp"] = _weights_fingerprint(weights)

    flags: list[str] = []
    if "train" in entry and "val" in entry:
        train_rows = rows if split == "train" else data_mod.load_split(workspace, "train")
        flags = list(
            guards.memorization_check(
                candidate.source,
                entry["train"]["mean"],
                entry["val"]["mean"],
                train_rows,
                schema,
            )
        )
        if flags:
            scores["flags"][candidate.name] = flags
        else:
            scores["flags"].pop(candidate.name, None)
    save_scores(workspace, scores)
    write_output_cache(workspace, candidate, split, cache_rows)
    if flags:
        warnings.warn("; ".join(flags), MemorizationWarning, stacklevel=2)

    return EvalReport(
        candidate=candidate.name,
        split=split,
        mean=sub["mean"],
        std=sub["std"],
        ci95=(sub["ci95"][0], sub["ci95"][1]),
        n_rows=len(rows),
        n_repeats=reps,
        repeat_variance=sub["repeat_variance"],
        per_row=dict(sub["rows"]) if (split == "train" and per_instance) else None,
        per_field=sub["per_field"],
        errors=run_errors,
        flags=flags,
        objectives=_objectives_report(sub),
        primary=primary,
        cold_start_s=sub.get("cold_start_s"),
    )


def _row_index(row_id: str) -> int:
    try:
        return int(row_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _cand_key(name: str):
    prefix, _, idx = name.rpartition("_")
    if prefix and idx.isdigit():
        return (0, int(idx), name)
    return (1, 0, name)


# ---------------------------------------------------------- score provenance


def score_provenance_current(workspace, candidate_name: str, split: str) -> bool:
    """Whether a stored split score belongs to the candidate currently on disk.

    Scores written before provenance tracking have no ``candidate_sha`` and are
    accepted for backwards compatibility. Every new evaluation is pinned.
    """
    scores = load_scores(workspace)
    sub = scores.get("candidates", {}).get(candidate_name, {}).get(split)
    if not isinstance(sub, dict):
        return False
    pinned = sub.get("candidate_sha")
    if not pinned:
        return True
    try:
        candidate = _sibling("candidates").load_candidate(workspace, candidate_name)
    except Exception:
        return False
    return pinned == _source_sha(workspace, candidate)


def assert_score_provenance(workspace, candidate_name: str, split: str) -> None:
    if not score_provenance_current(workspace, candidate_name, split):
        raise DataDisciplineError(
            f"Stored {split} scores for {candidate_name!r} are stale: the candidate "
            "file changed after evaluation. Selection numbers are pinned to exact "
            "source code, so re-run prg.eval() before comparing or finalizing it."
        )


# -------------------------------------------------------------------- compare


def compare(
    workspace, a: str, b: str, split: str = "val", objective: str | None = None
) -> CompareReport:
    """Compare two candidates using per-row scores already stored in scores.json.

    Rows are paired by row id, so the difference is a paired statistic —
    row-to-row difficulty cancels out. ``objective=None`` compares the primary
    quality metric (the back-compat default); otherwise the named objective's
    stored per-row scores are paired. Both candidates must have been evaluated
    on the split first; comparisons never trigger new spend.
    """
    scores = load_scores(workspace)
    stored = scores.get("candidates", {})
    for candidate_name in (a, b):
        if isinstance(stored.get(candidate_name, {}).get(split), dict):
            assert_score_provenance(workspace, candidate_name, split)
    row_maps: dict[str, dict] = {}
    label = "primary" if objective is None else f"{objective!r} objective"
    for name in (a, b):
        sub = stored.get(name, {}).get(split, {})
        if objective is None:
            rows = sub.get("rows")
        else:
            rows = ((sub.get("objectives") or {}).get(objective) or {}).get("rows")
        if not rows:
            raise DataDisciplineError(
                f"Cannot compare {a!r} vs {b!r} on {split!r} ({label}): {name!r} "
                f"has no stored per-row {split} scores for it. Comparisons pair "
                f"the per-row scores recorded by evaluate() so the difference is "
                f"measured on identical rows; run prg.eval({name!r}, "
                f"split={split!r}) first (every objective is scored from that one run)."
            )
        row_maps[name] = rows
    common = sorted(set(row_maps[a]) & set(row_maps[b]), key=_row_index)
    if not common:
        raise DataDisciplineError(
            f"Cannot compare {a!r} vs {b!r} on {split!r} ({label}): their stored "
            f"per-row scores share no row ids, so no paired difference exists. "
            f"Re-evaluate both candidates on the current {split} split."
        )
    a_vals = [float(row_maps[a][r]) for r in common]
    b_vals = [float(row_maps[b][r]) for r in common]
    if any(
        (not math.isfinite(v)) or (objective == "cost_dollars" and v >= UNKNOWN_COST)
        for v in (*a_vals, *b_vals)
    ):
        raise DataDisciplineError(
            f"Cannot compare {a!r} vs {b!r} on {objective!r}: at least one "
            "candidate did not report this objective. Unknown monetary cost is "
            "not treated as free; declare [tool.ap] cost_per_call or set "
            "AP_COST_DOLLARS after predict()."
        )
    diff_mean = fmean(bv - av for av, bv in zip(a_vals, b_vals))
    ci95 = paired_bootstrap_diff(a_vals, b_vals)
    # "Improved" is direction-aware: the challenger wins when the CI of b - a
    # excludes zero on the better side. For a maximized objective (quality)
    # that means the whole CI is above 0; for a minimized objective (cost,
    # latency) a smaller value is better, so the whole CI must be below 0.
    goal = _objective_goal(scores, objective)
    improved = ci95[1] < 0 if goal == "min" else ci95[0] > 0
    return CompareReport(
        a=a, b=b, split=split, diff_mean=diff_mean, ci95=ci95,
        improved=improved, objective=objective, goal=goal,
    )


def _objective_goal(scores: dict, objective: str | None) -> str:
    """The optimization direction of a compared objective ("max"/"min").

    ``objective=None`` is the primary quality metric, always maximized. A named
    objective's goal comes from the stored ``objectives`` directions, falling
    back to COST_OBJECTIVES membership.
    """
    if objective is None:
        return "max"
    stored = (scores.get("objectives") or {}).get(objective)
    if stored in ("min", "max"):
        return stored
    return COST_OBJECTIVES.get(objective, "max")


# ------------------------------------------------------------------- tradeoffs


def tradeoffs(workspace, split: str = "val", names=None) -> TradeoffReport:
    """The Pareto frontier over candidates' stored objective vectors for a split.

    Builds each candidate's objective means from ``["objectives"]`` on the
    given split (default all that have them), computes the nondominated set
    (quality maximized, cost minimized), and sorts rows by the primary metric
    descending so the headline candidate leads.
    """
    scores = load_scores(workspace)
    cands = scores.get("candidates", {})
    goals = dict(scores.get("objectives") or {})
    points: dict[str, dict] = {}
    for name, entry in cands.items():
        if not score_provenance_current(workspace, name, split):
            continue
        sub = entry.get(split)
        if not isinstance(sub, dict):
            continue
        objs = sub.get("objectives")
        if not objs:
            continue
        points[name] = {o: float(v["mean"]) for o, v in objs.items()}
    if names is not None:
        points = {n: points[n] for n in names if n in points}
    if not goals:
        goals = _directions_from_points(points)
    nondominated = set(pareto_nondominated(_domination_points(points), goals))
    primary = scores.get("primary")
    rows = [
        {"candidate": name, "objectives": points[name], "dominated": name not in nondominated}
        for name in points
    ]
    if primary and all(primary in p for p in points.values()):
        rows.sort(key=lambda r: (-r["objectives"][primary], _cand_key(r["candidate"])))
    else:
        rows.sort(key=lambda r: _cand_key(r["candidate"]))
    return TradeoffReport(
        rows=rows, nondominated=sorted(nondominated, key=_cand_key), split=split
    )
