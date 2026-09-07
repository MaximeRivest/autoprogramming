"""Problem analysis: what the task IS, recorded before any approach is chosen.

With capable coding agents, implementation is cheap; choosing experiments is
the scarce resource.  A plan that starts from "which tier of the ladder" skips
the question that actually points to the right mechanisms: what structure the
problem has, what knowledge exists beyond examples, and what evidence would
separate a real solution from a convincing imitation.  This module persists
that analysis so ``plan_portfolio`` can require it the way it requires web
research: an enforceable phase, not prompt decoration.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .errors import AutoProgrammingError


class ProblemAnalysisError(AutoProgrammingError):
    """The problem analysis is missing or does not answer the required questions."""


@dataclass(frozen=True)
class SubProblem:
    """One part of the task that may deserve its own mechanism.

    A compound system is built from parts.  Naming them lets avenues target a
    part (``AvenueSpec.targets``) instead of only the whole task, so a weak
    standalone component can still be the right piece of a system.
    """

    id: str
    description: str
    needs: str = ""  # e.g. "exact arithmetic", "world knowledge", "judgment"

    def __post_init__(self) -> None:
        if not self.id or not self.id.replace("-", "_").isidentifier():
            raise ProblemAnalysisError(f"Invalid sub-problem id {self.id!r}.")
        if not self.description.strip():
            raise ProblemAnalysisError(f"Sub-problem {self.id!r} needs a description.")


@dataclass(frozen=True)
class ProblemAnalysis:
    """Answers to the questions that should precede any choice of technology."""

    structure: str
    """What structure the task has: language, geometry, dynamics, causality, constraints."""
    knowledge: str
    """What knowledge exists beyond the examples: laws, domain expertise, simulators, references."""
    data_regime: str
    """Rich or scarce, clean or noisy, what the labels really measure."""
    generalization: str
    """What must generalize: new inputs, new populations, future conditions, interventions."""
    evidence: str
    """What evidence separates a real solution from a convincing imitation."""
    sub_problems: tuple[SubProblem, ...] = ()
    hard_cases: str = ""
    """Which inputs are genuinely difficult and why."""
    analyzed_by: str = "host"

    _REQUIRED = ("structure", "knowledge", "data_regime", "generalization", "evidence")
    _MIN_CHARS = 40

    def __post_init__(self) -> None:
        subs = tuple(
            s if isinstance(s, SubProblem) else SubProblem(**s) for s in self.sub_problems
        )
        object.__setattr__(self, "sub_problems", subs)
        thin = [
            name for name in self._REQUIRED
            if len(str(getattr(self, name)).strip()) < self._MIN_CHARS
        ]
        if thin:
            raise ProblemAnalysisError(
                f"Problem analysis is too thin on {thin}: each of "
                f"{list(self._REQUIRED)} needs a real answer (>= {self._MIN_CHARS} "
                "characters). These questions are what point the search toward "
                "the right mechanisms; a one-word answer is the ladder in disguise."
            )
        ids = [s.id for s in self.sub_problems]
        if len(ids) != len(set(ids)):
            raise ProblemAnalysisError("Sub-problem ids must be unique.")

    @property
    def sub_problem_ids(self) -> tuple[str, ...]:
        return tuple(s.id for s in self.sub_problems)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["sub_problems"] = [asdict(s) for s in self.sub_problems]
        return d

    @classmethod
    def from_dict(cls, value: dict) -> "ProblemAnalysis":
        accepted = {
            k: value[k]
            for k in (
                "structure", "knowledge", "data_regime", "generalization",
                "evidence", "sub_problems", "hard_cases", "analyzed_by",
            )
            if k in value
        }
        return cls(**accepted)


def record_analysis(workspace, analysis: ProblemAnalysis) -> Path:
    path = Path(workspace.analysis_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = analysis.to_dict()
    payload["recorded_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_analysis(workspace) -> ProblemAnalysis | None:
    path = Path(workspace.analysis_json)
    if not path.exists():
        return None
    try:
        return ProblemAnalysis.from_dict(json.loads(path.read_text()))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProblemAnalysisError(f"Malformed problem analysis at {path}: {exc}") from exc


def ensure_analyzed(workspace) -> ProblemAnalysis:
    analysis = load_analysis(workspace)
    if analysis is None:
        raise ProblemAnalysisError(
            "Portfolio planning was refused until the problem is analyzed. Call "
            "prg.analyze_problem(structure=..., knowledge=..., data_regime=..., "
            "generalization=..., evidence=..., sub_problems=[...]) first. The "
            "analysis, not a fixed list of approach families, is what should "
            "point the search toward mechanisms worth trying."
        )
    return analysis
