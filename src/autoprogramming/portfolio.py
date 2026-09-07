"""Resource-aware, breadth-first portfolios of implementation avenues.

The language model may suggest avenues, but this module is the deterministic
policy layer: it validates coverage, prevents early convergence, and records
why an approach family was excluded.  The orchestrator cannot waive these
rules merely because one local idea already has an acceptable score.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from enum import IntEnum, StrEnum

from .resources import Resources


class ApproachTier(IntEnum):
    GENERALIST_AGENT = 1
    MODEL_GRAPH = 2
    SINGLE_MODEL_CALL = 3
    FINETUNED_MODEL = 4
    SPECIALIZED_DEEP_MODEL = 5
    CLASSICAL_ML = 6
    CODE_AND_RULES = 7
    COMPOSITION = 8


TIER_LABELS = {
    ApproachTier.GENERALIST_AGENT: "runtime generalist or coding agent",
    ApproachTier.MODEL_GRAPH: "graph of model calls",
    ApproachTier.SINGLE_MODEL_CALL: "single model call",
    ApproachTier.FINETUNED_MODEL: "fine-tuned language model",
    ApproachTier.SPECIALIZED_DEEP_MODEL: "specialized pretrained/deep model",
    ApproachTier.CLASSICAL_ML: "classical machine learning",
    ApproachTier.CODE_AND_RULES: "algorithms, features, and rules",
    ApproachTier.COMPOSITION: "cross-tier cascade, router, ensemble, or pipeline",
}


class AvenueStatus(StrEnum):
    PLANNED = "planned"
    RUNNING = "running"
    READY = "ready"
    EVALUATED = "evaluated"
    STAGNANT = "stagnant"
    BLOCKED = "blocked"
    NONCOMPLIANT = "noncompliant"
    INFEASIBLE = "infeasible"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(frozen=True)
class AvenueSpec:
    """One independent hypothesis an implementation-only worker will pursue."""

    id: str
    tier: ApproachTier
    title: str
    hypothesis: str
    implementation_brief: str
    mechanism: str
    runtime_requirements: tuple[str, ...] = ()
    allowed_api_providers: tuple[str, ...] = ()
    required_capabilities: tuple[str, ...] = ()
    required_mechanisms: tuple[str, ...] = ()
    forbidden_substitutions: tuple[str, ...] = ()
    allow_cross_tier_fallback: bool = False
    max_rounds: int = 3
    wildcard: bool = False
    compose_from: tuple[str, ...] = ()
    research_sources: tuple[str, ...] = ()
    deployment_notes: tuple[str, ...] = ()
    worker_model: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tier", ApproachTier(int(self.tier)))
        object.__setattr__(self, "runtime_requirements", tuple(self.runtime_requirements))
        object.__setattr__(self, "allowed_api_providers", tuple(self.allowed_api_providers))
        object.__setattr__(self, "required_capabilities", tuple(self.required_capabilities))
        object.__setattr__(self, "required_mechanisms", tuple(self.required_mechanisms))
        object.__setattr__(self, "forbidden_substitutions", tuple(self.forbidden_substitutions))
        object.__setattr__(self, "compose_from", tuple(self.compose_from))
        object.__setattr__(self, "research_sources", tuple(self.research_sources))
        object.__setattr__(self, "deployment_notes", tuple(self.deployment_notes))
        if self.allow_cross_tier_fallback and self.tier != ApproachTier.COMPOSITION:
            raise ValueError(
                "Cross-tier fallback is reserved for an explicit composition avenue; "
                f"{self.id!r} is tier {int(self.tier)}."
            )
        if not self.id or not self.id.replace("-", "_").isidentifier():
            raise ValueError(f"Invalid avenue id {self.id!r}.")
        if not all((self.title.strip(), self.hypothesis.strip(), self.mechanism.strip())):
            raise ValueError(f"Avenue {self.id!r} needs a title, hypothesis, and mechanism.")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1.")
        if self.worker_model is not None and not self.worker_model.strip():
            raise ValueError("worker_model must be a non-empty Pi model pattern.")

    @property
    def fingerprint(self) -> str:
        normalized = " ".join(self.mechanism.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        result = asdict(self)
        result["tier"] = int(self.tier)
        result["fingerprint"] = self.fingerprint
        return result

    @classmethod
    def from_dict(cls, value: dict) -> "AvenueSpec":
        accepted = {
            key: value[key]
            for key in (
                "id", "tier", "title", "hypothesis", "implementation_brief",
                "mechanism", "runtime_requirements", "allowed_api_providers",
                "required_capabilities", "required_mechanisms",
                "forbidden_substitutions", "allow_cross_tier_fallback",
                "max_rounds", "wildcard", "compose_from",
                "research_sources", "deployment_notes", "worker_model",
            )
            if key in value
        }
        return cls(**accepted)


@dataclass
class AvenueState:
    spec: AvenueSpec
    status: AvenueStatus = AvenueStatus.PLANNED
    rounds: int = 0
    no_progress_rounds: int = 0
    candidates: list[str] = field(default_factory=list)
    pending_candidate: str | None = None
    last_objectives: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    compliance_attempts: int = 0
    restart_count: int = 0
    audits: list[dict] = field(default_factory=list)
    blocker: dict | None = None
    human_retry_confirmed: bool = False
    failures: list[dict] = field(default_factory=list)

    def begin_candidate(self, candidate: str) -> None:
        """Journal an imported candidate before evaluation starts."""
        self.pending_candidate = str(candidate)
        self.status = AvenueStatus.READY

    def record_result(self, candidate: str, objectives: dict[str, float], improved: bool) -> None:
        self.rounds += 1
        if candidate not in self.candidates:
            self.candidates.append(candidate)
        if self.pending_candidate == candidate:
            self.pending_candidate = None
        self.last_objectives = dict(objectives)
        self.status = AvenueStatus.EVALUATED
        self.no_progress_rounds = 0 if improved else self.no_progress_rounds + 1
        if self.no_progress_rounds >= 2 or self.rounds >= self.spec.max_rounds:
            self.status = AvenueStatus.STAGNANT

    def record_audit(self, audit: dict) -> None:
        self.compliance_attempts += 1
        self.audits.append(dict(audit))

    def record_failure(
        self, kind: str, details: list[str], *, candidate=None, repairable=True
    ) -> None:
        """Keep implementation evidence without declaring the family failed."""
        self.failures.append({
            "kind": str(kind),
            "details": [str(v) for v in details],
            "candidate": candidate,
            "repairable": bool(repairable),
        })

    def record_blocker(self, kind: str, details: list[str], *, candidate=None) -> None:
        """Pause this avenue until a human chooses retry or confirmed exclusion."""
        self.pending_candidate = None
        self.status = AvenueStatus.BLOCKED
        self.record_failure(kind, details, candidate=candidate)
        self.human_retry_confirmed = False
        self.blocker = {
            "kind": str(kind),
            "details": [str(v) for v in details],
            "candidate": candidate,
            "human_confirmation_required": True,
        }
        self.notes.append(
            f"blocked ({kind}); waiting for human confirmation before retry or exclusion"
        )

    def resolve_blocker(self, action: str, confirmed_by: str) -> None:
        if self.status != AvenueStatus.BLOCKED or self.blocker is None:
            raise ValueError(f"Avenue {self.spec.id!r} has no unresolved blocker.")
        if action not in ("retry", "exclude"):
            raise ValueError("Blocker action must be 'retry' or 'exclude'.")
        if not str(confirmed_by).strip():
            raise ValueError("confirmed_by must identify the human who made the decision.")
        resolution = {
            "action": action,
            "confirmed_by": str(confirmed_by),
            "blocker": self.blocker,
        }
        self.notes.append(f"human blocker resolution: {json.dumps(resolution, sort_keys=True)}")
        self.blocker = None
        self.human_retry_confirmed = action == "retry"
        if action == "exclude":
            self.status = AvenueStatus.INFEASIBLE
        else:
            self.status = (
                AvenueStatus.EVALUATED
                if self.candidates
                else AvenueStatus.PLANNED
            )

    @classmethod
    def from_dict(cls, value: dict) -> "AvenueState":
        return cls(
            spec=AvenueSpec.from_dict(value["spec"]),
            status=AvenueStatus(value.get("status", AvenueStatus.PLANNED.value)),
            rounds=int(value.get("rounds", 0)),
            no_progress_rounds=int(value.get("no_progress_rounds", 0)),
            candidates=list(value.get("candidates", [])),
            pending_candidate=(
                str(value["pending_candidate"])
                if value.get("pending_candidate") is not None
                else None
            ),
            last_objectives={
                str(k): float(v) for k, v in value.get("last_objectives", {}).items()
            },
            notes=list(value.get("notes", [])),
            compliance_attempts=int(value.get("compliance_attempts", 0)),
            restart_count=int(value.get("restart_count", 0)),
            audits=[dict(v) for v in value.get("audits", [])],
            blocker=(dict(value["blocker"]) if value.get("blocker") else None),
            human_retry_confirmed=bool(value.get("human_retry_confirmed", False)),
            failures=[dict(v) for v in value.get("failures", [])],
        )

    def to_dict(self) -> dict:
        return {
            "spec": self.spec.to_dict(),
            "status": self.status.value,
            "rounds": self.rounds,
            "no_progress_rounds": self.no_progress_rounds,
            "candidates": list(self.candidates),
            "pending_candidate": self.pending_candidate,
            "last_objectives": dict(self.last_objectives),
            "notes": list(self.notes),
            "compliance_attempts": self.compliance_attempts,
            "restart_count": self.restart_count,
            "audits": [dict(v) for v in self.audits],
            "blocker": self.blocker,
            "human_retry_confirmed": self.human_retry_confirmed,
            "failures": [dict(v) for v in self.failures],
        }


@dataclass(frozen=True)
class PortfolioPolicy:
    """Controller-enforced allocation and stopping policy."""

    breadth_fraction: float = 0.4
    deepening_fraction: float = 0.4
    composition_fraction: float = 0.2
    min_configs_before_abandon: int = 2
    stagnation_rounds: int = 2
    require_wildcard: bool = True

    def __post_init__(self) -> None:
        total = self.breadth_fraction + self.deepening_fraction + self.composition_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Portfolio budget fractions must sum to 1.0, got {total}.")
        if min(
            self.breadth_fraction,
            self.deepening_fraction,
            self.composition_fraction,
        ) < 0:
            raise ValueError("Portfolio budget fractions cannot be negative.")
        if self.min_configs_before_abandon < 1 or self.stagnation_rounds < 1:
            raise ValueError("Portfolio stopping counts must be positive.")


@dataclass
class Portfolio:
    resources: Resources
    avenues: list[AvenueState]
    exclusions: dict[int, str]
    policy: PortfolioPolicy = field(default_factory=PortfolioPolicy)

    @classmethod
    def create(
        cls,
        resources: Resources,
        specs: list[AvenueSpec],
        *,
        exclusions: dict[int, str] | None = None,
        policy: PortfolioPolicy | None = None,
        fill_missing: bool = True,
    ) -> "Portfolio":
        exclusions = {int(k): str(v) for k, v in (exclusions or {}).items()}
        specs = list(specs)
        effective_policy = policy or PortfolioPolicy()
        if fill_missing:
            present = {int(s.tier) for s in specs}
            for tier, fact in resources.feasibility().items():
                if fact["feasible"] and tier not in present and tier != 8:
                    specs.append(default_avenue(ApproachTier(tier), resources))
                elif not fact["feasible"]:
                    exclusions.setdefault(tier, str(fact["reason"]))
            has_pi_experiment = any(
                any(
                    capability.startswith("pi-model:")
                    for capability in spec.required_capabilities
                )
                for spec in specs
            )
            if (
                resources.search.pi_models
                and resources.feasibility()[int(ApproachTier.SINGLE_MODEL_CALL)][
                    "feasible"
                ]
                and not has_pi_experiment
            ):
                existing_ids = {spec.id for spec in specs}
                pi_id = "pi-subscription-call"
                suffix = 2
                while pi_id in existing_ids:
                    pi_id = f"pi-subscription-call-{suffix}"
                    suffix += 1
                specs.append(AvenueSpec(
                    id=pi_id,
                    tier=ApproachTier.SINGLE_MODEL_CALL,
                    title="Authenticated Pi subscription call",
                    hypothesis=(
                        "A capable subscription model may occupy a useful "
                        "quality frontier without a raw provider API key."
                    ),
                    implementation_brief=(
                        "Implement one persistent Pi CLI/RPC model call with "
                        "task-specific instructions and robust typed parsing."
                    ),
                    mechanism="one persistent OAuth-backed Pi subscription model call",
                    required_capabilities=(
                        f"pi-model:{resources.search.pi_models[0]}",
                    ),
                    required_mechanisms=(
                        "one live Pi CLI/RPC subscription model call",
                    ),
                    forbidden_substitutions=(
                        "raw SDK API substitute", "local rules fallback",
                        "classical ML fallback",
                    ),
                    worker_model=resources.search.pi_models[0],
                ))
            if effective_policy.require_wildcard and not any(s.wildcard for s in specs):
                specs.append(AvenueSpec(
                    id="wildcard",
                    tier=ApproachTier.CODE_AND_RULES,
                    title="Non-obvious wildcard",
                    hypothesis=(
                        "A task-specific mechanism outside the obvious baselines "
                        "may expose a missed quality/cost region."
                    ),
                    implementation_brief=(
                        "Invent one non-obvious algorithmic or hybrid mechanism "
                        "that is materially different from the other assigned avenues."
                    ),
                    mechanism="task-specific wildcard mechanism unlike the planned families",
                    wildcard=True,
                ))
        specs = [ensure_avenue_contract(spec, resources) for spec in specs]
        result = cls(
            resources=resources,
            avenues=[AvenueState(spec=s) for s in specs],
            exclusions=exclusions,
            policy=effective_policy,
        )
        result.validate()
        return result

    @classmethod
    def from_dict(cls, value: dict) -> "Portfolio":
        result = cls(
            resources=Resources.from_dict(value["resources"]),
            avenues=[AvenueState.from_dict(v) for v in value.get("avenues", [])],
            exclusions={int(k): str(v) for k, v in value.get("exclusions", {}).items()},
            policy=PortfolioPolicy(**value.get("policy", {})),
        )
        result.validate()
        return result

    @classmethod
    def load(cls, path) -> "Portfolio":
        from pathlib import Path
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate(self) -> None:
        ids = [a.spec.id for a in self.avenues]
        if len(ids) != len(set(ids)):
            raise ValueError("Portfolio avenue ids must be unique.")
        fingerprints: dict[tuple[int, str], str] = {}
        for avenue in self.avenues:
            key = (int(avenue.spec.tier), avenue.spec.fingerprint)
            if key in fingerprints:
                raise ValueError(
                    f"Avenues {fingerprints[key]!r} and {avenue.spec.id!r} repeat "
                    "the same mechanism within one tier."
                )
            fingerprints[key] = avenue.spec.id

        feasibility = self.resources.feasibility()
        represented = {int(a.spec.tier) for a in self.avenues}
        missing = [
            tier for tier, info in feasibility.items()
            if tier <= 7 and info["feasible"] and tier not in represented
        ]
        if missing:
            raise ValueError(f"Portfolio omitted feasible approach tiers: {missing}.")
        unexplained = [
            tier for tier, info in feasibility.items()
            if tier <= 7 and not info["feasible"] and tier not in self.exclusions
        ]
        if unexplained:
            raise ValueError(f"Portfolio exclusions need reasons for tiers: {unexplained}.")

    @staticmethod
    def _conclusive(avenue: AvenueState) -> bool:
        """Whether an avenue supplies honest family-level evidence.

        A worker crash, noncompliant substitute, or broken implementation is
        not evidence that the assigned mechanism was attempted successfully.
        Only evaluated implementations or an explicit infeasibility/closure can
        satisfy breadth.
        """
        return avenue.status in (
            AvenueStatus.EVALUATED,
            AvenueStatus.STAGNANT,
            AvenueStatus.CLOSED,
            AvenueStatus.INFEASIBLE,
        )

    @property
    def breadth_complete(self) -> bool:
        if any(avenue.pending_candidate for avenue in self.avenues):
            return False
        if self.unresolved_blockers:
            return False
        represented = {
            int(a.spec.tier) for a in self.avenues if self._conclusive(a)
        }
        return all(
            not info["feasible"] or tier > 7 or tier in represented
            for tier, info in self.resources.feasibility().items()
        )

    @property
    def may_finalize(self) -> bool:
        if any(avenue.pending_candidate for avenue in self.avenues):
            return False
        if not self.breadth_complete:
            return False
        if self.policy.require_wildcard:
            wildcards = [a for a in self.avenues if a.spec.wildcard]
            if not wildcards or not any(self._conclusive(a) for a in wildcards):
                return False
        successful = [
            a for a in self.avenues
            if a.status in (AvenueStatus.EVALUATED, AvenueStatus.STAGNANT)
        ]
        if not successful:
            return False
        # A family is not dismissed on one brittle implementation. Successful
        # avenues need the configured number of materially different attempts
        # unless their explicit contract caps them lower (bootstrap controllers
        # use a one-configuration policy).
        if any(
            a.rounds < min(self.policy.min_configs_before_abandon, a.spec.max_rounds)
            for a in successful
        ):
            return False
        return True

    @property
    def unresolved_blockers(self) -> list[AvenueState]:
        return [
            avenue for avenue in self.avenues
            if avenue.status == AvenueStatus.BLOCKED and avenue.blocker is not None
        ]

    def resolve_blocker(self, avenue_id: str, action: str, confirmed_by: str) -> None:
        matches = [a for a in self.avenues if a.spec.id == avenue_id]
        if not matches:
            raise ValueError(f"Unknown portfolio avenue {avenue_id!r}.")
        matches[0].resolve_blocker(action, confirmed_by)

    def to_dict(self) -> dict:
        return {
            "resources": self.resources.to_dict(),
            "policy": asdict(self.policy),
            "exclusions": {str(k): v for k, v in sorted(self.exclusions.items())},
            "avenues": [a.to_dict() for a in self.avenues],
            "breadth_complete": self.breadth_complete,
            "may_finalize": self.may_finalize,
        }

    def write(self, path) -> None:
        from pathlib import Path
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")


_DEFAULTS = {
    ApproachTier.GENERALIST_AGENT: (
        "runtime-agent", "Runtime reasoning agent",
        "A generalist tool-using agent can solve ambiguous cases dynamically.",
        "Implement predict using one runtime coding/reasoning agent with a tightly scoped task prompt.",
        "one tool-using runtime agent with explicit termination and output parsing",
    ),
    ApproachTier.MODEL_GRAPH: (
        "model-graph", "Decomposed model graph",
        "Planning, solving, and checking in separate calls may outperform one monolithic prompt.",
        "Implement a small graph of model calls with explicit intermediate contracts and a final verifier.",
        "planner-solver-verifier graph with bounded calls",
    ),
    ApproachTier.SINGLE_MODEL_CALL: (
        "single-model", "Single model call",
        "A strong task-specific prompt may provide the best quality/cost balance.",
        "Implement one model call with robust formatting, parsing, retries, and honest cost reporting.",
        "single current model call with task-specific prompt and parser",
    ),
    ApproachTier.FINETUNED_MODEL: (
        "finetuned-model", "Fine-tuned model",
        "Task examples may support a compact specialized fine-tune.",
        "Build or configure a fine-tuned model and provide a lazy runtime wrapper.",
        "supervised fine-tune with held-out build validation",
    ),
    ApproachTier.SPECIALIZED_DEEP_MODEL: (
        "specialized-model", "Specialized pretrained model",
        "A task-specific pretrained architecture may solve the task locally and cheaply.",
        "Research a current specialized model, verify it loads, and implement lazy local inference.",
        "current task-specialized pretrained model used as a pipeline stage",
    ),
    ApproachTier.CLASSICAL_ML: (
        "classical-ml", "Classical machine learning",
        "Compact learned features may capture the task without a generative model.",
        "Train a classical model on the provided examples; persist artifacts and implement inference.",
        "cross-validated classical estimator with task-appropriate sparse or engineered features",
    ),
    ApproachTier.CODE_AND_RULES: (
        "code-rules", "Algorithms and rules",
        "Direct algorithms and carefully generalized rules may be fastest and most reliable.",
        "Implement a dependency-light algorithmic baseline; generalize patterns rather than memorizing examples.",
        "stdlib algorithm and generalized feature/rule system",
    ),
    ApproachTier.COMPOSITION: (
        "composition", "Cross-tier composition",
        "Complementary implementations may form a better cascade or router than either alone.",
        "Build a bounded cascade, router, pipeline, or ensemble from complementary mechanisms.",
        "confidence-aware router or cascade across complementary implementation families",
    ),
}


def ensure_avenue_contract(spec: AvenueSpec, resources: Resources) -> AvenueSpec:
    """Fill omitted machine-readable constraints from the tier contract.

    Pi may make the prose task-specific, but it cannot weaken the controller's
    default mechanism boundary by omitting contract fields from its JSON plan.
    """
    default = default_avenue(spec.tier, resources)
    pi_assigned = any(
        capability.startswith("pi-model:")
        for capability in spec.required_capabilities
    )
    permitted = tuple(
        provider for provider in spec.allowed_api_providers
        if provider in set(default.allowed_api_providers)
    )
    default_capabilities = tuple(
        capability
        for capability in default.required_capabilities
        if not (pi_assigned and capability.startswith("candidate-api:"))
    )
    pi_note = (
        (
            "requires the Pi CLI and an authenticated model subscription at "
            "deployment; no raw provider API key is required",
        )
        if pi_assigned
        else ()
    )
    if (
        spec.worker_model is not None
        and spec.worker_model not in set(resources.search.pi_models)
    ):
        raise ValueError(
            f"Avenue {spec.id!r} requested unavailable Pi worker model "
            f"{spec.worker_model!r}. Choose from search.pi_models."
        )
    return replace(
        spec,
        allowed_api_providers=(
            () if pi_assigned else (permitted or default.allowed_api_providers)
        ),
        required_capabilities=tuple(dict.fromkeys(
            (*default_capabilities, *spec.required_capabilities)
        )),
        required_mechanisms=tuple(dict.fromkeys(
            (*default.required_mechanisms, *spec.required_mechanisms)
        )),
        forbidden_substitutions=tuple(dict.fromkeys(
            (*default.forbidden_substitutions, *spec.forbidden_substitutions)
        )),
        deployment_notes=tuple(dict.fromkeys(
            (*default.deployment_notes, *pi_note, *spec.deployment_notes)
        )),
        max_rounds=(
            spec.max_rounds
            if spec.tier == ApproachTier.COMPOSITION
            else max(2, spec.max_rounds)
        ),
        worker_model=(
            spec.worker_model
            if spec.worker_model in set(resources.search.pi_models)
            else (
                resources.search.pi_models[0]
                if resources.search.pi_models else None
            )
        ),
        allow_cross_tier_fallback=(spec.tier == ApproachTier.COMPOSITION),
    )


def default_avenue(tier: ApproachTier, resources: Resources) -> AvenueSpec:
    avenue_id, title, hypothesis, brief, mechanism = _DEFAULTS[tier]
    providers = tuple(
        provider for provider in resources.runtime.api_providers
        if provider in set(resources.search.candidate_api_providers or ())
    ) if tier in (
        ApproachTier.GENERALIST_AGENT,
        ApproachTier.MODEL_GRAPH,
        ApproachTier.SINGLE_MODEL_CALL,
    ) else ()
    pi_models = tuple(resources.search.pi_models) if tier in (
        ApproachTier.GENERALIST_AGENT,
        ApproachTier.MODEL_GRAPH,
        ApproachTier.SINGLE_MODEL_CALL,
    ) else ()
    model_capabilities = (
        tuple(f"candidate-api:{provider}" for provider in providers)
        if providers
        else tuple(f"pi-model:{model}" for model in pi_models[:1])
    )
    pi_runtime_notes = (
        (
            "requires the Pi CLI and an authenticated model subscription at "
            "deployment; no raw provider API key is required",
        )
        if pi_models and not providers
        else ()
    )
    deployment_fit_notes = (
        (
            "search-feasible but outside the confirmed runtime deployment "
            "envelope; remote/build resources are not assumed in production",
        )
        if not bool(resources.feasibility()[int(tier)]["deployable"])
        else ()
    )
    contract = {
        ApproachTier.GENERALIST_AGENT: {
            "required_mechanisms": ("a live runtime tool-using reasoning agent",),
            "forbidden_substitutions": (
                "classical ML fallback", "rules fallback", "lookup fallback",
            ),
            "required_capabilities": model_capabilities,
        },
        ApproachTier.MODEL_GRAPH: {
            "required_mechanisms": ("multiple live model calls with explicit stages",),
            "forbidden_substitutions": (
                "single-call substitute", "classical ML fallback", "rules fallback",
            ),
            "required_capabilities": model_capabilities,
        },
        ApproachTier.SINGLE_MODEL_CALL: {
            "required_mechanisms": ("one live model-provider call",),
            "forbidden_substitutions": (
                "classical ML fallback", "rules fallback", "lookup fallback",
            ),
            "required_capabilities": model_capabilities,
        },
        ApproachTier.FINETUNED_MODEL: {
            "required_mechanisms": ("inference through the assigned fine-tuned model",),
            "forbidden_substitutions": (
                "base-model substitute", "classical ML fallback", "rules fallback",
            ),
            "required_capabilities": ("fine-tuning",),
        },
        ApproachTier.SPECIALIZED_DEEP_MODEL: {
            "required_mechanisms": ("inference through a task-specialized deep model",),
            "forbidden_substitutions": (
                "classical ML fallback", "classical CV fallback", "rules fallback",
            ),
            "required_capabilities": ("package-installs", "model-downloads"),
        },
        ApproachTier.CLASSICAL_ML: {
            "required_mechanisms": ("a fitted classical estimator or learned classical features",),
            "forbidden_substitutions": ("model API substitute", "deep-model substitute"),
            "required_capabilities": ("package-installs",),
        },
        ApproachTier.CODE_AND_RULES: {
            "required_mechanisms": ("a direct generalized algorithm, feature, or rule system",),
            "forbidden_substitutions": ("model API substitute", "deep-model substitute"),
            "required_capabilities": (),
        },
        ApproachTier.COMPOSITION: {
            "required_mechanisms": ("an explicit bounded composition of the listed components",),
            "forbidden_substitutions": (),
            "required_capabilities": (),
        },
    }[tier]
    return AvenueSpec(
        id=avenue_id,
        tier=tier,
        title=title,
        hypothesis=hypothesis,
        implementation_brief=brief,
        mechanism=mechanism,
        allowed_api_providers=providers,
        required_capabilities=contract["required_capabilities"],
        required_mechanisms=contract["required_mechanisms"],
        forbidden_substitutions=contract["forbidden_substitutions"],
        allow_cross_tier_fallback=(tier == ApproachTier.COMPOSITION),
        deployment_notes=tuple(dict.fromkeys(
            (*pi_runtime_notes, *deployment_fit_notes)
        )),
        worker_model=(
            resources.search.pi_models[0]
            if resources.search.pi_models else None
        ),
    )
