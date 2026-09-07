"""Approach adherence checks for implementation-only Pi workers.

A portfolio avenue is an experiment in one implementation mechanism.  A worker
may make that mechanism robust, or report that its environment blocks it, but it
must never quietly replace it with another tier merely to return a working
function.  This module provides the deterministic half of the pre-import gate;
the Pi controller adds an independent semantic review because source heuristics
alone cannot establish architectural intent.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .portfolio import ApproachTier, AvenueSpec


@dataclass(frozen=True)
class ApproachAudit:
    """Structured result of checking one worker solution against its avenue."""

    adherent: bool
    required_mechanisms_found: tuple[str, ...] = ()
    forbidden_substitutions_found: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()
    repair_instructions: str = ""
    reviewer: str = "deterministic"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict, *, reviewer: str = "pi") -> "ApproachAudit":
        def strings(raw) -> tuple[str, ...]:
            if isinstance(raw, str):
                return (raw,)
            if not isinstance(raw, (list, tuple)):
                return ()
            return tuple(str(v) for v in raw)

        return cls(
            # Fail closed on malformed reviewer output: only JSON true accepts.
            adherent=value.get("adherent") is True,
            required_mechanisms_found=strings(
                value.get("required_mechanisms_found", ())
            ),
            forbidden_substitutions_found=strings(
                value.get("forbidden_substitutions_found", ())
            ),
            violations=strings(value.get("violations", ())),
            repair_instructions=str(value.get("repair_instructions") or ""),
            reviewer=reviewer,
        )


_API_MARKERS = (
    "openai", "anthropic", "google.generativeai", "google.genai", "groq",
    "mistral", "openrouter", "together", "fireworks", "deepseek", "xai",
    "bedrock", "boto3", "httpx", "requests",
)
_DEEP_MARKERS = (
    "torch", "transformers", "tensorflow", "keras", "jax", "flax",
    "onnxruntime", "sentence_transformers", "sentence-transformers",
)
_CLASSICAL_MARKERS = (
    "sklearn", "scikit-learn", "xgboost", "lightgbm", "catboost",
    "randomforest", "random_forest", "svc(", "svm", "tfidf", "tf-idf",
    "opencv", "cv2",
)
_RULE_MARKERS = ("regex", "re.compile", "lookup", "rule_based", "rules =")
_ENV_BRANCH = re.compile(
    r"(?is)(?:if|unless|except).{0,180}(?:api[_ ]?key|credential|cuda|gpu|"
    r"importerror|modulenotfound|no module|unavailable|not installed).{0,500}"
)
_FALLBACK_WORD = re.compile(r"(?i)fallback|fall back|degrad(?:e|ed|ation)|otherwise use")


def _contains_any(source: str, markers: tuple[str, ...]) -> bool:
    lowered = source.lower()
    return any(marker in lowered for marker in markers)


def deterministic_audit(spec: AvenueSpec, source: str) -> ApproachAudit:
    """Catch clear mechanism substitutions before spending on semantic review.

    These checks intentionally handle only high-confidence cases.  Pre/post
    processing can legitimately cross library boundaries, so an independent Pi
    reviewer decides subtler cases from the complete mechanism contract.
    """
    violations: list[str] = []
    found: list[str] = []
    forbidden: list[str] = []

    if not re.search(r"(?m)^def\s+predict\s*\(", source):
        violations.append("solution.py does not define predict with a top-level def")

    tier = spec.tier
    needs_pi = any(
        item.startswith("pi-model:")
        for item in (*spec.required_capabilities, *spec.required_mechanisms)
    )
    declares_pi_runtime = bool(
        re.search(r"(?im)^\s*#\s*pi_runtime\s*=\s*true\s*$", source)
    )
    has_api = _contains_any(source, _API_MARKERS) or (
        needs_pi and declares_pi_runtime
    )
    has_deep = _contains_any(source, _DEEP_MARKERS)
    has_classical = _contains_any(source, _CLASSICAL_MARKERS)
    has_rules = _contains_any(source, _RULE_MARKERS)

    if needs_pi and not declares_pi_runtime:
        violations.append(
            "Pi-backed implementation must declare `pi_runtime = true` under "
            "[tool.ap] so OAuth-bound evaluation remains beside the host"
        )

    if tier in (
        ApproachTier.GENERALIST_AGENT,
        ApproachTier.MODEL_GRAPH,
        ApproachTier.SINGLE_MODEL_CALL,
    ):
        if has_api:
            found.append("runtime model/provider call")
        elif has_classical:
            violations.append(
                "the assigned runtime-model approach visibly consists of a local "
                "classical implementation instead"
            )
    elif tier in (
        ApproachTier.FINETUNED_MODEL,
        ApproachTier.SPECIALIZED_DEEP_MODEL,
    ):
        if has_deep:
            found.append("deep-model runtime")
        elif has_classical:
            violations.append(
                "the assigned deep-model approach visibly consists of a classical "
                "implementation instead"
            )
    elif tier == ApproachTier.CLASSICAL_ML:
        if has_classical:
            found.append("classical model")
    elif tier == ApproachTier.CODE_AND_RULES:
        if has_api or has_deep:
            forbidden.append("model/API implementation inside a code-and-rules avenue")
            violations.append(
                "the code-and-rules avenue was replaced by a model or API approach"
            )
        elif has_rules:
            found.append("algorithmic/rule mechanism")

    # The characteristic failure this gate exists to prevent: an exception or
    # missing-resource branch quietly invokes another family.  Avoid rejecting
    # ordinary preprocessing unless fallback intent is explicit in the source.
    fallback_context = "\n".join(
        match.group(0) for match in _ENV_BRANCH.finditer(source)
    )
    explicit_fallback = bool(_FALLBACK_WORD.search(source))
    if tier in (
        ApproachTier.GENERALIST_AGENT,
        ApproachTier.MODEL_GRAPH,
        ApproachTier.SINGLE_MODEL_CALL,
    ) and (has_classical or has_rules) and (explicit_fallback or fallback_context):
        item = "classical/rule fallback in a runtime-model avenue"
        forbidden.append(item)
        violations.append(
            "cross-family fallback is forbidden: missing credentials or provider "
            "failures must raise clearly; they may not route to classical ML, CV, "
            "rules, regex, or lookup logic"
        )
    if tier in (
        ApproachTier.FINETUNED_MODEL,
        ApproachTier.SPECIALIZED_DEEP_MODEL,
    ) and (has_classical or has_rules) and (explicit_fallback or fallback_context):
        item = "classical/rule fallback in a deep-model avenue"
        forbidden.append(item)
        violations.append(
            "cross-family fallback is forbidden: missing Torch/model/GPU resources "
            "must raise clearly; they may not route to classical ML, CV, rules, "
            "regex, or lookup logic"
        )

    # ``forbidden_substitutions`` is semantic prose, not a source-token denylist:
    # code may legitimately say "no classical fallback" in a comment. The Pi
    # reviewer applies those task-specific constraints to control flow.
    adherent = not violations
    repair = ""
    if violations:
        repair = (
            "Remove every cross-approach substitute and implement the assigned "
            "mechanism as the actual prediction path. If its dependency or runtime "
            "capability is unavailable, fail with a precise error instead of "
            "returning an answer from another mechanism."
        )
    return ApproachAudit(
        adherent=adherent,
        required_mechanisms_found=tuple(found),
        forbidden_substitutions_found=tuple(dict.fromkeys(forbidden)),
        violations=tuple(dict.fromkeys(violations)),
        repair_instructions=repair,
    )


def semantic_audit_prompt(spec: AvenueSpec, source: str) -> str:
    """Prompt for an implementation-blind-spot reviewer (no data or metrics)."""
    contract = {
        "tier": int(spec.tier),
        "title": spec.title,
        "hypothesis": spec.hypothesis,
        "mechanism": spec.mechanism,
        "implementation_brief": spec.implementation_brief,
        "required_mechanisms": list(spec.required_mechanisms),
        "forbidden_substitutions": list(spec.forbidden_substitutions),
        "allow_cross_tier_fallback": spec.allow_cross_tier_fallback,
    }
    import json

    return f"""Review whether this Python implementation faithfully uses its assigned
mechanism. Judge mechanism adherence only, not task quality. Pre/post-processing,
retries, and error handling are allowed, but an implementation from another
family must never produce the answer when the assigned mechanism is unavailable.
Missing packages, models, API credentials, GPU, or network must cause a clear
failure, not a classical/rules/model substitute. The only exception is when the
contract explicitly permits cross-tier fallback (normally composition only).

Return exactly one JSON object:
{{"adherent": true, "required_mechanisms_found": ["..."],
  "forbidden_substitutions_found": [], "violations": [],
  "repair_instructions": ""}}

Approach contract:
{json.dumps(contract, indent=2)}

Implementation source:
---
{source}
---
"""
