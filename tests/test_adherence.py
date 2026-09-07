"""Hard mechanism fidelity: fallback implementations never enter an avenue."""

from __future__ import annotations

import json
from types import SimpleNamespace

import autoprogramming as ap

from autoprogramming.adherence import deterministic_audit
from autoprogramming.pi_backend import (
    PiOrchestratorBackend,
    PiResult,
    _complete_environment_failure,
    _missing_avenue_capabilities,
)
from autoprogramming.portfolio import (
    ApproachTier,
    AvenueState,
    default_avenue,
)


def api_resources(*, available=True):
    return ap.Resources(
        search=ap.SearchResources(
            allow_package_installs=True,
            allow_model_downloads=True,
            candidate_api_providers=(("openai",) if available else ()),
        ),
        runtime=ap.RuntimeResources(network=True, api_providers=("openai",)),
        data=ap.DataPolicy(external_egress=True),
        confirmed=True,
    )


def deep_resources():
    return ap.Resources(
        search=ap.SearchResources(
            allow_package_installs=True,
            allow_model_downloads=True,
        ),
        runtime=ap.RuntimeResources(network=False, memory_gb=8),
        data=ap.DataPolicy(external_egress=False),
        confirmed=True,
    )


def test_api_avenue_rejects_classical_fallback_for_missing_key():
    spec = default_avenue(ApproachTier.SINGLE_MODEL_CALL, api_resources())
    source = '''
import os
from openai import OpenAI
import cv2

def predict(text):
    if not os.getenv("OPENAI_API_KEY"):
        # fallback to classical CV so the function still works
        return cv2.mean(text)[0]
    return OpenAI().responses.create(model="gpt", input=text).output_text
'''
    audit = deterministic_audit(spec, source)
    assert audit.adherent is False
    assert any("fallback" in item for item in audit.violations)


def test_deep_avenue_rejects_cv_substitute_when_torch_is_missing():
    spec = default_avenue(ApproachTier.SPECIALIZED_DEEP_MODEL, deep_resources())
    source = '''
import cv2

def predict(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()
'''
    audit = deterministic_audit(spec, source)
    assert audit.adherent is False
    assert any("deep-model" in item for item in audit.violations)


def test_deep_avenue_accepts_faithful_torch_code_that_fails_closed():
    spec = default_avenue(ApproachTier.SPECIALIZED_DEEP_MODEL, deep_resources())
    source = '''
# /// script
# dependencies = ["torch"]
# ///
import torch

_model = None

def predict(image):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this assigned implementation")
    return _model(image)
'''
    assert deterministic_audit(spec, source).adherent is True


def test_pi_runtime_candidate_must_declare_oauth_bound_placement():
    resources = ap.Resources(
        search=ap.SearchResources(pi_models=("openai-codex/gpt-5",)),
        runtime=ap.RuntimeResources(network=True),
        data=ap.DataPolicy(external_egress=True),
        confirmed=True,
    )
    spec = default_avenue(ApproachTier.SINGLE_MODEL_CALL, resources)
    source = '''
import subprocess

def predict(text):
    return subprocess.check_output(["pi", "--mode", "json", text])
'''
    assert deterministic_audit(spec, source).adherent is False
    declared = '''
# /// script
# [tool.ap]
# pi_runtime = true
# ///
import subprocess

def predict(text):
    return subprocess.check_output(["pi", "--mode", "json", text])
'''
    assert deterministic_audit(spec, declared).adherent is True


def test_api_capability_is_preflighted_instead_of_left_to_worker_fallback():
    resources = api_resources(available=False)
    # Construct directly because resource feasibility correctly excludes this
    # tier; a resumed/custom plan is still defended by preflight.
    spec = default_avenue(ApproachTier.SINGLE_MODEL_CALL, api_resources())
    missing = _missing_avenue_capabilities(spec, resources)
    assert any("provider access" in item for item in missing)


def test_pi_preflight_checks_the_assigned_registry_model_exactly(monkeypatch):
    monkeypatch.setattr(
        "autoprogramming.pi_backend.shutil.which", lambda _name: "/usr/bin/pi"
    )
    resources = ap.Resources(
        search=ap.SearchResources(
            pi_models=("openai-codex/available:max",),
            allow_package_installs=False,
            allow_model_downloads=False,
        ),
        runtime=ap.RuntimeResources(network=False),
        data=ap.DataPolicy(external_egress=True),
        confirmed=True,
    )
    spec = default_avenue(ApproachTier.SINGLE_MODEL_CALL, resources)
    from dataclasses import replace

    spec = replace(
        spec,
        required_capabilities=("pi-model:openai-codex/missing:max",),
    )
    assert any(
        "not in the confirmed" in item
        for item in _missing_avenue_capabilities(spec, resources)
    )


def test_complete_environment_failure_requires_every_run_to_fail():
    report = SimpleNamespace(
        n_rows=2,
        n_repeats=1,
        errors=[
            "row_0 repeat 1: RuntimeError: OPENAI API key is missing",
            "row_1 repeat 1: RuntimeError: OPENAI API key is missing",
        ],
    )
    assert _complete_environment_failure(report)
    report.errors.pop()
    assert _complete_environment_failure(report) == []


def test_independent_semantic_review_can_reject_a_subtle_substitution(monkeypatch):
    resources = api_resources()
    state = AvenueState(
        spec=default_avenue(ApproachTier.SINGLE_MODEL_CALL, resources)
    )
    backend = PiOrchestratorBackend(semantic_adherence_review=True)
    source = '''
from openai import OpenAI

def handmade_answer(text):
    return text.lower()

def predict(text, key=None):
    if key is None:
        return handmade_answer(text)
    return OpenAI(api_key=key).responses.create(model="gpt", input=text).output_text
'''
    response = {
        "adherent": False,
        "required_mechanisms_found": ["OpenAI call"],
        "forbidden_substitutions_found": ["handwritten local substitute"],
        "violations": ["missing credentials route to handwritten logic"],
        "repair_instructions": "Raise on missing credentials.",
    }
    monkeypatch.setattr(
        backend,
        "_charged_rpc_prompt",
        lambda *_args, **_kwargs: PiResult(json.dumps(response)),
    )
    audit = backend._audit_solution(
        SimpleNamespace(workspace=object()), state, resources, source
    )
    assert audit.adherent is False
    assert audit.reviewer == "pi"
    assert len(state.audits) == 2  # deterministic gate, then semantic reviewer


def test_noncompliant_solution_is_repaired_before_it_can_be_returned(tmp_path, monkeypatch):
    resources = api_resources()
    state = AvenueState(
        spec=default_avenue(ApproachTier.SINGLE_MODEL_CALL, resources)
    )
    root = tmp_path / "worker"
    root.mkdir()
    solution = root / "solution.py"
    solution.write_text('''
import os, cv2
from openai import OpenAI

def predict(text):
    if not os.getenv("OPENAI_API_KEY"):  # fallback
        return cv2.mean(text)[0]
    return OpenAI().responses.create(model="gpt", input=text).output_text
''')
    backend = PiOrchestratorBackend(semantic_adherence_review=False)
    harness = SimpleNamespace(workspace=SimpleNamespace())

    def repair(*_args, **_kwargs):
        solution.write_text('''
from openai import OpenAI

def predict(text):
    return OpenAI().responses.create(model="gpt", input=text).output_text
''')
        return PiResult("repaired")

    monkeypatch.setattr(backend, "_charged_worker_turn", repair)
    source = backend._ensure_adherent_solution(
        harness, state, resources, object(), root, initial_task="implement"
    )
    assert source is not None
    assert "cv2" not in source
    assert state.compliance_attempts >= 2
    assert any("rejected noncompliant" in note for note in state.notes)
