"""Tests for autoprogramming.program (and the lazy package __init__).

Tests that need sibling modules still being written (data, workspace,
harness, ...) are gated with importorskip and run in the integration phase.
"""

from __future__ import annotations

import json
import shutil as _shutil
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import autoprogramming as ap
from autoprogramming.budget import Budget
from autoprogramming.errors import (
    BudgetError,
    BudgetExceededError,
    DataDisciplineError,
    NotOptimizedError,
    SchemaError,
    WorkspaceError,
)
from autoprogramming.program import Program, program


class Loud(str):
    """The input, uppercased, with an exclamation mark."""


def make_shout() -> Program:
    """A fresh Program per test — workspace binding is stateful."""

    def shout(text: str) -> Loud:
        """Uppercase the text."""

    return program(shout)


def rows_for(n: int) -> list[dict]:
    return [{"text": f"word {i}", "Loud": f"WORD {i}!"} for i in range(n)]


CANDIDATE_SRC = (
    "# /// script\n"
    "# dependencies = []\n"
    "#\n"
    "# [tool.ap]\n"
    "# deterministic = true\n"
    "# ///\n"
    "def predict(text):\n"
    '    return text.upper() + "!"\n'
)

METRIC_SRC = (
    "def metric(predicted, expected):\n"
    "    return 1.0 if predicted == expected else 0.0\n"
)


class ScriptedBackend:
    """Test double for AgentBackend: records contexts, runs a script."""

    def __init__(self, script=None):
        self.script = script
        self.contexts = []
        self.harnesses = []

    def run(self, harness, context):
        self.contexts.append(context)
        self.harnesses.append(harness)
        if self.script is not None:
            self.script(harness, context)


def drive_full(harness, context):
    """Scripted agent: metric sign-off, one candidate, train + val evals."""
    assert harness.propose_metric(METRIC_SRC, [("A!", "A!"), ("A!", "B!")]) is True
    cand = harness.new_candidate(source=CANDIDATE_SRC)
    harness.eval(cand.name, split="train")
    harness.eval(cand.name)


def require_siblings():
    for mod in ("data", "workspace", "guards", "harness", "logs"):
        pytest.importorskip(
            f"autoprogramming.{mod}",
            reason=f"sibling module autoprogramming.{mod} not written yet",
        )


# ----------------------------------------------------- standalone: decorator


def test_decorator_copies_function_metadata():
    prg = make_shout()
    assert isinstance(prg, Program)
    assert prg.__name__ == "shout"
    assert prg.__doc__ == "Uppercase the text."
    assert prg.__wrapped__ is prg._fn
    assert prg.schema.input_names == ("text",)
    assert prg.schema.output_names == ("Loud",)
    assert prg.workspace is None


def test_schema_errors_surface_at_decoration_time():
    def bad(x) -> Loud:
        """Missing input annotation."""

    with pytest.raises(SchemaError):
        program(bad)
    with pytest.raises(SchemaError):
        program(lambda x: x)


def test_repr_mentions_schema_and_state():
    text = repr(make_shout())
    assert "shout" in text and "Loud" in text and "unoptimized" in text


# ---------------------------------------------------- standalone: refusals


def test_call_without_workspace_refused():
    prg = make_shout()
    with pytest.raises(NotOptimizedError, match="optimize"):
        prg("hello")


def test_optimize_requires_explicit_budget():
    prg = make_shout()
    with pytest.raises(BudgetError, match="no default"):
        prg.optimize(rows_for(30))
    with pytest.raises(BudgetError):
        prg.optimize(rows_for(30), budget="20 dollars")


def test_optimize_on_raw_logs_refused_with_reasoning():
    prg = make_shout()
    with pytest.raises(DataDisciplineError) as excinfo:
        prg.optimize("logs", budget=Budget(dollars=1))
    message = str(excinfo.value)
    assert "reinforces your own errors" in message
    assert "review_logs" in message
    assert "logs:reviewed" in message
    assert "distill" in message


def test_optimize_logs_reviewed_needs_a_workspace():
    prg = make_shout()
    with pytest.raises(NotOptimizedError, match="workspace"):
        prg.optimize("logs:reviewed", budget=Budget(dollars=1))


def test_save_without_workspace_refused():
    with pytest.raises(NotOptimizedError, match="nothing to save"):
        make_shout().save("anywhere_ap")


def test_review_logs_without_workspace_refused():
    with pytest.raises(NotOptimizedError, match="logs"):
        make_shout().review_logs()


def test_distill_without_workspace_refused():
    with pytest.raises(NotOptimizedError, match="optimize"):
        make_shout().distill(model="tiny", budget=Budget(dollars=1))


def test_distill_requires_explicit_budget():
    prg = make_shout()
    prg._workspace = SimpleNamespace(root=Path("shout_ap"))
    with pytest.raises(BudgetError, match="willing to spend"):
        prg.distill(model="tiny")


def test_distill_rejects_non_log_data():
    prg = make_shout()
    prg._workspace = SimpleNamespace(root=Path("shout_ap"))
    with pytest.raises(DataDisciplineError, match="imitates"):
        prg.distill(model="tiny", data="train.csv", budget=Budget(dollars=1))


# ----------------------------------------------------- standalone: binding


def test_bind_inputs_like_a_normal_function():
    prg = make_shout()
    assert prg._bind_inputs(("hi",), {}) == {"text": "hi"}
    assert prg._bind_inputs((), {"text": "hi"}) == {"text": "hi"}
    with pytest.raises(TypeError, match="shout"):
        prg._bind_inputs((), {})
    with pytest.raises(TypeError):
        prg._bind_inputs(("a", "b"), {})
    with pytest.raises(TypeError):
        prg._bind_inputs(("a",), {"text": "b"})
    with pytest.raises(TypeError):
        prg._bind_inputs(("a",), {"lang": "fr"})


def test_logging_toggles_return_self():
    prg = make_shout()
    assert prg.enable_logging() is prg
    assert prg._logging is True
    assert prg.disable_logging() is prg
    assert prg._logging is False


# ------------------------------------------------------ standalone: package


def test_package_exports():
    assert ap.program is program
    assert ap.Program is Program
    assert ap.Budget is Budget
    assert isinstance(ap.__version__, str) and ap.__version__
    assert ap.NotOptimizedError is NotOptimizedError
    assert "attach" in dir(ap)
    with pytest.raises(AttributeError):
        ap.definitely_not_a_name


def test_package_attach_is_harness_attach():
    harness_mod = pytest.importorskip("autoprogramming.harness")
    assert ap.attach is harness_mod.attach


# ------------------------------------------------------------- full stack


@pytest.fixture(scope="module")
def optimized(tmp_path_factory):
    """One fully optimized + finalized workspace shared by read-only tests."""
    mp = pytest.MonkeyPatch()
    mp.setenv("AP_AUTO_APPROVE_METRIC", "1")
    try:
        require_siblings()
        root = tmp_path_factory.mktemp("full") / "shout_ap"
        prg = make_shout()
        backend = ScriptedBackend(drive_full)
        report = prg.optimize(
            rows_for(30),
            budget=Budget(eval_calls=1000),
            workspace=root,
            backend=backend,
        )
        return SimpleNamespace(prg=prg, root=root, report=report, backend=backend)
    finally:
        mp.undo()


def test_optimize_full_run_finalizes_and_activates(optimized):
    report = optimized.report
    assert report is not None
    assert report.activated == "candidate_0"
    assert optimized.backend.contexts == [{"mode": "optimize"}]

    active = json.loads((optimized.root / "active.json").read_text())
    assert active["program"] == "shout"
    assert active["active"] == "candidate_0"
    assert active["finalized"] is True
    assert (optimized.root / "final_report.json").exists()
    assert (optimized.root / "candidates" / "candidate_0.py").exists()


def test_call_runs_active_candidate(optimized):
    result = optimized.prg("hello world")
    assert result == "HELLO WORLD!"
    assert isinstance(result, Loud)
    assert optimized.prg(text="ok") == "OK!"


def test_call_binds_like_a_normal_function(optimized):
    with pytest.raises(TypeError):
        optimized.prg()
    with pytest.raises(TypeError):
        optimized.prg("a", "b")
    with pytest.raises(TypeError):
        optimized.prg("a", text="b")
    with pytest.raises(TypeError):
        optimized.prg("a", lang="fr")


def test_logging_writes_readme_jsonl_format(optimized):
    prg = optimized.prg
    prg.enable_logging()
    try:
        prg("log me")
    finally:
        prg.disable_logging()

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_file = optimized.root / "logs" / f"{day}.jsonl"
    assert log_file.exists()
    entry = json.loads(log_file.read_text().splitlines()[-1])
    assert entry["inputs"] == {"text": "log me"}
    assert entry["outputs"] == {"Loud": "LOG ME!"}
    assert entry["candidate"] == "candidate_0"
    assert entry["n_repeat"] == 1
    assert "timestamp" in entry


def test_use_binds_existing_workspace(optimized):
    other = make_shout().use(optimized.root)
    assert other.workspace is not None
    assert other("reuse me") == "REUSE ME!"


def test_use_refuses_program_name_mismatch(optimized):
    def whisper(text: str) -> Loud:
        """Lowercase the text."""

    with pytest.raises(WorkspaceError, match="whisper"):
        program(whisper).use(optimized.root)


def test_optimize_same_data_on_finalized_workspace_returns_report(optimized):
    prg = make_shout()
    backend = ScriptedBackend()
    report = prg.optimize(
        rows_for(30),
        budget=Budget(eval_calls=10),
        workspace=optimized.root,
        backend=backend,
    )
    assert report is not None
    assert report.activated == "candidate_0"
    assert report.entries
    assert len(backend.contexts) == 1


def test_optimize_refuses_data_sha_mismatch(optimized):
    prg = make_shout()
    with pytest.raises(DataDisciplineError, match="split once"):
        prg.optimize(
            rows_for(31),
            budget=Budget(eval_calls=10),
            workspace=optimized.root,
            backend=ScriptedBackend(),
        )


def test_optimize_refuses_workspace_of_other_program(optimized, tmp_path):
    def whisper(text: str) -> Loud:
        """Lowercase the text."""

    with pytest.raises(WorkspaceError, match="shout"):
        program(whisper).optimize(
            rows_for(30),
            budget=Budget(eval_calls=10),
            workspace=optimized.root,
            backend=ScriptedBackend(),
        )


def test_save_moves_and_repoints(optimized, tmp_path):
    src = tmp_path / "moving_src_ap"
    _shutil.copytree(optimized.root, src)
    prg = make_shout().use(src)

    target = tmp_path / "kept_ap"
    returned = prg.save(target)
    assert returned == target
    assert not src.exists()
    assert prg.workspace.root == target
    assert prg("after move") == "AFTER MOVE!"

    existing = tmp_path / "occupied_ap"
    existing.mkdir()
    with pytest.raises(WorkspaceError, match="exists"):
        prg.save(existing)
    with pytest.raises(WorkspaceError, match="identifier"):
        prg.save(tmp_path / "bad-name-ap")
    assert prg("still fine") == "STILL FINE!"


def test_bootstrap_mode_prints_notice(tmp_path, capsys):
    require_siblings()
    prg = make_shout()
    report = prg.optimize(
        rows_for(10),
        budget=Budget(eval_calls=10),
        workspace=tmp_path / "tiny_ap",
        backend=ScriptedBackend(),
    )
    out = capsys.readouterr().out
    assert "bootstrap" in out.lower()
    assert "synthetic" in out.lower()
    assert report is None
    split = json.loads((tmp_path / "tiny_ap" / "data" / "split.json").read_text())
    assert split["bootstrap"] is True


def test_nothing_scored_prints_attach_and_returns_none(tmp_path, capsys):
    require_siblings()
    report = make_shout().optimize(
        rows_for(30),
        budget=Budget(eval_calls=10),
        workspace=tmp_path / "manual_ap",
        backend=ScriptedBackend(),
    )
    assert report is None
    out = capsys.readouterr().out
    assert "attach" in out
    assert str(tmp_path / "manual_ap") in out


def test_default_workspace_path_is_name_ap(tmp_path, monkeypatch, capsys):
    require_siblings()
    monkeypatch.chdir(tmp_path)
    report = make_shout().optimize(
        rows_for(30), budget=Budget(eval_calls=10), backend=ScriptedBackend()
    )
    assert report is None
    assert (tmp_path / "shout_ap" / "data" / "split.json").exists()


def test_budget_exceeded_in_backend_still_finalizes(tmp_path, monkeypatch):
    require_siblings()
    monkeypatch.setenv("AP_AUTO_APPROVE_METRIC", "1")

    def drive_then_blow_budget(harness, context):
        drive_full(harness, context)
        raise BudgetExceededError("simulated: dollars limit 1 reached")

    report = make_shout().optimize(
        rows_for(30),
        budget=Budget(eval_calls=1000),
        workspace=tmp_path / "spent_ap",
        backend=ScriptedBackend(drive_then_blow_budget),
    )
    assert report is not None
    assert report.activated == "candidate_0"


def test_live_pi_session_uses_human_facing_host_orchestrator(
    tmp_path, monkeypatch, capsys
):
    require_siblings()
    import importlib
    program_mod = importlib.import_module("autoprogramming.program")
    from autoprogramming.pi_backend import PiOrchestratorBackend

    monkeypatch.setenv("PI_SESSION_ID", "host-session")
    monkeypatch.setenv("PI_PROVIDER", "openai-codex")
    monkeypatch.setenv("PI_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("PI_REASONING_LEVEL", "max")
    monkeypatch.setattr(program_mod.shutil, "which", lambda name: "/usr/bin/pi")
    monkeypatch.setattr(
        PiOrchestratorBackend,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("must not spawn a second strategy orchestrator")
        ),
    )
    resources = ap.Resources(
        search=ap.SearchResources(
            allow_package_installs=False,
            allow_model_downloads=False,
        ),
        runtime=ap.RuntimeResources(network=False),
        data=ap.DataPolicy(external_egress=True),
        confirmed=True,
    )
    prg = make_shout()
    assert prg.optimize(
        rows_for(5),
        budget=Budget(eval_calls=10),
        workspace=tmp_path / "host_ap",
        resources=resources,
    ) is None
    output = capsys.readouterr().out
    assert "current human-facing Pi session" in output
    recorded = json.loads(prg.workspace.resources_json.read_text())
    assert recorded["search"]["pi_models"][0] == (
        "openai-codex/gpt-5.6-sol:max"
    )


def test_budget_exhaustion_does_not_finalize_incomplete_pi_portfolio(
    tmp_path, monkeypatch
):
    require_siblings()
    monkeypatch.setenv("AP_AUTO_APPROVE_METRIC", "1")

    def drive_with_pending_journal(harness, context):
        from autoprogramming.portfolio import Portfolio

        drive_full(harness, context)
        resources = ap.Resources(
            search=ap.SearchResources(
                allow_package_installs=False, allow_model_downloads=False
            ),
            runtime=ap.RuntimeResources(network=False),
            data=ap.DataPolicy(external_egress=False),
            confirmed=True,
        )
        portfolio = Portfolio.create(resources, [])
        portfolio.avenues[0].begin_candidate("candidate_999")
        harness.workspace.portfolio_json.parent.mkdir(parents=True, exist_ok=True)
        portfolio.write(harness.workspace.portfolio_json)
        raise BudgetExceededError("simulated exhausted Pi portfolio")

    prg = make_shout()
    report = prg.optimize(
        rows_for(30),
        budget=Budget(eval_calls=1000),
        workspace=tmp_path / "pending_ap",
        backend=ScriptedBackend(drive_with_pending_journal),
    )
    assert report is None
    assert prg.workspace.active["finalized"] is False


def test_optimize_logs_reviewed_flow(tmp_path, monkeypatch, capsys):
    require_siblings()
    ws_root = tmp_path / "shout_ap"
    prg = make_shout()
    prg.optimize(
        rows_for(30),
        budget=Budget(eval_calls=10),
        workspace=ws_root,
        backend=ScriptedBackend(),
    )

    with pytest.raises(DataDisciplineError, match="review"):
        prg.optimize("logs:reviewed", budget=Budget(eval_calls=10))

    now = datetime.now(timezone.utc).isoformat()
    reviewed = [
        {
            "inputs": {"text": f"redo {i}"},
            "outputs": {"Loud": f"REDO {i}!"},
            "verdict": "corrected",
            "source_sha": f"sha{i}",
            "reviewed_at": now,
        }
        for i in range(6)
    ]
    reviewed.append(
        {
            "inputs": {"text": "junk"},
            "outputs": {"Loud": "JUNK!"},
            "verdict": "rejected",
            "source_sha": "sha-r",
            "reviewed_at": now,
        }
    )
    logs_dir = ws_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    (logs_dir / "reviewed.jsonl").write_text(
        "\n".join(json.dumps(e) for e in reviewed) + "\n"
    )

    reopt_root = tmp_path / "shout_v2_ap"
    report = prg.optimize(
        "logs:reviewed",
        budget=Budget(eval_calls=10),
        workspace=reopt_root,
        backend=ScriptedBackend(),
    )
    assert report is None
    split = json.loads((reopt_root / "data" / "split.json").read_text())
    counts = split["counts"]
    assert counts["train"] + counts["val"] + counts["test"] == 6


def test_distill_uses_unreviewed_logs_on_purpose(optimized, tmp_path):
    prg = optimized.prg
    prg.enable_logging()
    try:
        for i in range(6):
            prg(f"distill me {i}")
    finally:
        prg.disable_logging()

    backend = ScriptedBackend()
    out_root = tmp_path / "shout_distilled_ap"
    report = prg.distill(
        model="tiny-model",
        budget=Budget(eval_calls=10),
        output=out_root,
        backend=backend,
    )
    assert report is None
    assert backend.contexts == [
        {
            "mode": "distill",
            "target_model": "tiny-model",
            "parent": str(optimized.root),
        }
    ]
    assert (out_root / "data" / "split.json").exists()
    train_csv = (out_root / "data" / "train.csv").read_text()
    assert "DISTILL ME" in train_csv or "LOG ME" in train_csv or "!" in train_csv
    assert prg.workspace.root == optimized.root


def test_distill_refuses_without_logs(tmp_path, monkeypatch):
    require_siblings()
    prg = make_shout()
    prg.optimize(
        rows_for(30),
        budget=Budget(eval_calls=10),
        workspace=tmp_path / "nolog_ap",
        backend=ScriptedBackend(),
    )
    with pytest.raises(DataDisciplineError, match="enable_logging"):
        prg.distill(model="tiny", budget=Budget(eval_calls=10))


def test_review_logs_delegates_to_logs_module(monkeypatch):
    logs_mod = pytest.importorskip("autoprogramming.logs")
    seen = {}

    def fake_review(ws, sample=None):
        seen["ws"] = ws
        seen["sample"] = sample
        return {"reviewed": 2, "accepted": 1, "corrected": 1, "rejected": 0}

    monkeypatch.setattr(logs_mod, "review_logs", fake_review)
    prg = make_shout()
    prg._workspace = SimpleNamespace(root=Path("shout_ap"))
    result = prg.review_logs(sample=7)
    assert result["reviewed"] == 2
    assert seen["ws"] is prg._workspace
    assert seen["sample"] == 7


def test_call_refuses_when_nothing_activated(tmp_path):
    require_siblings()
    prg = make_shout()
    prg.optimize(
        rows_for(30),
        budget=Budget(eval_calls=10),
        workspace=tmp_path / "idle_ap",
        backend=ScriptedBackend(),
    )
    with pytest.raises(NotOptimizedError, match="active"):
        prg("hello")


def test_readme_quickstart_save_to_own_path_is_noop(tmp_path, monkeypatch):
    """README front page: optimize() with the default workspace, then
    save("<name>_ap") — the target IS the workspace, so save must not refuse."""
    require_siblings()
    monkeypatch.chdir(tmp_path)
    prg = make_shout()
    prg.optimize(rows_for(30), budget=Budget(eval_calls=10), backend=ScriptedBackend())
    root = (tmp_path / "shout_ap").resolve()
    assert prg.workspace.root == root

    assert prg.save("shout_ap") == Path("shout_ap")
    assert prg.workspace.root == root
    assert (root / "active.json").exists()

    assert prg.save(root) == root
    assert prg.workspace.root == root


def test_optimize_logs_reviewed_defaults_to_fresh_reopt_workspace(tmp_path, capsys):
    """README §Improve from production: the re-optimize call as written —
    optimize(data="logs:reviewed", budget=...) with no workspace argument."""
    require_siblings()
    ws_root = tmp_path / "shout_ap"
    prg = make_shout()
    prg.optimize(
        rows_for(30),
        budget=Budget(eval_calls=10),
        workspace=ws_root,
        backend=ScriptedBackend(),
    )
    original_split = (ws_root / "data" / "split.json").read_text()

    now = datetime.now(timezone.utc).isoformat()
    reviewed = [
        {
            "inputs": {"text": f"redo {i}"},
            "outputs": {"Loud": f"REDO {i}!"},
            "verdict": "corrected",
            "source_sha": f"sha{i}",
            "reviewed_at": now,
        }
        for i in range(6)
    ]
    logs_dir = ws_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    (logs_dir / "reviewed.jsonl").write_text(
        "\n".join(json.dumps(e) for e in reviewed) + "\n"
    )

    report = prg.optimize(
        "logs:reviewed", budget=Budget(eval_calls=10), backend=ScriptedBackend()
    )
    assert report is None
    out = capsys.readouterr().out
    assert "re-optimizing" in out

    reopt_root = (tmp_path / "shout_reopt_ap").resolve()
    assert prg.workspace.root == reopt_root
    split = json.loads((reopt_root / "data" / "split.json").read_text())
    counts = split["counts"]
    assert counts["train"] + counts["val"] + counts["test"] == 6
    # the original workspace's split stays fixed
    assert (ws_root / "data" / "split.json").read_text() == original_split

    # repeating the same call from the original workspace resumes the same
    # re-optimization workspace instead of piling up directories
    again = make_shout().use(ws_root)
    again.optimize(
        "logs:reviewed", budget=Budget(eval_calls=10), backend=ScriptedBackend()
    )
    assert again.workspace.root == reopt_root
    assert not (tmp_path / "shout_reopt2_ap").exists()
