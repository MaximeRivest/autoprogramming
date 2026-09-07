"""Unit tests for autoprogramming.backend (prompt, NoOp/ClaudeCode backends)."""

from __future__ import annotations

import importlib.resources
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoprogramming.backend import (
    OPTIMIZER_PROMPT,
    AgentBackend,
    ClaudeCodeBackend,
    NoOpBackend,
    build_prompt,
    default_backend,
)
from autoprogramming.errors import RunnerError
from autoprogramming.schema import Schema


class Loud(str):
    """The input, uppercased, with an exclamation mark."""


def shout(text: str) -> Loud:
    """Uppercase the text."""


def make_harness(tmp_path, bootstrap=False, budget=None, split_json=True):
    """A duck-typed AgentHarness good enough for prompt building."""
    root = tmp_path / "shout_ap"
    (root / "data").mkdir(parents=True, exist_ok=True)
    split_path = root / "data" / "split.json"
    if split_json:
        split_path.write_text(
            json.dumps(
                {
                    "seed": 0,
                    "ratios": [0.6, 0.2, 0.2],
                    "counts": {"train": 18, "val": 6, "test": 6},
                    "data_sha": "abc",
                    "bootstrap": bootstrap,
                }
            )
        )
    workspace = SimpleNamespace(root=root, split_json=split_path)
    if budget is None:
        budget = {"dollars": 12.5, "eval_calls": 100, "minutes": None}
    return SimpleNamespace(
        workspace=workspace, schema=Schema.from_function(shout), budget=budget
    )


# ---------------------------------------------------------------- template


def test_optimizer_prompt_is_a_template_with_placeholders():
    for placeholder in (
        "$workspace",
        "$schema",
        "$data_sizes",
        "$bootstrap_note",
        "$budget",
        "$context",
    ):
        assert placeholder in OPTIMIZER_PROMPT


def test_optimizer_prompt_states_the_contract():
    for needle in (
        "propose_metric",
        "PEP 723",
        "predict",
        "prg.finalize",
        "prg.eval",
        "prg.run",
        "prg.compare",
        "prg.frontier",
        "prg.budget",
        "train",
        "val",
        "test",
        "deterministic = true",
        "AP_COST_DOLLARS",
        "artifacts",
    ):
        assert needle in OPTIMIZER_PROMPT, needle


def test_prompt_metric_signoff_comes_before_the_loop():
    assert OPTIMIZER_PROMPT.index("metric sign-off") < OPTIMIZER_PROMPT.index(
        "## The loop"
    )


def test_optimizer_prompt_embeds_the_packaged_skill_body(tmp_path):
    """The sync guarantee: editing SKILL.md changes the CLI prompt too."""
    skill_text = (
        importlib.resources.files("autoprogramming")
        / "skills" / "candidate-optimizer" / "SKILL.md"
    ).read_text(encoding="utf-8")
    frontmatter_end = skill_text.index("\n---\n", 4)
    body = skill_text[frontmatter_end + len("\n---\n"):].strip()

    # The whole skill body is embedded verbatim — no second copy to drift.
    assert body in OPTIMIZER_PROMPT
    # ...but the frontmatter (discovery metadata) is stripped.
    assert "name: candidate-optimizer" not in OPTIMIZER_PROMPT

    # And it survives substitution into the concrete prompt (so the body may
    # never contain a bare '$', which string.Template would choke on).
    prompt = build_prompt(make_harness(tmp_path), {"mode": "optimize"})
    assert body in prompt
    assert "Candidate optimizer for autoprogramming workspaces" in prompt


# ------------------------------------------------------------ build_prompt


def test_build_prompt_substitutes_workspace_facts(tmp_path):
    harness = make_harness(tmp_path)
    prompt = build_prompt(harness, {"mode": "optimize"})
    assert str(harness.workspace.root) in prompt
    assert "shout" in prompt
    assert "Uppercase the text." in prompt
    assert "18 train / 6 val / 6 test rows" in prompt
    assert '"dollars": 12.5' in prompt
    assert '"mode": "optimize"' in prompt
    assert "$workspace" not in prompt


def test_build_prompt_bootstrap_note(tmp_path):
    quiet = build_prompt(make_harness(tmp_path), {"mode": "optimize"})
    assert "BOOTSTRAP MODE" not in quiet
    assert "Full optimization mode." in quiet

    loud = build_prompt(
        make_harness(tmp_path, bootstrap=True), {"mode": "optimize"}
    )
    assert "BOOTSTRAP MODE" in loud
    assert "synthetic examples" in loud


def test_build_prompt_includes_distill_context(tmp_path):
    prompt = build_prompt(
        make_harness(tmp_path),
        {"mode": "distill", "target_model": "gpt-4.1-nano", "parent": "x_ap"},
    )
    assert '"mode": "distill"' in prompt
    assert '"target_model": "gpt-4.1-nano"' in prompt


def test_build_prompt_survives_missing_split_and_budget(tmp_path):
    harness = make_harness(tmp_path, split_json=False)
    harness.budget = object()  # not JSON-serializable
    prompt = build_prompt(harness, {"mode": "optimize"})
    assert "unknown" in prompt
    assert "not recorded yet" in prompt


# ---------------------------------------------------------------- backends


def test_noop_backend_prints_attach_instructions(tmp_path, capsys):
    harness = make_harness(tmp_path)
    assert NoOpBackend().run(harness, {"mode": "optimize"}) is None
    out = capsys.readouterr().out
    assert str(harness.workspace.root) in out
    assert "candidate-optimizer skill" in out
    assert ".agents/skills/" in out
    assert ".claude/skills/" in out
    assert "ap.attach" in out
    assert "propose_metric" in out
    assert "prg.eval" in out
    assert "finalize" in out


def test_backends_satisfy_the_protocol():
    assert isinstance(NoOpBackend(), AgentBackend)
    assert isinstance(ClaudeCodeBackend(), AgentBackend)


def test_default_backend_prefers_claude_when_present(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: "/usr/bin/claude")
    assert isinstance(default_backend(), ClaudeCodeBackend)


def test_default_backend_falls_back_to_noop(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda cmd: None)
    assert isinstance(default_backend(), NoOpBackend)


def test_claude_backend_missing_binary_refused(tmp_path):
    backend = ClaudeCodeBackend(command=("ap-test-no-such-binary-xyz",))
    with pytest.raises(RunnerError, match="NoOpBackend"):
        backend.run(make_harness(tmp_path), {"mode": "optimize"})


def test_claude_backend_invokes_command_with_prompt(tmp_path):
    record = tmp_path / "record.json"
    exe = tmp_path / "fakeclaude"
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        f"dest = {str(record)!r}\n"
        "with open(dest, 'w') as fh:\n"
        "    json.dump({'argv': sys.argv[1:], 'cwd': os.getcwd()}, fh)\n"
    )
    exe.chmod(0o755)

    harness = make_harness(tmp_path)
    ClaudeCodeBackend(command=(str(exe),)).run(harness, {"mode": "optimize"})

    rec = json.loads(record.read_text())
    assert rec["argv"][0] == "-p"
    prompt = rec["argv"][1]
    assert str(harness.workspace.root) in prompt
    assert "propose_metric" in prompt
    assert Path(rec["cwd"]).resolve() == harness.workspace.root.resolve()
