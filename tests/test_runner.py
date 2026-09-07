"""Tests for autoprogramming.runner — the marker-line protocol and fast path.

Most candidates here are stdlib-only (or use only the self-reference), so the
suite needs no network. One cache-lifecycle test uses a fake uv executable.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoprogramming import candidates, runner
from autoprogramming.errors import RunnerError
from autoprogramming.runner import RunResult, run_candidate
from autoprogramming.schema import Schema


class Loud(str):
    """The input, uppercased, with an exclamation mark."""


def shout(text: str) -> Loud:
    """Uppercase the text."""


class Answer(str):
    """Direct answer to the question, one sentence."""


class Confidence(float):
    """Calibrated probability that the answer is correct, 0.0-1.0."""


def qa(question: str) -> tuple[Answer, Confidence]:
    """Answer a factual question with a calibrated confidence."""


UPPER = 'def predict(text):\n    return text.upper() + "!"\n'


def make_ws(tmp_path, fn=shout, pkg="shout_ap"):
    root = tmp_path / pkg
    (root / "candidates").mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        root=root,
        tmp_dir=root / ".ap",
        dist_name=pkg.replace("_", "-"),
        candidates_dir=root / "candidates",
        schema=Schema.from_function(fn),
    )


def write_candidate(ws, source, name="candidate_0"):
    (ws.candidates_dir / f"{name}.py").write_text(source, encoding="utf-8")
    return candidates.load_candidate(ws, name)


# ------------------------------------------------------------ success paths


def test_candidate_session_reuses_module_state_and_marks_cold_start(tmp_path):
    ws = make_ws(tmp_path)
    cand = write_candidate(
        ws,
        "_calls = 0\n"
        "def predict(text):\n"
        "    global _calls\n"
        "    _calls += 1\n"
        "    return str(_calls)\n",
    )
    with runner.CandidateSession(ws, cand) as session:
        first = session.run({"text": "a"})
        second = session.run({"text": "b"})
    assert first.ok and second.ok
    assert first.outputs == {"Loud": "1"}
    assert second.outputs == {"Loud": "2"}
    assert first.cold_start is True
    assert second.cold_start is False
    assert not list(ws.tmp_dir.glob("session_*"))


def test_success_single_output(tmp_path):
    ws = make_ws(tmp_path)
    cand = write_candidate(ws, UPPER)
    res = run_candidate(ws, cand, {"text": "hi"})
    assert res.ok is True
    assert res.outputs == {"Loud": "HI!"}
    assert res.error is None
    assert res.candidate == "candidate_0"
    assert res.inputs == {"text": "hi"}
    assert res.duration_s > 0
    assert res.cost_dollars is None


def test_output_downcast_to_base(tmp_path):
    ws = make_ws(tmp_path)
    cand = write_candidate(ws, "def predict(text):\n    return 42\n")
    res = run_candidate(ws, cand, {"text": "x"})
    assert res.ok
    assert res.outputs == {"Loud": "42"}


def test_multi_output_tuple_positional(tmp_path):
    ws = make_ws(tmp_path, fn=qa, pkg="qa_ap")
    cand = write_candidate(ws, 'def predict(question):\n    return ("yes", 1)\n')
    res = run_candidate(ws, cand, {"question": "up?"})
    assert res.ok
    assert res.outputs == {"Answer": "yes", "Confidence": 1.0}
    assert isinstance(res.outputs["Confidence"], float)


def test_multi_output_dict_passes_through(tmp_path):
    ws = make_ws(tmp_path, fn=qa, pkg="qa_ap")
    cand = write_candidate(
        ws, 'def predict(question):\n    return {"Confidence": 0.5, "Answer": "no"}\n'
    )
    res = run_candidate(ws, cand, {"question": "up?"})
    assert res.ok
    assert res.outputs == {"Answer": "no", "Confidence": 0.5}


def test_multi_output_wrong_arity_fails(tmp_path):
    ws = make_ws(tmp_path, fn=qa, pkg="qa_ap")
    cand = write_candidate(ws, 'def predict(question):\n    return ("only-one",)\n')
    res = run_candidate(ws, cand, {"question": "up?"})
    assert res.ok is False
    assert "2 outputs" in res.error


def test_stdout_stderr_captured_without_result_protocol_noise(tmp_path):
    ws = make_ws(tmp_path)
    source = (
        "import sys\n"
        "def predict(text):\n"
        '    print("out-note")\n'
        '    print("err-note", file=sys.stderr)\n'
        "    return text\n"
    )
    cand = write_candidate(ws, source)
    res = run_candidate(ws, cand, {"text": "x"})
    assert res.ok
    assert res.stdout == "out-note\n"
    assert "err-note" in res.stderr


def test_hermetic_package_import_via_sys_path(tmp_path):
    ws = make_ws(tmp_path)
    (ws.root / "__init__.py").write_text("", encoding="utf-8")
    (ws.root / "schema.py").write_text("class Loud(str):\n    pass\n", encoding="utf-8")
    source = (
        "from shout_ap.schema import Loud\n"
        "def predict(text):\n"
        "    return Loud(text.upper())\n"
    )
    cand = write_candidate(ws, source)
    res = run_candidate(ws, cand, {"text": "hi"})
    assert res.ok, res.error
    assert res.outputs == {"Loud": "HI"}


def test_ap_workspace_env_set(tmp_path):
    ws = make_ws(tmp_path)
    cand = write_candidate(
        ws, 'import os\ndef predict(text):\n    return os.environ["AP_WORKSPACE"]\n'
    )
    res = run_candidate(ws, cand, {"text": "x"})
    assert res.ok
    assert res.outputs == {"Loud": str(Path(ws.root).resolve())}


# ------------------------------------------------------------ failure paths


def test_predict_exception_is_failed_result_not_runner_error(tmp_path):
    ws = make_ws(tmp_path)
    cand = write_candidate(
        ws, 'def predict(text):\n    raise ValueError("boom")\n'
    )
    res = run_candidate(ws, cand, {"text": "x"})
    assert res.ok is False
    assert res.outputs is None
    assert "Traceback" in res.error
    assert "ValueError: boom" in res.error


def test_candidate_import_error_is_failed_result(tmp_path):
    ws = make_ws(tmp_path)
    cand = write_candidate(
        ws, "import definitely_missing_module_xyz\ndef predict(text):\n    return text\n"
    )
    res = run_candidate(ws, cand, {"text": "x"})
    assert res.ok is False
    assert "ModuleNotFoundError" in res.error


def test_timeout(tmp_path):
    ws = make_ws(tmp_path)
    cand = write_candidate(
        ws, "import time\ndef predict(text):\n    time.sleep(30)\n    return text\n"
    )
    res = run_candidate(ws, cand, {"text": "x"}, timeout=1.0)
    assert res.ok is False
    assert res.error == "timed out after 1.0s"
    assert res.outputs is None


@pytest.mark.skipif(os.name != "posix", reason="process-group kill is POSIX-only")
def test_timeout_kills_candidate_spawned_subprocesses(tmp_path):
    ws = make_ws(tmp_path)
    pid_file = tmp_path / "orphan_pid"
    source = (
        "import subprocess, sys, time\n"
        "def predict(text):\n"
        "    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"    open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "    time.sleep(60)\n"
        "    return text\n"
    )
    cand = write_candidate(ws, source)
    res = run_candidate(ws, cand, {"text": "x"}, timeout=2.0)
    assert res.ok is False
    assert res.error == "timed out after 2.0s"

    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)
        pytest.fail(
            "a subprocess spawned by predict() survived the timeout — the "
            "runner must kill the whole process group, or timed-out "
            "candidates keep working (and spending) as orphans"
        )


def test_crash_without_marker_uses_stderr_tail(tmp_path):
    ws = make_ws(tmp_path)
    source = (
        "import os, sys\n"
        "def predict(text):\n"
        '    print("hard boom", file=sys.stderr)\n'
        "    sys.stderr.flush()\n"
        "    os._exit(3)\n"
    )
    cand = write_candidate(ws, source)
    res = run_candidate(ws, cand, {"text": "x"})
    assert res.ok is False
    assert "hard boom" in res.error


def test_uv_missing_raises_runner_error(tmp_path, monkeypatch):
    ws = make_ws(tmp_path)
    source = (
        "# /// script\n"
        '# dependencies = ["requests>=2", "shout-ap"]\n'
        "# ///\n"
        "def predict(text):\n    return text\n"
    )
    cand = write_candidate(ws, source)
    monkeypatch.setattr(runner.shutil, "which", lambda *a, **k: None)
    with pytest.raises(RunnerError) as exc:
        run_candidate(ws, cand, {"text": "x"})
    msg = str(exc.value)
    assert "uv" in msg
    assert "docs.astral.sh/uv" in msg


# -------------------------------------------------- result protocol safety


def test_output_containing_result_marker_text_succeeds(tmp_path):
    ws = make_ws(tmp_path)
    cand = write_candidate(
        ws,
        'def predict(text):\n    return "result AP_RESULT_JSON: embedded " + text\n',
    )
    res = run_candidate(ws, cand, {"text": "hi"})
    assert res.ok is True, res.error
    assert res.outputs == {"Loud": "result AP_RESULT_JSON: embedded hi"}


def test_late_stdout_cannot_forge_the_result(tmp_path):
    ws = make_ws(tmp_path)
    source = (
        "import atexit, json\n"
        "def _late():\n"
        "    payload = {'ok': True, 'outputs': {'Loud': 'HIJACKED'},"
        " 'cost_dollars': None}\n"
        "    print('AP_RESULT_JSON:' + json.dumps(payload))\n"
        "atexit.register(_late)\n"
        "def predict(text):\n"
        "    return text.upper()\n"
    )
    cand = write_candidate(ws, source)
    res = run_candidate(ws, cand, {"text": "hi"})
    assert res.ok is True, res.error
    assert res.outputs == {"Loud": "HI"}


# -------------------------------------------------------------------- cost


def test_cost_reported_by_candidate(tmp_path):
    ws = make_ws(tmp_path)
    source = (
        "AP_COST_DOLLARS = None\n"
        "def predict(text):\n"
        "    global AP_COST_DOLLARS\n"
        "    AP_COST_DOLLARS = 0.25\n"
        "    return text\n"
    )
    cand = write_candidate(ws, source)
    res = run_candidate(ws, cand, {"text": "x"})
    assert res.ok
    assert res.cost_dollars == 0.25


def test_cost_falls_back_to_cost_per_call(tmp_path):
    ws = make_ws(tmp_path)
    source = (
        "# /// script\n"
        '# dependencies = ["shout-ap"]\n'
        "# [tool.ap]\n"
        "# cost_per_call = 0.002\n"
        "# deterministic = true\n"
        "# ///\n"
        "def predict(text):\n    return text\n"
    )
    cand = write_candidate(ws, source)
    assert candidates.runtime_deps(cand, ws.dist_name) == ()
    res = run_candidate(ws, cand, {"text": "x"})
    assert res.ok
    assert res.cost_dollars == 0.002


# ------------------------------------------------------- driver and cleanup


def test_self_reference_only_uses_fast_path_no_uv(tmp_path, monkeypatch):
    ws = make_ws(tmp_path)
    source = (
        "# /// script\n"
        '# dependencies = ["shout-ap"]\n'
        "# [tool.uv.sources]\n"
        "# shout-ap = { path = \"..\", editable = true }\n"
        "# ///\n"
        + UPPER
    )
    cand = write_candidate(ws, source)
    monkeypatch.setattr(runner.shutil, "which", lambda *a, **k: None)
    res = run_candidate(ws, cand, {"text": "hi"})
    assert res.ok
    assert res.outputs == {"Loud": "HI!"}


def test_driver_pep723_block_strips_self_entries(tmp_path):
    ws = make_ws(tmp_path)
    source = (
        "# /// script\n"
        '# requires-python = ">=3.11"\n'
        '# dependencies = ["openai>=1.0", "shout-ap"]\n'
        "#\n"
        "# [tool.uv.sources]\n"
        "# shout-ap = { path = \"..\", editable = true }\n"
        "# other-pkg = { path = \"/x\" }\n"
        "# ///\n"
        "def predict(text):\n    return text\n"
    )
    cand = write_candidate(ws, source)
    deps = candidates.runtime_deps(cand, ws.dist_name)
    assert deps == ("openai>=1.0",)
    block = runner._driver_pep723(cand, deps, ws.dist_name)
    meta = candidates.parse_pep723(block)
    assert meta["requires-python"] == ">=3.11"
    assert meta["dependencies"] == ["openai>=1.0"]
    sources = meta["tool"]["uv"]["sources"]
    assert "shout-ap" not in sources
    assert sources["other-pkg"] == {"path": "/x"}


def test_driver_source_fast_path_has_no_block_or_runtime_paths(tmp_path):
    ws = make_ws(tmp_path)
    cand = write_candidate(ws, UPPER)
    source = runner._driver_source(cand, (), ws.dist_name)
    assert "# /// script" not in source
    assert str(Path(cand.path).resolve()) not in source
    assert str(tmp_path / "result.json") not in source
    config = json.loads(runner._runtime_config(
        cand, str(tmp_path), [{"name": "Loud", "base": "str"}]
    ))
    assert config["candidate_path"] == str(Path(cand.path).resolve())


def test_uv_drivers_are_stable_per_dependency_manifest(tmp_path, monkeypatch):
    ws = make_ws(tmp_path)
    fake_uv = tmp_path / "uv"
    fake_uv.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "args = sys.argv[1:]\n"
        "script = args[args.index('--quiet') + 1]\n"
        "rest = args[args.index('--quiet') + 2:]\n"
        "os.execv(sys.executable, [sys.executable, script, *rest])\n"
    )
    fake_uv.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    cand = write_candidate(ws, (
        "# /// script\n# dependencies = [\"not-really-installed\"]\n# ///\n"
        + UPPER
    ))

    assert run_candidate(ws, cand, {"text": "one"}).ok
    assert run_candidate(ws, cand, {"text": "two"}).ok
    with runner.CandidateSession(ws, cand) as session:
        assert session.run({"text": "three"}).ok
    with runner.CandidateSession(ws, cand) as session:
        assert session.run({"text": "four"}).ok

    assert len(list(Path(ws.tmp_dir).glob("ap_run_*.py"))) == 1
    assert len(list(Path(ws.tmp_dir).glob("ap_session_*.py"))) == 1
    assert not list(Path(ws.tmp_dir).glob("driver_*.py"))
    assert not list(Path(ws.tmp_dir).glob("session_driver_*.py"))


def test_remote_candidate_uses_workspace_owned_uv_cache(tmp_path, monkeypatch):
    import autoprogramming as ap

    ws = make_ws(tmp_path)
    target = ap.RemoteCompute(
        endpoint="gpu-box", transport="ssh", workdir="/remote/ap"
    )
    resources = ap.Resources(
        search=ap.SearchResources(
            allow_package_installs=True,
            allow_model_downloads=True,
            candidate_api_providers=(),
            remote_compute=target,
        ),
        runtime=ap.RuntimeResources(network=False),
        data=ap.DataPolicy(external_egress=False),
        confirmed=True,
    )
    ws.resources_json = ws.root / "resources.json"
    ws.resources_json.write_text(json.dumps(resources.to_dict()))
    cand = write_candidate(ws, UPPER)

    class FakeRemoteExecutor:
        def __init__(self, config, **kwargs):
            assert config == target
        def staged_root(self, local_root):
            return "/remote/ap/workspace-owner"
        def staged_dir(self, local_root, *, namespace):
            assert namespace == "evaluation-candidate_0"
            return "/remote/ap/workspace-owner/evaluation-candidate_0"
        def sync_to(self, *args, **kwargs):
            return None

    commands = []

    def fail_popen(command, **kwargs):
        commands.append(command)
        raise FileNotFoundError("test stop")

    monkeypatch.setattr(runner, "RemoteExecutor", FakeRemoteExecutor)
    monkeypatch.setattr(runner.subprocess, "Popen", fail_popen)
    with pytest.raises(RunnerError, match="Could not launch remote candidate"):
        runner.RemoteCandidateSession(ws, cand).start()
    assert "UV_CACHE_DIR=/remote/ap/workspace-owner/.uv-cache" in commands[0][2]


def test_tmp_dir_cleaned_up(tmp_path):
    ws = make_ws(tmp_path)
    cand = write_candidate(ws, UPPER)
    run_candidate(ws, cand, {"text": "x"})
    assert list(Path(ws.tmp_dir).iterdir()) == []
    failing = write_candidate(
        ws, 'def predict(text):\n    raise RuntimeError("nope")\n', name="candidate_1"
    )
    run_candidate(ws, failing, {"text": "x"})
    assert list(Path(ws.tmp_dir).iterdir()) == []


# -------------------------------------------------------------------- trace


def test_trace_success_block(tmp_path):
    ws = make_ws(tmp_path)
    cand = write_candidate(ws, UPPER)
    res = run_candidate(ws, cand, {"text": "hi"})
    trace = res.trace()
    assert "candidate_0" in trace
    assert '"text": "hi"' in trace
    assert '"Loud": "HI!"' in trace


def test_trace_failure_includes_traceback_and_stdio(tmp_path):
    ws = make_ws(tmp_path)
    source = (
        "import sys\n"
        "def predict(text):\n"
        '    print("before the crash")\n'
        '    print("stderr line", file=sys.stderr)\n'
        '    raise KeyError("missing")\n'
    )
    cand = write_candidate(ws, source)
    res = run_candidate(ws, cand, {"text": "x"})
    trace = res.trace()
    assert "FAILED" in trace
    assert "KeyError" in trace
    assert "before the crash" in trace
    assert "stderr line" in trace


def test_run_result_trace_without_run():
    res = RunResult(
        ok=False,
        outputs=None,
        error="timed out after 1.0s",
        stdout="",
        stderr="",
        duration_s=1.0,
        cost_dollars=0.5,
        candidate="candidate_9",
        inputs={"text": "x"},
    )
    trace = res.trace()
    assert "candidate_9" in trace
    assert "timed out" in trace
    assert "$0.5" in trace
