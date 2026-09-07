"""Pi RPC/JSON process integration without spending on a real model."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from autoprogramming.pi_backend import (
    PiOrchestratorBackend,
    PiResult,
    PiRpcClient,
    PiUsage,
    PiWorkerRunner,
    _execute_pi_calls,
    _WORKER_SYSTEM,
    _json_object,
    _materialize_bundle,
    _normalize_metric_suite_proposal,
    _task_document,
    _worker_env,
)
from autoprogramming import metric
from autoprogramming.budget import Budget, BudgetLedger
from autoprogramming.harness import AgentHarness
from autoprogramming.portfolio import default_avenue, ApproachTier, Portfolio
from autoprogramming.pi_worker import cleanup_worker_cache, worker_run_dir
from autoprogramming.schema import Schema
from autoprogramming.workspace import Workspace
from autoprogramming.errors import RunnerError


class PiOutput(str):
    pass


def pi_solve(text: str) -> PiOutput:
    """Transform text."""


def fake_pi(tmp_path: Path) -> Path:
    exe = tmp_path / "fake-pi"
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "if 'rpc' in args:\n"
        "  for line in sys.stdin:\n"
        "    cmd = json.loads(line)\n"
        "    if cmd.get('type') == 'abort': break\n"
        "    print(json.dumps({'id': cmd.get('id'), 'type':'response', 'command':'prompt', 'success':True}), flush=True)\n"
        "    message = {'role':'assistant','content':[{'type':'text','text':'{\\\"avenues\\\": []}'}],"
        "'usage':{'input':10,'output':5,'cacheRead':2,'cacheWrite':1,'cost':{'total':0.012}},'stopReason':'stop'}\n"
        "    print(json.dumps({'type':'message_end','message':message}), flush=True)\n"
        "    print(json.dumps({'type':'agent_settled'}), flush=True)\n"
        "else:\n"
        "  pathlib.Path('solution.py').write_text('def predict(text):\\n    return text\\n')\n"
        "  pathlib.Path('invocation.json').write_text(json.dumps({'argv':args,'cwd':os.getcwd(),'ap':os.environ.get('AP_WORKSPACE'),'openai':os.environ.get('OPENAI_API_KEY'),'groq':os.environ.get('GROQ_API_KEY'),'remote':os.environ.get('AP_REMOTE_ENDPOINT'),'remote_cwd':os.environ.get('AP_REMOTE_CWD'),'uv_cache':os.environ.get('UV_CACHE_DIR'),'remote_prefix':os.environ.get('AP_REMOTE_ENV_PREFIX')}))\n"
        "  message = {'role':'assistant','content':[{'type':'text','text':'implemented'}],"
        "'usage':{'input':3,'output':4,'cacheRead':0,'cacheWrite':0,'cost':{'total':0.005}},'stopReason':'stop'}\n"
        "  print(json.dumps({'type':'message_end','message':message}))\n"
    )
    exe.chmod(0o755)
    return exe


def failing_pi(tmp_path: Path) -> Path:
    exe = tmp_path / "failing-pi"
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "args = sys.argv[1:]\n"
        "message = {'role':'assistant','content':[],'usage':{'cost':{'total':0}},"
        "'stopReason':'error','errorMessage':'Connection error.'}\n"
        "if 'rpc' in args:\n"
        "  for line in sys.stdin:\n"
        "    cmd = json.loads(line)\n"
        "    print(json.dumps({'id':cmd.get('id'),'type':'response','command':'prompt','success':True}), flush=True)\n"
        "    print(json.dumps({'type':'message_end','message':message}), flush=True)\n"
        "    print(json.dumps({'type':'agent_settled'}), flush=True)\n"
        "else:\n"
        "  print(json.dumps({'type':'message_end','message':message}))\n"
    )
    exe.chmod(0o755)
    return exe


def test_pi_integration_is_split_into_rpc_worker_and_controller_modules():
    assert PiRpcClient.__module__ == "autoprogramming.pi_rpc"
    assert PiWorkerRunner.__module__ == "autoprogramming.pi_worker"
    assert PiOrchestratorBackend.__module__ == "autoprogramming.pi_backend"


def test_parallel_pi_calls_reserve_headroom_and_settle_actual_cost(tmp_path):
    ledger = BudgetLedger.start(tmp_path / "budget.json", Budget(dollars=1))
    lock = threading.Lock()
    active = 0
    max_active = 0

    def invoke(item):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return PiResult(str(item), usage=PiUsage(dollars=0.2))

    completed, undispatched = _execute_pi_calls(
        [1, 2, 3], max_workers=3, ledger=ledger,
        reservation_dollars=0.4, invoke=invoke,
    )
    assert len(completed) == 3
    assert undispatched == []
    assert max_active == 2  # only two $0.40 commitments fit initially
    assert ledger.spent["dollars"] == pytest.approx(0.6)
    assert ledger.reservations == {}


def test_parallel_pi_calls_serialize_without_confirmed_per_call_bound(tmp_path):
    ledger = BudgetLedger.start(tmp_path / "budget.json", Budget(dollars=0.7))
    lock = threading.Lock()
    active = 0
    max_active = 0

    def invoke(item):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return PiResult(str(item), usage=PiUsage(dollars=0.4))

    completed, undispatched = _execute_pi_calls(
        [1, 2, 3], max_workers=3, ledger=ledger,
        reservation_dollars=None, invoke=invoke,
    )
    assert len(completed) == 2
    assert undispatched == [3]
    assert max_active == 1
    assert ledger.spent["dollars"] == pytest.approx(0.8)


def test_backend_rejects_nonpositive_pi_timeouts():
    with pytest.raises(ValueError, match="timeouts"):
        PiOrchestratorBackend(orchestrator_timeout=0)
    with pytest.raises(ValueError, match="timeouts"):
        PiOrchestratorBackend(worker_timeout=-1)


def test_rpc_client_collects_final_text_and_usage(tmp_path):
    exe = fake_pi(tmp_path)
    with PiRpcClient(command=(str(exe),), cwd=tmp_path) as client:
        result = client.prompt("plan")
    assert json.loads(result.text) == {"avenues": []}
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 5
    assert result.usage.dollars == pytest.approx(0.012)
    assert result.usage.turns == 1


def test_rpc_client_rejects_provider_error_with_zero_process_exit(tmp_path):
    exe = failing_pi(tmp_path)
    with PiRpcClient(command=(str(exe),), cwd=tmp_path) as client:
        with pytest.raises(RunnerError, match="Connection error"):
            client.prompt("plan")


def test_worker_rejects_provider_error_with_zero_process_exit(tmp_path):
    exe = failing_pi(tmp_path)
    work = tmp_path / "task"
    work.mkdir()
    with pytest.raises(RunnerError, match="Connection error"):
        PiWorkerRunner(command=(str(exe),)).run(work, "implement")


def test_worker_environment_scrubs_host_session_and_unapproved_tokens(monkeypatch):
    monkeypatch.setenv("PI_SESSION_ID", "private-host-session")
    monkeypatch.setenv("PI_SESSION_FILE", "/private/session.jsonl")
    monkeypatch.setenv("GITHUB_TOKEN", "secret")
    monkeypatch.setenv("OPENAI_CODEX_OAUTH_TOKEN", "pi-authorized")
    env = _worker_env((), pi_model="openai-codex/gpt-5:max")
    assert "PI_SESSION_ID" not in env
    assert "PI_SESSION_FILE" not in env
    assert "GITHUB_TOKEN" not in env
    assert env["OPENAI_CODEX_OAUTH_TOKEN"] == "pi-authorized"


def test_worker_inherits_host_pi_subscription_model_without_api_key(tmp_path, monkeypatch):
    exe = fake_pi(tmp_path)
    work = tmp_path / "subscription-task"
    work.mkdir()
    monkeypatch.setenv("PI_PROVIDER", "openai-codex")
    monkeypatch.setenv("PI_MODEL", "gpt-5.6-sol")
    monkeypatch.setenv("PI_REASONING_LEVEL", "max")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    PiWorkerRunner(command=(str(exe),)).run(work, "implement")
    invocation = json.loads((work / "invocation.json").read_text())
    argv = invocation["argv"]
    model_at = argv.index("--model")
    assert argv[model_at + 1] == "openai-codex/gpt-5.6-sol:max"
    assert invocation["openai"] is None


def test_worker_uses_isolated_discovery_flags_and_scrubs_workspace(tmp_path, monkeypatch):
    exe = fake_pi(tmp_path)
    work = tmp_path / "task"
    work.mkdir()
    monkeypatch.setenv("AP_WORKSPACE", "/secret/optimizer")
    monkeypatch.setenv("OPENAI_API_KEY", "allowed")
    monkeypatch.setenv("GROQ_API_KEY", "blocked")
    result = PiWorkerRunner(command=(str(exe),)).run(
        work, "implement", session_id="avenue-one",
        allowed_api_providers=("openai",),
    )
    assert result.returncode == 0
    assert result.usage.dollars == pytest.approx(0.005)
    invocation = json.loads((work / "invocation.json").read_text())
    argv = invocation["argv"]
    assert invocation["cwd"] == str(work)
    assert invocation["ap"] is None
    assert invocation["openai"] == "allowed"
    assert invocation["groq"] is None
    assert invocation["uv_cache"] == str(tmp_path / ".uv-cache")
    assert "--no-context-files" in argv
    assert "--no-skills" in argv
    assert "--no-extensions" in argv
    assert "--extension" in argv  # explicit cooperative root guard still loads
    assert "--session-id" in argv
    assert (work / "solution.py").exists()


def test_worker_uses_remote_tools_only_when_user_provides_target(
    tmp_path, monkeypatch
):
    import autoprogramming as ap

    exe = fake_pi(tmp_path)
    work = tmp_path / "remote-task"
    work.mkdir()

    sync_calls = {}

    class FakeRemote:
        def __init__(self, config):
            self.config = config
        def staged_dir(self, *_args, **_kwargs):
            return "/remote/ap/task"
        def sync_to(self, *_args, **kwargs):
            sync_calls["to"] = kwargs
        def sync_from(self, *_args, **kwargs):
            sync_calls["from"] = kwargs

    monkeypatch.setattr("autoprogramming.pi_worker.RemoteExecutor", FakeRemote)
    target = ap.RemoteCompute(endpoint="gpu-box", transport="ssh", workdir="/remote/ap")
    PiWorkerRunner(
        command=(str(exe),), remote_compute=target
    ).run(work, "implement")
    invocation = json.loads((work / "invocation.json").read_text())
    assert invocation["remote"] == "gpu-box"
    assert invocation["remote_cwd"] == "/remote/ap/task"
    assert "UV_CACHE_DIR=/remote/ap/task-uv-cache" in invocation["remote_prefix"]
    assert ".venv-*" in sync_calls["to"]["excludes"]
    assert sync_calls["from"]["excludes"] == sync_calls["to"]["excludes"]
    extensions = [
        invocation["argv"][i + 1]
        for i, arg in enumerate(invocation["argv"][:-1])
        if arg == "--extension"
    ]
    assert any(path.endswith("remote-worker.ts") for path in extensions)


def test_worker_cache_cleanup_is_guarded_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("AP_WORKER_DIR", str(tmp_path / "ap-work"))
    schema = Schema.from_function(pi_solve)
    rows = [{"text": "x", "PiOutput": "x"}]
    workspace = Workspace.create(
        tmp_path / "cleanup_ap", schema,
        {"train": rows, "val": rows, "test": rows},
        seed=0, ratios=(0.6, 0.2, 0.2), data_sha="cleanup", bootstrap=True,
    )
    run_root = worker_run_dir(workspace)
    avenue = run_root / "avenue"
    venv = avenue / ".venv"
    venv.mkdir(parents=True)
    (venv / "package.bin").write_bytes(b"cache")
    uv_cache = run_root / ".uv-cache"
    uv_cache.mkdir()
    (uv_cache / "archive").write_bytes(b"uv")
    solution = avenue / "solution.py"
    solution.write_text("def predict(text): return text\n")
    diagnostics = avenue / ".ap" / "inactive_artifacts" / "candidate_1"
    diagnostics.mkdir(parents=True)
    (diagnostics / "trace.jsonl").write_text("diagnostic\n")

    with pytest.raises(RunnerError, match="unfinished"):
        cleanup_worker_cache(workspace)
    assert run_root.exists()
    forced = cleanup_worker_cache(workspace, force=True)
    assert forced["removed"] is True
    assert forced["preserved_worker_root"] is True
    assert not venv.exists()
    assert not uv_cache.exists()
    assert solution.exists()
    assert (diagnostics / "trace.jsonl").read_text() == "diagnostic\n"

    uv_cache.mkdir()
    (uv_cache / "archive").write_bytes(b"new")
    workspace.mark_finalized({"activated": "candidate_0"})
    result = cleanup_worker_cache(workspace)
    assert result["bytes_removed"] >= 3
    assert result["removed"] is True
    assert run_root.exists()
    assert solution.exists()
    assert diagnostics.exists()
    again = cleanup_worker_cache(workspace)
    assert again["bytes_removed"] == 0
    assert again["removed"] is True
    assert again["preserved_worker_root"] is True


def test_worker_cache_cleanup_removes_only_remote_caches(tmp_path, monkeypatch):
    import autoprogramming as ap

    monkeypatch.setenv("AP_WORKER_DIR", str(tmp_path / "ap-work"))
    schema = Schema.from_function(pi_solve)
    rows = [{"text": "x", "PiOutput": "x"}]
    workspace = Workspace.create(
        tmp_path / "remote_cleanup_ap", schema,
        {"train": rows, "val": rows, "test": rows},
        seed=0, ratios=(0.6, 0.2, 0.2), data_sha="remote-cleanup",
        bootstrap=True,
    )
    worker_run_dir(workspace).mkdir(exist_ok=True)
    workspace.mark_finalized({"activated": "candidate_0"})
    workspace.portfolio_json.parent.mkdir(parents=True, exist_ok=True)
    workspace.portfolio_json.write_text("{}\n")
    remote = ap.RemoteCompute(
        endpoint="gpu-box", transport="ssh", workdir="/remote/ap"
    )
    spec = SimpleNamespace(
        id="deep-model",
        tier=ApproachTier.SPECIALIZED_DEEP_MODEL,
        required_capabilities=(),
        required_mechanisms=(),
    )
    fake_portfolio = SimpleNamespace(
        resources=SimpleNamespace(
            search=SimpleNamespace(remote_compute=remote)
        ),
        avenues=[SimpleNamespace(spec=spec)],
    )
    monkeypatch.setattr(
        Portfolio, "load", classmethod(lambda _cls, _path: fake_portfolio)
    )
    commands = []

    class FakeRemoteExecutor:
        def __init__(self, config):
            assert config is remote
        def staged_root(self, local_root):
            return "/remote/ap/workspace"
        def staged_dir(self, local_root, *, namespace):
            assert namespace == "deep-model"
            return "/remote/ap/run/deep-model"
        def ssh(self, command):
            commands.append(command)

    monkeypatch.setattr(
        "autoprogramming.pi_worker.RemoteExecutor", FakeRemoteExecutor
    )
    result = cleanup_worker_cache(workspace)
    assert result["remote_errors"] == []
    assert result["remote_removed"] == [
        "/remote/ap/workspace/.uv-cache",
        "/remote/ap/run/deep-model-uv-cache",
    ]
    assert len(commands) == 1
    assert commands[0].startswith(
        "rm -rf -- /remote/ap/workspace/.uv-cache "
        "/remote/ap/run/deep-model-uv-cache; "
    )
    assert "find /remote/ap/run/deep-model" in commands[0]
    assert "rm -rf -- /remote/ap/run/deep-model " not in commands[0]


def test_bundle_import_replaces_orphan_from_pre_candidate_crash(tmp_path):
    schema = Schema.from_function(pi_solve)
    rows = [{"text": "x", "PiOutput": "x"}]
    workspace = Workspace.create(
        tmp_path / "bundle_ap", schema,
        {"train": rows, "val": rows, "test": rows},
        seed=0, ratios=(0.6, 0.2, 0.2), data_sha="bundle", bootstrap=True,
    )
    sandbox = tmp_path / "worker"
    source_artifacts = sandbox / "artifacts" / "rules"
    source_artifacts.mkdir(parents=True)
    (source_artifacts / "model.bin").write_bytes(b"new")
    orphan = workspace.artifacts_dir / "rules-candidate_0"
    orphan.mkdir()
    (orphan / "model.bin").write_bytes(b"orphan")
    source = (
        "# /// script\n# [tool.ap]\n# artifact_namespace = \"rules\"\n# ///\n"
        "def predict(text):\n    return text\n"
    )

    rewritten = _materialize_bundle(source, sandbox, workspace, "rules")
    assert 'artifact_namespace = "rules-candidate_0"' in rewritten
    assert (orphan / "model.bin").read_bytes() == b"new"


def test_controller_recovers_scored_pending_candidate_without_duplication(tmp_path):
    import autoprogramming as ap

    schema = Schema.from_function(pi_solve)
    rows = [{"text": "x", "PiOutput": "x"}, {"text": "y", "PiOutput": "y"}]
    workspace = Workspace.create(
        tmp_path / "pi_solve_ap", schema,
        {"train": rows, "val": rows, "test": rows},
        seed=0, ratios=(0.6, 0.2, 0.2), data_sha="pending", bootstrap=True,
    )
    BudgetLedger.start(workspace.budget_json, Budget(eval_calls=100))
    metric.write_metric(workspace, "def metric(p, e):\n    return float(p == e)\n")
    metric.approve(workspace, "tester")
    harness = AgentHarness(workspace)
    candidate = harness.new_candidate(source=(
        "# /// script\n# [tool.ap]\n# deterministic = true\n"
        "# cost_per_call = 0.0\n# ///\n"
        "def predict(text):\n    return text\n"
    ))
    harness.eval(candidate.name, split="train", per_instance=True)
    harness.eval(candidate.name)

    resources = ap.Resources(
        search=ap.SearchResources(
            allow_package_installs=False, allow_model_downloads=False
        ),
        runtime=ap.RuntimeResources(network=False),
        data=ap.DataPolicy(external_egress=False),
        confirmed=True,
    )
    portfolio = Portfolio.create(resources, [])
    state = next(
        avenue for avenue in portfolio.avenues
        if avenue.spec.tier == ApproachTier.CODE_AND_RULES and not avenue.spec.wildcard
    )
    state.begin_candidate(candidate.name)
    path = tmp_path / "portfolio.json"
    portfolio.write(path)

    PiOrchestratorBackend()._recover_pending_avenues(harness, portfolio, path)
    assert state.pending_candidate is None
    assert state.candidates == [candidate.name]
    assert state.rounds == 1
    assert "recovered" in state.notes[-1]
    assert len(list(workspace.candidates_dir.glob("candidate_*.py"))) == 1


def test_pi_subscription_avenue_tells_worker_not_to_require_raw_key():
    import autoprogramming as ap

    resources = ap.Resources(
        search=ap.SearchResources(
            pi_models=("openai-codex/gpt-5.6-sol:max",),
            candidate_api_providers=(),
            allow_package_installs=False,
            allow_model_downloads=False,
        ),
        runtime=ap.RuntimeResources(network=False),
        data=ap.DataPolicy(external_egress=True),
        confirmed=True,
    )
    brief = _task_document(
        Schema.from_function(pi_solve),
        default_avenue(ApproachTier.SINGLE_MODEL_CALL, resources),
        resources,
    )
    assert "authenticated pi models" in brief.lower()
    assert "absence of an `*_API_KEY`" in brief
    assert "openai-codex/gpt-5.6-sol:max" in brief


def test_worker_brief_contains_no_optimizer_or_metric_context():
    import autoprogramming as ap

    resources = ap.Resources(
        search=ap.SearchResources(
            allow_package_installs=False, allow_model_downloads=False
        ),
        runtime=ap.RuntimeResources(network=False),
        data=ap.DataPolicy(external_egress=False),
        confirmed=True,
    )
    brief = _task_document(
        Schema.from_function(pi_solve),
        default_avenue(ApproachTier.CODE_AND_RULES, resources),
        resources,
    )
    hidden_terms = ("autoprogramming", "prg.", "metric", "leaderboard", "candidate_")
    combined = (_WORKER_SYSTEM + brief).lower()
    assert all(term not in combined for term in hidden_terms)
    assert "non-negotiable approach contract" in combined
    assert "never replace the assigned mechanism" in combined
    assert "pep 723" in combined
    assert "max_parallel_agents" not in combined  # does not reveal other workers


def test_json_extractor_accepts_plain_and_fenced():
    assert _json_object('{"x": 1}') == {"x": 1}
    assert _json_object('result:\n```json\n{"x": 2}\n```') == {"x": 2}
    with pytest.raises(RunnerError, match="required JSON"):
        _json_object("not json")


def test_metric_suite_proposal_repairs_names_after_critic_rewrite():
    suite, adjustments = _normalize_metric_suite_proposal(
        {
            "acceptance": ["old_exact"],
            "diagnostic": ["old_similarity"],
            "preference_order": ["old_exact"],
            "floors": {"old_exact": 0.9, "exact_match": "not-a-number"},
        },
        ("exact_match", "char_similarity"),
    )
    assert suite.acceptance == ("exact_match",)
    assert suite.diagnostic == ("char_similarity",)
    assert suite.policy.preference_order == ("exact_match",)
    assert suite.policy.floors == {}
    assert adjustments
