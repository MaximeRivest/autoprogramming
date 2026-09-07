"""Implementation-only Pi workers and their isolated task bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from .errors import RunnerError
from .pi_rpc import PiResult, PiUsage, assistant_failure
from .portfolio import AvenueSpec
from .remote import (
    RemoteExecutor,
    gpu_environment_prefix,
    use_remote_for_avenue,
)
from .resources import RemoteCompute, Resources

_WORKER_SYNC_EXCLUDES = (".venv", ".venv-*", ".venv_*", ".uv-cache")


WORKER_SYSTEM = """You are the sole implementation engineer for a standalone
Python function. Your first and non-negotiable obligation is MECHANISM FIDELITY:
implement exactly the approach contract in task.md and push that particular
mechanism as far as possible. Returning a plausible answer through a different
approach is a failure, even if it avoids an exception or appears more reliable.

NEVER replace the assigned mechanism because a package, model, API credential,
GPU, network service, or other capability is missing in your authoring shell.
Third-party packages need not already be installed: declare them in the PEP 723
block so the execution controller can resolve them. Do not create multiple
persistent virtual environments for experiments. If a dependency must be tested,
reuse one `.venv` in this task directory and treat it as disposable scratch; never
place required runtime files there. Pi model access is different
from a raw SDK key: when task.md lists an authenticated Pi model, the local Pi
CLI resolves its stored OAuth/subscription login itself. Do not reject that
capability merely because OPENAI_API_KEY/ANTHROPIC_API_KEY is absent; use Pi's
CLI/RPC runtime for the implementation when assigned and declare
`pi_runtime = true` under `[tool.ap]`. If an assigned capability
really cannot be exercised, still implement it faithfully, syntax-check what
you can, and make the runtime fail clearly with a precise setup error. Do not
add classical ML, classical CV, rules, regex, lookup, local-model, or API
fallbacks from another approach family. Retries, parsing, preprocessing, and
error handling are welcome only when the assigned mechanism remains the path
that produces the answer. Cross-family routing is allowed solely when task.md
explicitly identifies this as a composition contract.

You have no broader coordination duties. Work only in the provided directory.
Read task.md and examples.jsonl, then create solution.py defining predict with
exactly the input parameters in task.md. solution.py may contain one valid PEP
723 script block for dependencies. Do not hard-code or copy example outputs into
a lookup table. Load clients and models lazily. If files are needed at runtime,
put them under the artifact namespace named in task.md and declare that same
directory as `artifact_namespace` under `[tool.ap]`; resolve it as
`Path(__file__).parents[1] / "artifacts" / <namespace>`. If a call spends money,
report AP_COST_DOLLARS after each prediction. Finish only after checking that no
error branch substitutes a different mechanism and syntax-checking solution.py.
"""


class PiWorkerRunner:
    """Run one implementation-only Pi worker in an isolated context."""

    def __init__(
        self,
        command: tuple[str, ...] = ("pi",),
        timeout: float = 1200.0,
        *,
        remote_compute: RemoteCompute | None = None,
    ):
        self.command = tuple(command)
        self.timeout = timeout
        self.remote_compute = remote_compute

    def run(
        self,
        cwd: Path,
        task: str,
        *,
        model: str | None = None,
        session_id: str | None = None,
        allowed_api_providers: tuple[str, ...] = (),
    ) -> PiResult:
        remote_executor = (
            RemoteExecutor(self.remote_compute) if self.remote_compute else None
        )
        remote_root = None
        remote_uv_cache = None
        if remote_executor is not None:
            remote_root = remote_executor.staged_dir(cwd, namespace=cwd.name)
            remote_uv_cache = f"{remote_root}-uv-cache"
            remote_executor.sync_to(
                cwd, remote_root, excludes=_WORKER_SYNC_EXCLUDES
            )

        guard_source = Path(__file__).parent / "pi" / "worker-guard.ts"
        guard = cwd / ".tools" / "root-guard.ts"
        guard.parent.mkdir(exist_ok=True)
        shutil.copyfile(guard_source, guard)
        args = [
            *self.command,
            "--mode", "json", "--print",
            "--no-extensions", "--extension", str(guard),
            "--no-skills", "--no-prompt-templates",
            "--no-themes", "--no-context-files", "--no-approve",
            "--tools", "read,bash,edit,write",
            "--system-prompt", WORKER_SYSTEM,
        ]
        if remote_executor is not None:
            remote_source = Path(__file__).parent / "pi" / "remote-worker.ts"
            remote_extension = cwd / ".tools" / "remote-worker.ts"
            shutil.copyfile(remote_source, remote_extension)
            args.extend(("--extension", str(remote_extension)))
        if session_id:
            session_dir = cwd / ".pi-sessions"
            session_dir.mkdir(exist_ok=True)
            args.extend(("--session-dir", str(session_dir), "--session-id", session_id))
        else:
            args.append("--no-session")
        model = model or inherited_pi_model()
        if model:
            args.extend(("--model", model))
        args.append(task)
        env = worker_env(allowed_api_providers, pi_model=model)
        # Worker package installs belong to this search run, not to the user's
        # global uv cache. The run cache stays available for repair turns and is
        # removed by cleanup_worker_cache after finalization.
        env["UV_CACHE_DIR"] = str(worker_uv_cache_dir(cwd))
        if remote_executor is not None:
            env["AP_REMOTE_ENDPOINT"] = self.remote_compute.endpoint
            env["AP_REMOTE_CWD"] = str(remote_root)
            remote_prefix = gpu_environment_prefix(self.remote_compute)
            cache_assignment = f"UV_CACHE_DIR={shlex.quote(str(remote_uv_cache))}"
            env["AP_REMOTE_ENV_PREFIX"] = " ".join(
                value for value in (remote_prefix, cache_assignment) if value
            )
        try:
            proc = subprocess.run(
                args,
                cwd=str(cwd),
                text=True,
                capture_output=True,
                timeout=self.timeout,
                check=False,
                env=env,
            )
        except FileNotFoundError as exc:
            raise RunnerError(
                f"Pi worker executable {self.command[0]!r} was not found."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RunnerError(
                f"Pi implementation worker timed out after {self.timeout:g}s."
            ) from exc
        finally:
            if remote_executor is not None and remote_root is not None:
                # Retrieve even a partial solution: the controller will audit it
                # and can ask the same approach worker to repair it.
                remote_executor.sync_from(
                    remote_root, cwd, excludes=_WORKER_SYNC_EXCLUDES
                )

        messages: list[dict] = []
        usage = PiUsage()
        text = ""
        for line in proc.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "message_end" and isinstance(
                event.get("message"), dict
            ):
                msg = event["message"]
                messages.append(msg)
                usage.add_message(msg)
                if msg.get("role") == "assistant":
                    parts = [
                        part.get("text", "")
                        for part in msg.get("content", [])
                        if part.get("type") == "text"
                    ]
                    if parts:
                        text = "".join(parts)
        if proc.returncode != 0:
            raise RunnerError(
                f"Pi implementation worker exited with code {proc.returncode}: "
                f"{proc.stderr[-2000:]}"
            )
        failure = assistant_failure(messages)
        if failure:
            raise RunnerError(f"{failure}\n{proc.stderr[-2000:]}")
        return PiResult(
            text=text,
            usage=usage,
            messages=messages,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )


def inherited_pi_model() -> str | None:
    """Exact provider/model/thinking selected in the human-facing Pi session."""
    provider = os.environ.get("PI_PROVIDER", "").strip()
    model = os.environ.get("PI_MODEL", "").strip()
    if not provider or not model:
        return None
    result = f"{provider}/{model}"
    thinking = os.environ.get("PI_REASONING_LEVEL", "").strip().lower()
    if thinking:
        result += f":{thinking}"
    return result


_PROVIDER_ENV_MARKERS = {
    "anthropic": ("ANTHROPIC_", "ANT_LING_"),
    "openai": ("OPENAI_", "AZURE_OPENAI_"),
    "google": ("GEMINI_", "GOOGLE_"),
    "groq": ("GROQ_",),
    "mistral": ("MISTRAL_",),
    "openrouter": ("OPENROUTER_",),
    "together": ("TOGETHER_",),
    "fireworks": ("FIREWORKS_",),
    "deepseek": ("DEEPSEEK_",),
    "xai": ("XAI_",),
    "aws": ("AWS_",),
    "github": ("GITHUB_", "GH_",),
    "github-copilot": ("GITHUB_", "GH_", "COPILOT_"),
    "openai-codex": ("OPENAI_CODEX_", "CODEX_", "CHATGPT_"),
    "claude-code": ("CLAUDE_CODE_",),
}


def worker_env(
    allowed_api_providers: tuple[str, ...] = (), *, pi_model: str | None = None
) -> dict[str, str]:
    """Scrub controller state and candidate API credentials outside allowlists."""
    allowed = {str(name).lower() for name in allowed_api_providers}
    if pi_model:
        lower_model = pi_model.lower()
        if "/" in lower_model:
            allowed.add(lower_model.split("/", 1)[0])
        elif any(
            token in lower_model for token in ("claude", "sonnet", "haiku", "opus")
        ):
            allowed.add("anthropic")
        elif "gemini" in lower_model:
            allowed.add("google")
        elif any(token in lower_model for token in ("gpt", "o3", "o4")):
            allowed.add("openai")
    allowed_markers = tuple(
        marker
        for provider in allowed
        for marker in _PROVIDER_ENV_MARKERS.get(
            provider, (provider.upper() + "_",)
        )
    )
    all_markers = tuple(
        marker for markers in _PROVIDER_ENV_MARKERS.values() for marker in markers
    )
    result: dict[str, str] = {}
    for key, value in os.environ.items():
        if (
            key == "AP_WORKSPACE"
            or key.startswith("AUTOPROGRAMMING_")
            or key == "PYTHONPATH"
            or key in ("PI_SESSION_ID", "PI_SESSION_FILE")
        ):
            continue
        credential_like = (
            key.endswith((
                "_API_KEY", "_OAUTH_TOKEN", "_TOKEN", "_SECRET", "_PASSWORD"
            ))
            or key
            in (
                "AWS_SECRET_ACCESS_KEY",
                "AWS_ACCESS_KEY_ID",
                "AWS_BEARER_TOKEN_BEDROCK",
            )
            or key.startswith(all_markers)
        )
        if credential_like and not key.startswith(allowed_markers):
            continue
        result[key] = value
    return result


def materialize_bundle(
    source: str, sandbox: Path, workspace, namespace: str
) -> str:
    """Copy worker artifacts into an immutable per-candidate namespace."""
    from .candidates import next_name

    candidate_name = next_name(workspace)
    versioned = f"{namespace}-{candidate_name}"
    source = source.replace(f'"{namespace}"', f'"{versioned}"')
    source = source.replace(f"'{namespace}'", f"'{versioned}'")
    source_artifacts = sandbox / "artifacts" / namespace
    if source_artifacts.is_dir():
        from .candidates import parse_pep723

        metadata = parse_pep723(source) or {}
        declared = ((metadata.get("tool") or {}).get("ap") or {}).get(
            "artifact_namespace"
        )
        if declared != versioned:
            raise RunnerError(
                "Worker created runtime artifacts but solution.py does not declare "
                f"[tool.ap] artifact_namespace = {versioned!r}."
            )
        target = Path(workspace.artifacts_dir) / versioned
        if target.exists():
            candidate_path = Path(workspace.candidates_dir) / f"{candidate_name}.py"
            if candidate_path.exists():
                raise RunnerError(
                    f"Refusing to overwrite candidate artifact bundle {target}."
                )
            # The controller journals the expected candidate name before bundle
            # import. A crash after copy but before candidate creation can leave
            # only this namespace; it is safe to replace because no candidate
            # file can reference it yet.
            shutil.rmtree(target)
        shutil.copytree(source_artifacts, target)
    return source


def task_document(schema, spec: AvenueSpec, resources: Resources) -> str:
    inputs = "\n".join(
        f"- `{field.name}: {field.type_name}` — {field.description}"
        for field in schema.inputs
    )
    outputs = "\n".join(
        f"- `{field.name}: {field.type_name}` — {field.description}"
        for field in schema.outputs
    )
    search_resources = {
        key: getattr(resources.search, key)
        for key in (
            "cpu_cores", "memory_gb", "disk_gb", "gpu", "gpu_vram_gb",
            "allow_package_installs", "allow_model_downloads", "fine_tuning",
        )
    }
    search_resources["available_runtime_api_access"] = (
        resources.search.candidate_api_providers
    )
    search_resources["authenticated_pi_models"] = resources.search.pi_models
    supplied_remote = resources.search.remote_compute
    remote = (
        supplied_remote
        if supplied_remote is not None and use_remote_for_avenue(spec)
        else None
    )
    search_resources["remote_compute_available"] = supplied_remote is not None
    search_resources["remote_compute_provided"] = remote is not None
    if remote is not None:
        search_resources["effective_compute_location"] = "user-provided remote target"
        search_resources["effective_remote_compute"] = {
            "cpu_cores": remote.cpu_cores,
            "memory_gb": remote.memory_gb,
            "disk_gb": remote.disk_gb,
            "gpu": remote.gpu,
            "gpu_vram_gb": remote.gpu_vram_gb,
            "max_parallel_gpu_jobs": remote.max_parallel_gpu_jobs,
            "min_free_gpu_vram_gb": remote.min_free_gpu_vram_gb,
        }
    return f"""# Implementation task

## Goal
{schema.doc}

## Inputs
{inputs}

## Outputs
{outputs}

`predict` must accept exactly: {', '.join(schema.input_names)}.
For one output, return its value. For several outputs, return a tuple in the
order above or a dict keyed by output name.

## Non-negotiable approach contract
{spec.title}: {spec.implementation_brief}

Hypothesis: {spec.hypothesis}
Required mechanism boundary: {spec.mechanism}
Required mechanism evidence: {list(spec.required_mechanisms)}
Forbidden substitutions: {list(spec.forbidden_substitutions)}
Cross-tier fallback permitted: {spec.allow_cross_tier_fallback}

The implementation is invalid if another family produces the answer when this
mechanism is unavailable. Missing dependencies or capabilities must cause a
clear setup/runtime failure, never a substitute implementation.

## Available build/search resources
{json.dumps(search_resources, default=str, indent=2)}

Declare placement facts under `[tool.ap]`: set `network_required = true` and
`api_providers = ["..."]` for a network/provider runtime; set `pi_runtime =
true` for Pi; and set `compute_heavy = true` for training, large local models,
or compute-heavy inference (`false` for genuinely lightweight code). These are
capability metadata, never credentials.

Packages declared in solution.py's PEP 723 block are resolved by the execution
controller. Their absence from the current shell is not a reason to avoid them.
When `remote_compute_provided` is true, your file and shell tools already operate
on that user-provided target. Treat `effective_remote_compute` as the available
build machine; do not reject an avenue because the host computer is smaller and
do not route heavy work back to the host.

## Permitted runtime resources
{json.dumps(resources.runtime.__dict__, default=str, indent=2)}
Allowed API providers for this implementation: {list(spec.allowed_api_providers)}
Authenticated Pi models available without raw SDK keys: {list(resources.search.pi_models)}
Deployment notes: {list(spec.deployment_notes)}

When this contract names a `pi-model:` capability, implement model calls through
Pi (`pi --mode rpc` or another persistent Pi CLI/SDK integration), and declare
`pi_runtime = true` under `[tool.ap]`. Pi reads the user's stored
OAuth/subscription login. The absence of an `*_API_KEY` environment variable is
not evidence that this capability is unavailable. Report Pi usage cost from its
JSON events and load the Pi process lazily. The controller evaluates this
network-bound point beside the authenticated host even when heavy compute is
staged remotely; it never copies OAuth credentials to the compute target.

## Runtime artifact namespace
If runtime files are needed, use `artifacts/{spec.id}/` here and declare
`artifact_namespace = "{spec.id}"` under `[tool.ap]` in solution.py. At runtime
that directory is `Path(__file__).parents[1] / "artifacts" / "{spec.id}"`.
"""


def _worker_token(workspace) -> str:
    try:
        token = workspace.active.get("private_data_id")
    except Exception:
        token = None
    if not token:
        token = hashlib.sha256(
            str(Path(workspace.root).resolve()).encode()
        ).hexdigest()[:24]
    token = str(token)
    if not token or Path(token).name != token or token in (".", ".."):
        raise RunnerError(f"Unsafe AutoProgramming worker cache id {token!r}.")
    return token


def worker_cache_base() -> Path:
    return Path(
        os.environ.get("AP_WORKER_DIR", Path.home() / ".cache" / "ap-work")
    ).expanduser()


def worker_run_path(workspace) -> Path:
    """Worker root without creating it; cleanup must never recreate dead runs."""
    return worker_cache_base() / _worker_token(workspace)


def worker_run_dir(workspace) -> Path:
    """Opaque worker root outside the optimizer/package workspace."""
    path = worker_run_path(workspace)
    path.mkdir(parents=True, exist_ok=True)
    return path


def worker_uv_cache_dir(cwd: Path) -> Path:
    """Shared uv cache owned by the worker run containing ``cwd``."""
    return Path(cwd).resolve().parent / ".uv-cache"


def avenue_dir(workspace, avenue_id: str) -> Path:
    return worker_run_dir(workspace) / avenue_id


def _tree_size(path: Path) -> int:
    total = 0
    if not path.exists() or path.is_symlink():
        return total
    for current, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(current) / name).stat().st_size
            except OSError:
                pass
    return total


def _is_worker_environment(name: str) -> bool:
    return (
        name == ".uv-cache"
        or name == ".venv"
        or name.startswith(".venv-")
        or name.startswith(".venv_")
    )


def _local_worker_caches(root: Path) -> list[Path]:
    """Cache/environment directories only; never candidate diagnostics."""
    if not root.exists():
        return []
    if root.is_symlink():
        raise RunnerError(f"Refusing to traverse symlinked worker root {root}.")
    result: list[Path] = []
    for current, dirs, _files in os.walk(root, topdown=True, followlinks=False):
        for name in tuple(dirs):
            if not _is_worker_environment(name):
                continue
            result.append(Path(current) / name)
            dirs.remove(name)
    return result


def _remove_cache_directory(path: Path) -> None:
    if path.is_symlink():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)


def cleanup_worker_cache(workspace, *, force: bool = False) -> dict:
    """Remove only owned UV caches and disposable virtual environments.

    Worker source, task files, artifacts, output caches, and inactive-candidate
    diagnostics are always preserved. Unfinished runs require ``force=True`` so
    an active package setup is not removed while it is in use.
    """
    try:
        finalized = bool(workspace.active.get("finalized"))
    except Exception:
        finalized = False
    if not finalized and not force:
        raise RunnerError(
            "Refusing to remove worker cache for an unfinished search. Finalize "
            "the workspace first, or pass force=True to discard only its package "
            "cache and disposable virtual environments."
        )

    base = worker_cache_base().resolve()
    root = worker_run_path(workspace)
    if root.parent.resolve() != base:
        raise RunnerError(f"Refusing unsafe worker cache cleanup outside {base}.")

    remote_removed: list[str] = []
    remote_errors: list[str] = []
    portfolio_path = getattr(workspace, "portfolio_json", None)
    if portfolio_path is not None and Path(portfolio_path).exists():
        try:
            from .portfolio import Portfolio
            from .remote import use_remote_for_avenue

            portfolio = Portfolio.load(portfolio_path)
            remote = portfolio.resources.search.remote_compute
            if remote is not None:
                executor = RemoteExecutor(remote)
                worker_roots: list[str] = []
                cache_targets: list[str] = [
                    f"{executor.staged_root(workspace.root)}/.uv-cache"
                ]
                for avenue in portfolio.avenues:
                    if not use_remote_for_avenue(avenue.spec):
                        continue
                    local_avenue = root / avenue.spec.id
                    remote_root = executor.staged_dir(
                        local_avenue, namespace=local_avenue.name
                    )
                    worker_roots.append(remote_root)
                    cache_targets.append(f"{remote_root}-uv-cache")
                cache_targets = list(dict.fromkeys(cache_targets))
                # Stage roots contain implementation and inactive-candidate
                # diagnostics. Delete only exact cache paths and named venvs.
                commands = [
                    "rm -rf -- "
                    + " ".join(shlex.quote(value) for value in cache_targets)
                ]
                for remote_root in worker_roots:
                    quoted = shlex.quote(remote_root)
                    commands.append(
                        f"if [ -d {quoted} ]; then find {quoted} -mindepth 1 "
                        "-maxdepth 1 "
                        "\\( -name .venv -o -name '.venv-*' -o "
                        "-name '.venv_*' -o -name .uv-cache \\) "
                        "-exec rm -rf -- {} +; fi"
                    )
                executor.ssh("; ".join(commands))
                remote_removed.extend(cache_targets)
        except Exception as exc:
            # Finalization is already durable. A disconnected optional compute
            # target must not turn cache hygiene into a failed program result.
            remote_errors.append(str(exc))

    local_caches = _local_worker_caches(root)
    bytes_removed = sum(_tree_size(path) for path in local_caches)
    for path in local_caches:
        _remove_cache_directory(path)
    return {
        "path": str(root),
        "bytes_removed": bytes_removed,
        "removed": all(not path.exists() for path in local_caches),
        "cache_paths_removed": [str(path) for path in local_caches],
        "preserved_worker_root": root.exists(),
        "remote_removed": remote_removed,
        "remote_errors": remote_errors,
    }
