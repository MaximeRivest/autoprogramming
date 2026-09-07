"""Resource and data-governance contracts for an optimization run.

AutoProgramming must not infer that a build machine's capabilities are also
available to the shipped program.  The objects in this module keep search-time
resources, deployment-time resources, and data-egress policy separate.  They
contain capability names and limits only -- never API keys or other secrets.
"""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .errors import AutoProgrammingError


class ResourceError(AutoProgrammingError):
    """A resource profile is invalid or has not been confirmed."""


def _positive(name: str, value, *, allow_none: bool = True) -> None:
    if value is None and allow_none:
        return
    if value is None or not math.isfinite(value) or value <= 0:
        raise ResourceError(f"{name} must be positive and finite, got {value!r}.")


def _tuple(value) -> tuple:
    if value is None:
        return ()
    return tuple(value)


def _current_pi_model_pattern() -> str | None:
    """The human-facing Pi model, including thinking level, when inside Pi.

    Pi injects these values into its bash tool. Reading them lets child workers
    inherit the exact authenticated model selected by the user instead of
    guessing from API-key environment variables or global defaults.
    """
    provider = os.environ.get("PI_PROVIDER", "").strip()
    model = os.environ.get("PI_MODEL", "").strip()
    if not provider or not model:
        return None
    pattern = f"{provider}/{model}"
    thinking = os.environ.get("PI_REASONING_LEVEL", "").strip().lower()
    if thinking:
        pattern += f":{thinking}"
    return pattern


def _available_pi_model_patterns(current: str) -> tuple[str, ...]:
    """Authenticated models exposed by Pi's registry, active model first.

    ``pi --list-models`` prints capability names only and lets Pi resolve OAuth;
    no token or API-key material is read or persisted here. Discovery failure is
    non-fatal because the active model remains a known-good capability.
    """
    executable = shutil.which("pi")
    if executable is None:
        return (current,)
    try:
        proc = subprocess.run(
            [executable, "--offline", "--list-models"],
            text=True,
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return (current,)
    if proc.returncode != 0:
        return (current,)
    discovered: list[str] = [current]
    current_base = current.rsplit(":", 1)[0]
    current_provider = current_base.split("/", 1)[0]
    for line in proc.stdout.splitlines()[1:]:
        columns = line.split()
        if len(columns) < 2 or columns[0] != current_provider:
            continue
        pattern = f"{columns[0]}/{columns[1]}"
        if pattern == current_base or pattern in discovered:
            continue
        discovered.append(pattern)
        if len(discovered) >= 32:
            break
    return tuple(discovered)


@dataclass(frozen=True)
class DataPolicy:
    """Where task data may go during search and at runtime.

    ``external_egress`` is deliberately tri-state.  ``None`` means the user has
    not answered yet; a networked approach must never treat that as consent.
    """

    external_egress: bool | None = None
    allowed_domains: tuple[str, ...] = ()
    sensitive: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_domains", _tuple(self.allowed_domains))


@dataclass(frozen=True)
class RemoteCompute:
    """Optional user-provided search-time compute reached through a transport.

    Remote execution is never inferred and never required by the skill.  When
    supplied, the controller keeps orchestration/bookkeeping and lightweight
    work local while preferring this target for package/model setup, training,
    and compute-heavy evaluation. The transport must be chosen explicitly;
    authentication stays in that adapter's normal
    configuration (for SSH, the user's agent/config); no credentials are stored
    here.
    """

    endpoint: str
    transport: str | None = None
    workdir: str | None = None
    cpu_cores: int | None = None
    memory_gb: float | None = None
    disk_gb: float | None = None
    gpu: str | None = None
    gpu_vram_gb: float | None = None
    max_parallel_cpu_jobs: int | None = None
    max_parallel_gpu_jobs: int = 1
    min_free_gpu_vram_gb: float | None = None

    def __post_init__(self) -> None:
        endpoint = str(self.endpoint).strip()
        if not endpoint:
            raise ResourceError("remote_compute.endpoint must not be empty.")
        if endpoint.startswith("-") or any(ch in endpoint for ch in "\r\n\0"):
            raise ResourceError(
                "remote_compute.endpoint must be an SSH host/alias, not options "
                "or control characters. Put SSH options in the user's SSH config."
            )
        if self.transport is None:
            raise ResourceError(
                "remote_compute.transport must be chosen explicitly; remote "
                "compute is never assumed to mean SSH. This release provides "
                "the 'ssh' transport."
            )
        if self.transport != "ssh":
            raise ResourceError(
                f"Unsupported remote compute transport {self.transport!r}; "
                "this release currently provides the pluggable 'ssh' adapter."
            )
        if self.workdir is not None and any(
            ch in str(self.workdir) for ch in "\r\n\0"
        ):
            raise ResourceError("remote_compute.workdir contains control characters.")
        _positive("remote_compute.cpu_cores", self.cpu_cores)
        _positive("remote_compute.memory_gb", self.memory_gb)
        _positive("remote_compute.disk_gb", self.disk_gb)
        _positive("remote_compute.gpu_vram_gb", self.gpu_vram_gb)
        _positive(
            "remote_compute.max_parallel_cpu_jobs",
            self.max_parallel_cpu_jobs,
        )
        _positive(
            "remote_compute.max_parallel_gpu_jobs",
            self.max_parallel_gpu_jobs,
            allow_none=False,
        )
        _positive(
            "remote_compute.min_free_gpu_vram_gb",
            self.min_free_gpu_vram_gb,
        )
        if (
            self.gpu is not None
            and self.gpu_vram_gb is not None
            and self.min_free_gpu_vram_gb is None
        ):
            # Conservative external-contention gate; callers can lower it for
            # small models or raise it for near-full-card jobs.
            object.__setattr__(
                self,
                "min_free_gpu_vram_gb",
                0.8 * float(self.gpu_vram_gb),
            )
        if (
            self.min_free_gpu_vram_gb is not None
            and self.gpu_vram_gb is not None
            and self.min_free_gpu_vram_gb > self.gpu_vram_gb
        ):
            raise ResourceError(
                "remote_compute.min_free_gpu_vram_gb cannot exceed gpu_vram_gb."
            )


@dataclass(frozen=True)
class SearchResources:
    """Resources available while candidates are researched and built.

    ``candidate_api_providers`` is tri-state: ``None`` is unanswered, ``()``
    confirms that no provider access exists, and names identify providers whose
    authentication/access is usable by candidate evaluation. It records
    capabilities only, never credential values.
    """

    cpu_cores: int | None = None
    memory_gb: float | None = None
    disk_gb: float | None = None
    gpu: str | None = None
    gpu_vram_gb: float | None = None
    max_parallel_agents: int = 4
    max_dollars_per_agent_call: float | None = None
    allow_package_installs: bool | None = None
    allow_model_downloads: bool | None = None
    fine_tuning: bool = False
    pi_models: tuple[str, ...] = ()
    pi_local: bool = False
    candidate_api_providers: tuple[str, ...] | None = None
    remote_compute: RemoteCompute | None = None

    def __post_init__(self) -> None:
        _positive("search.cpu_cores", self.cpu_cores)
        _positive("search.memory_gb", self.memory_gb)
        _positive("search.disk_gb", self.disk_gb)
        _positive("search.gpu_vram_gb", self.gpu_vram_gb)
        _positive("search.max_parallel_agents", self.max_parallel_agents, allow_none=False)
        _positive(
            "search.max_dollars_per_agent_call",
            self.max_dollars_per_agent_call,
        )
        object.__setattr__(self, "pi_models", _tuple(self.pi_models))
        if self.candidate_api_providers is not None:
            object.__setattr__(
                self, "candidate_api_providers", _tuple(self.candidate_api_providers)
            )
        remote = self.remote_compute
        if isinstance(remote, dict):
            object.__setattr__(self, "remote_compute", RemoteCompute(**remote))

    @classmethod
    def detect(cls) -> "SearchResources":
        """Conservatively detect local hardware; never infer network consent."""
        cores = os.cpu_count() or 1
        memory_gb = None
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            memory_gb = pages * page_size / (1024 ** 3)
        except (AttributeError, OSError, ValueError):
            pass
        disk_gb = shutil.disk_usage(Path.cwd()).free / (1024 ** 3)

        gpu = None
        vram = None
        if shutil.which("nvidia-smi"):
            try:
                proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=False,
                )
                line = proc.stdout.splitlines()[0] if proc.stdout.splitlines() else ""
                if line:
                    name, _, memory = line.rpartition(",")
                    gpu = name.strip() or "cuda"
                    vram = float(memory.strip()) / 1024 if memory.strip() else None
            except (OSError, ValueError, subprocess.SubprocessError):
                gpu = "cuda"

        current_pi = _current_pi_model_pattern()
        return cls(
            cpu_cores=cores,
            memory_gb=memory_gb,
            disk_gb=disk_gb,
            gpu=gpu,
            gpu_vram_gb=vram,
            max_parallel_agents=max(1, min(4, cores)),
            # Detecting the currently selected Pi model records a capability,
            # never a credential. The confirmed budget still governs its use.
            pi_models=(
                _available_pi_model_patterns(current_pi)
                if current_pi else ()
            ),
            # These require consent and deliberately remain unanswered.
            allow_package_installs=None,
            allow_model_downloads=None,
        )


@dataclass(frozen=True)
class RuntimeResources:
    """The resource envelope of the package that will actually be shipped."""

    network: bool | None = None
    api_providers: tuple[str, ...] = ()
    agent_runtime: bool = False
    gpu: str | None = None
    memory_gb: float | None = None
    disk_gb: float | None = None
    max_dollars_per_call: float | None = None
    max_latency_ms: float | None = None
    max_artifact_mb: float | None = None
    offline: bool = False

    def __post_init__(self) -> None:
        _positive("runtime.memory_gb", self.memory_gb)
        _positive("runtime.disk_gb", self.disk_gb)
        _positive("runtime.max_dollars_per_call", self.max_dollars_per_call)
        _positive("runtime.max_latency_ms", self.max_latency_ms)
        _positive("runtime.max_artifact_mb", self.max_artifact_mb)
        object.__setattr__(self, "api_providers", _tuple(self.api_providers))
        if self.offline and self.network is True:
            raise ResourceError("runtime.offline=True conflicts with runtime.network=True.")
        if self.offline and self.api_providers:
            raise ResourceError(
                "An offline runtime cannot declare external API providers."
            )


@dataclass(frozen=True)
class Resources:
    """Confirmed capabilities and constraints for one optimization run."""

    search: SearchResources = field(default_factory=SearchResources.detect)
    runtime: RuntimeResources = field(default_factory=RuntimeResources)
    data: DataPolicy = field(default_factory=DataPolicy)
    confirmed: bool = False

    @classmethod
    def detect(cls) -> "Resources":
        """Detect hardware and leave consent-sensitive fields unanswered."""
        return cls(search=SearchResources.detect())

    @property
    def questions(self) -> tuple[str, ...]:
        """Questions still requiring a user answer before agentic search."""
        questions: list[str] = []
        if self.data.external_egress is None:
            questions.append("May task examples or derived content leave this machine?")
        if self.runtime.network is None:
            questions.append("May the shipped program use the network at runtime?")
        if self.search.allow_package_installs is None:
            questions.append("May workers install third-party Python packages?")
        if self.search.allow_model_downloads is None:
            questions.append("May workers download pretrained model artifacts?")
        if (
            self.runtime.api_providers
            and self.search.candidate_api_providers is None
            and not self.search.pi_models
        ):
            questions.append(
                "Which runtime API providers are actually available to candidate "
                "evaluations during this search (credentials/access, not secret values)? "
                "An authenticated Pi model may be listed instead; no raw API key "
                "is required for Pi-backed candidates."
            )
        if not self.confirmed:
            questions.append("Confirm this search and deployment resource profile.")
        return tuple(questions)

    def ensure_confirmed(self) -> None:
        if self.questions:
            rendered = "\n".join(f"  - {q}" for q in self.questions)
            raise ResourceError(
                "Resource profile is incomplete. AutoProgramming will not guess "
                "about data egress, package installation, model downloads, or "
                f"deployment networking. Answer and confirm:\n{rendered}"
            )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "Resources":
        search = dict(value.get("search", {}))
        if isinstance(search.get("remote_compute"), dict):
            search["remote_compute"] = RemoteCompute(**search["remote_compute"])
        return cls(
            search=SearchResources(**search),
            runtime=RuntimeResources(**value.get("runtime", {})),
            data=DataPolicy(**value.get("data", {})),
            confirmed=bool(value.get("confirmed", False)),
        )

    def with_current_pi_model(self) -> "Resources":
        """Record the active host Pi model as an authenticated capability.

        Explicit ``search.pi_models`` remain authoritative. When they are
        empty and this code is running from a live Pi session, the exact active
        provider/model/thinking tuple is recorded first, followed by available
        models from that authenticated provider's Pi registry. OAuth stays in
        Pi's auth store and is never copied here.
        """
        current = _current_pi_model_pattern()
        if not current or self.search.pi_models:
            return self
        return replace(
            self,
            search=replace(
                self.search,
                pi_models=_available_pi_model_patterns(current),
            ),
        )

    @property
    def pi_may_receive_task_data(self) -> bool:
        """Whether Pi workers may legally inspect examples and task context."""
        return self.search.pi_local or self.data.external_egress is True

    def feasibility(self) -> dict[int, dict[str, str | bool]]:
        """Search feasibility plus deployment fit for the approach ladder.

        A Pi subscription is a real search/evaluation capability even when no
        raw provider API key exists. Such an avenue remains visible and is
        labelled as requiring Pi at runtime; the user chooses among those
        tradeoffs rather than having the controller silently prune it.
        """
        runtime_network = self.runtime.network is True and not self.runtime.offline
        runtime_apis = set(self.runtime.api_providers)
        search_apis = set(self.search.candidate_api_providers or ())
        usable_apis = runtime_apis & search_apis
        APIs = bool(usable_apis)
        pi_models = bool(self.search.pi_models)
        can_externalize = self.data.external_egress is True
        pi_data_access = can_externalize or self.search.pi_local
        install = self.search.allow_package_installs is True
        download = self.search.allow_model_downloads is True
        remote = self.search.remote_compute
        deployment_memory = self.runtime.memory_gb or 0
        deployment_disk = self.runtime.disk_gb or 0
        search_gpu = self.search.gpu or (remote.gpu if remote else None)
        search_memory = max(
            self.search.memory_gb or 0,
            remote.memory_gb if remote and remote.memory_gb else 0,
        )
        search_disk = max(
            self.search.disk_gb or 0,
            remote.disk_gb if remote and remote.disk_gb else 0,
        )

        def item(
            ok: bool, reason: str, *, deployable: bool | None = None
        ) -> dict[str, str | bool]:
            return {
                "feasible": ok,
                "deployable": ok if deployable is None else bool(deployable),
                "reason": reason,
            }

        result = {
            1: item(
                (pi_models and pi_data_access)
                or (can_externalize and runtime_network and self.runtime.agent_runtime),
                "requires a live Pi/generalist runtime or another runtime agent; "
                "Pi subscription-backed variants are reported even when they are "
                "outside the preferred deployment envelope",
                deployable=(
                    runtime_network and self.runtime.agent_runtime and can_externalize
                ),
            ),
            2: item(
                (can_externalize and APIs) or (pi_models and pi_data_access),
                "requires model calls through either a confirmed candidate API or "
                "an authenticated Pi subscription; Pi-backed variants require Pi "
                "and its logged-in model at deployment",
                deployable=(runtime_network and APIs and can_externalize),
            ),
            3: item(
                (can_externalize and APIs) or (pi_models and pi_data_access),
                "requires one model call through either a confirmed candidate API "
                "or an authenticated Pi subscription; Pi-backed variants require "
                "Pi and its logged-in model at deployment",
                deployable=(runtime_network and APIs and can_externalize),
            ),
            4: item(
                self.search.fine_tuning
                and (
                    (APIs and can_externalize)
                    or (pi_models and pi_data_access)
                    or search_gpu is not None
                    or search_disk > 0
                ),
                "requires fine-tuning access plus a search-time endpoint or "
                "sufficient local/remote build compute",
                deployable=(
                    (runtime_network and APIs and can_externalize)
                    or self.runtime.gpu is not None
                    or deployment_disk > 0
                ),
            ),
            5: item(
                install and download and (
                    search_gpu is not None
                    or search_memory >= 2
                    or search_disk >= 2
                ),
                "requires package/model downloads and sufficient search compute; "
                "remote search hardware is not assumed available at deployment",
                deployable=(
                    self.runtime.gpu is not None
                    or deployment_memory >= 2
                    or deployment_disk >= 2
                ),
            ),
            6: item(
                install,
                "requires permission to install classical-ML dependencies",
            ),
            7: item(True, "stdlib algorithms, rules, and feature engineering are always feasible"),
        }
        result[8] = item(
            sum(bool(v["feasible"]) for k, v in result.items() if k <= 7) >= 2,
            "composition requires at least two feasible implementation families",
            deployable=(
                sum(bool(v["deployable"]) for k, v in result.items() if k <= 7)
                >= 2
            ),
        )
        return result
