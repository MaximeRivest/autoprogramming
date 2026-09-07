"""Optional remote search compute and GPU admission control.

Remote execution is capability-driven: nothing in the skill assumes a remote
machine.  When ``SearchResources.remote_compute`` is present, implementation
workers and evaluation sessions may stage their isolated directories there.
The human-facing Pi session and trusted portfolio bookkeeping remain local.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shlex
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .errors import RunnerError
from .portfolio import ApproachTier, AvenueSpec
from .resources import RemoteCompute, Resources


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def gpu_environment_prefix(config: RemoteCompute) -> str:
    """A shell assignment for an explicitly selected CUDA device, if any."""
    gpu = str(config.gpu or "").strip().casefold()
    if gpu.startswith("cuda:") and gpu.split(":", 1)[1].isdigit():
        return f"CUDA_VISIBLE_DEVICES={gpu.split(':', 1)[1]}"
    return ""


def use_remote_for_avenue(spec: AvenueSpec) -> bool:
    """Place build-heavy families remotely; keep lightweight/API work local."""
    if spec.tier in (
        ApproachTier.FINETUNED_MODEL,
        ApproachTier.SPECIALIZED_DEEP_MODEL,
        ApproachTier.CLASSICAL_ML,
        ApproachTier.COMPOSITION,
    ):
        return True
    heavy = " ".join(
        (*spec.required_capabilities, *spec.required_mechanisms)
    ).casefold()
    return any(token in heavy for token in ("gpu", "model-download", "fine-tun"))


def is_gpu_avenue(spec: AvenueSpec) -> bool:
    """Whether an avenue should hold a GPU lease for its whole worker turn."""
    if spec.tier in (
        ApproachTier.FINETUNED_MODEL,
        ApproachTier.SPECIALIZED_DEEP_MODEL,
    ):
        return True
    text = " ".join((
        *spec.required_capabilities,
        *spec.runtime_requirements,
        spec.mechanism,
    )).lower()
    return any(token in text for token in ("gpu", "cuda", "vram"))


def _placement_path(workspace) -> Path:
    return Path(workspace.root) / ".ap" / "controller" / "compute-placement.json"


def record_candidate_placement(workspace, candidate_name: str, placement: str) -> None:
    """Persist the controller's avenue-aware local/remote placement decision."""
    if placement not in ("local", "remote"):
        raise ValueError("placement must be 'local' or 'remote'.")
    path = _placement_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        values = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError, TypeError):
        values = {}
    values[str(candidate_name)] = placement
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def candidate_placement(workspace, candidate_name: str) -> str | None:
    path = _placement_path(workspace)
    try:
        value = json.loads(path.read_text()).get(str(candidate_name))
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    return value if value in ("local", "remote") else None


def load_remote_compute(workspace) -> RemoteCompute | None:
    """Read an optional search-time target from a workspace resource record."""
    path = getattr(workspace, "resources_json", Path(workspace.root) / "resources.json")
    path = Path(path)
    if not path.exists():
        return None
    try:
        return Resources.from_dict(json.loads(path.read_text())).search.remote_compute
    except (OSError, ValueError, TypeError, KeyError):
        return None


@dataclass(frozen=True)
class GpuSnapshot:
    free_gb: tuple[float, ...]
    total_gb: tuple[float, ...]

    @property
    def best_free_gb(self) -> float:
        return max(self.free_gb, default=0.0)


class RemoteExecutor:
    """Small SSH transport used only when the user supplied one explicitly."""

    def __init__(self, config: RemoteCompute, *, timeout: float = 120.0):
        self.config = config
        self.timeout = float(timeout)
        self._base: str | None = None

    def ssh(
        self,
        command: str,
        *,
        input_bytes: bytes | None = None,
        timeout: float | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess:
        try:
            proc = subprocess.run(
                ["ssh", self.config.endpoint, command],
                input=input_bytes,
                capture_output=True,
                timeout=self.timeout if timeout is None else timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RunnerError(
                "Remote compute was provided, but the ssh executable was not found."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RunnerError(
                f"SSH operation on {self.config.endpoint!r} timed out."
            ) from exc
        if check and proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", "replace")[-2000:]
            raise RunnerError(
                f"Remote compute command failed on {self.config.endpoint!r} "
                f"(exit {proc.returncode}): {detail}"
            )
        return proc

    @property
    def base_dir(self) -> str:
        if self._base is not None:
            return self._base
        if self.config.workdir:
            base = self.config.workdir.rstrip("/")
        else:
            proc = self.ssh("pwd")
            home = proc.stdout.decode("utf-8", "replace").strip()
            if not home:
                raise RunnerError("Remote compute returned an empty working directory.")
            base = f"{home}/.cache/autoprogramming"
        self.ssh(f"mkdir -p {_q(base)}")
        self._base = base
        return base

    def staged_root(self, local_root: str | Path) -> str:
        """Stable remote owner directory for one local workspace/task root."""
        identity = hashlib.sha256(
            str(Path(local_root).resolve()).encode("utf-8")
        ).hexdigest()[:20]
        return f"{self.base_dir}/{identity}"

    def staged_dir(self, local_root: str | Path, *, namespace: str = "work") -> str:
        safe_namespace = "".join(
            ch if ch.isalnum() or ch in "-_" else "-" for ch in namespace
        ).strip("-") or "work"
        return f"{self.staged_root(local_root)}/{safe_namespace}"

    def sync_to(
        self,
        local_root: str | Path,
        remote_root: str,
        *,
        excludes: tuple[str, ...] = (),
    ) -> None:
        """Replace a remote tree, optionally omitting controller-private files."""
        local = Path(local_root).resolve()
        if not local.is_dir():
            raise RunnerError(f"Cannot stage missing local directory {local}.")
        command = (
            f"rm -rf {_q(remote_root)} && mkdir -p {_q(remote_root)} && "
            f"tar -xzf - -C {_q(remote_root)}"
        )
        try:
            tar_args = [
                "tar", "-czf", "-", "--exclude=.pi-sessions", "--exclude=.tools",
                *(f"--exclude={item}" for item in excludes),
                ".",
            ]
            tar = subprocess.Popen(
                tar_args,
                cwd=str(local),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert tar.stdout is not None
            ssh = subprocess.Popen(
                ["ssh", self.config.endpoint, command],
                stdin=tar.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            tar.stdout.close()
            ssh_out, ssh_err = ssh.communicate(timeout=self.timeout)
            tar_err = tar.stderr.read() if tar.stderr is not None else b""
            tar_code = tar.wait(timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            with contextlib.suppress(Exception):
                tar.kill()  # type: ignore[possibly-undefined]
            with contextlib.suppress(Exception):
                ssh.kill()  # type: ignore[possibly-undefined]
            raise RunnerError(f"Failed to stage remote compute directory: {exc}") from exc
        if tar_code != 0 or ssh.returncode != 0:
            detail = (tar_err + ssh_err + ssh_out).decode("utf-8", "replace")[-2000:]
            raise RunnerError(f"Failed to stage remote compute directory: {detail}")

    def sync_from(
        self,
        remote_root: str,
        local_root: str | Path,
        *,
        excludes: tuple[str, ...] = (),
    ) -> None:
        """Pull a worker tree transactionally, including remote deletions."""
        local = Path(local_root).resolve()
        local.mkdir(parents=True, exist_ok=True)
        exclude_args = " ".join(
            f"--exclude={_q(item)}" for item in excludes
        )
        remote_command = (
            f"test -d {_q(remote_root)} && tar -czf - {exclude_args} "
            f"-C {_q(remote_root)} ."
        )
        with tempfile.TemporaryDirectory(
            prefix="ap-remote-pull-", dir=str(local.parent)
        ) as temp_name:
            extracted = Path(temp_name)
            try:
                ssh = subprocess.Popen(
                    ["ssh", self.config.endpoint, remote_command],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                assert ssh.stdout is not None
                tar = subprocess.Popen(
                    ["tar", "-xzf", "-"],
                    cwd=str(extracted),
                    stdin=ssh.stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                ssh.stdout.close()
                tar_out, tar_err = tar.communicate(timeout=self.timeout)
                ssh_err = ssh.stderr.read() if ssh.stderr is not None else b""
                ssh_code = ssh.wait(timeout=5)
            except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
                with contextlib.suppress(Exception):
                    ssh.kill()  # type: ignore[possibly-undefined]
                with contextlib.suppress(Exception):
                    tar.kill()  # type: ignore[possibly-undefined]
                raise RunnerError(
                    f"Failed to retrieve remote worker files: {exc}"
                ) from exc
            if ssh_code != 0 or tar.returncode != 0:
                detail = (ssh_err + tar_err + tar_out).decode(
                    "utf-8", "replace"
                )[-2000:]
                raise RunnerError(
                    f"Failed to retrieve remote worker files: {detail}"
                )
            # Extraction succeeded; now mirror it. Keep local-only guard/session
            # infrastructure, but remove stale worker outputs before replacement.
            for child in local.iterdir():
                if child.name in (".tools", ".pi-sessions"):
                    continue
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink(missing_ok=True)
            for child in extracted.iterdir():
                shutil.move(str(child), str(local / child.name))

    def gpu_snapshot(self) -> GpuSnapshot:
        gpu = str(self.config.gpu or "").strip().casefold()
        selected = ""
        if gpu.startswith("cuda:") and gpu.split(":", 1)[1].isdigit():
            selected = f" --id={gpu.split(':', 1)[1]}"
        proc = self.ssh(
            "nvidia-smi" + selected + " --query-gpu=memory.free,memory.total "
            "--format=csv,noheader,nounits",
            check=False,
        )
        if proc.returncode != 0:
            return GpuSnapshot((), ())
        free: list[float] = []
        total: list[float] = []
        for line in proc.stdout.decode("utf-8", "replace").splitlines():
            try:
                f, t = (float(part.strip()) / 1024.0 for part in line.split(",", 1))
            except (ValueError, TypeError):
                continue
            free.append(f)
            total.append(t)
        return GpuSnapshot(tuple(free), tuple(total))

    def wait_for_gpu(
        self,
        required_free_gb: float | None,
        *,
        timeout: float = 900.0,
        poll_s: float = 5.0,
    ) -> GpuSnapshot:
        """Wait for user/other avenues to release enough remote VRAM."""
        threshold = float(required_free_gb or 0.0)
        deadline = time.monotonic() + timeout
        while True:
            snapshot = self.gpu_snapshot()
            # No nvidia-smi result means there is no measurable admission gate;
            # the faithful implementation can report a precise setup failure.
            gpu = str(self.config.gpu or "").strip().casefold()
            available = (
                snapshot.free_gb[0]
                if gpu == "cuda" and snapshot.free_gb
                else snapshot.best_free_gb
            )
            if not snapshot.free_gb or available >= threshold:
                return snapshot
            if time.monotonic() >= deadline:
                raise RunnerError(
                    f"Remote GPUs on {self.config.endpoint!r} stayed below the "
                    f"required {threshold:g} GiB free VRAM for {timeout:g}s. "
                    "The avenue was paused rather than treated as a mechanism failure."
                )
            time.sleep(max(0.05, poll_s))


_GPU_LOCKS: dict[str, threading.BoundedSemaphore] = {}
_GPU_EXCLUSIVE_LOCKS: dict[str, threading.Lock] = {}
_GPU_LOCKS_GUARD = threading.Lock()


class RemoteAdmission:
    """Per-target CPU/GPU concurrency gates shared by worker threads."""

    def __init__(self, config: RemoteCompute | None):
        self.config = config
        self.executor = RemoteExecutor(config) if config else None
        self._cpu = threading.BoundedSemaphore(
            config.max_parallel_cpu_jobs if config and config.max_parallel_cpu_jobs else 10_000
        )
        if config:
            key = (
                f"{config.transport}:{config.endpoint}:"
                f"gpu-slots={config.max_parallel_gpu_jobs}"
            )
            with _GPU_LOCKS_GUARD:
                self._gpu = _GPU_LOCKS.setdefault(
                    key,
                    threading.BoundedSemaphore(config.max_parallel_gpu_jobs),
                )
                self._gpu_exclusive = _GPU_EXCLUSIVE_LOCKS.setdefault(
                    key, threading.Lock()
                )
            self._gpu_slots = config.max_parallel_gpu_jobs
        else:
            self._gpu = threading.BoundedSemaphore(10_000)
            self._gpu_exclusive = threading.Lock()
            self._gpu_slots = 1

    @contextlib.contextmanager
    def lease(self, spec: AvenueSpec, *, exclusive: bool = False):
        if self.config is None or not use_remote_for_avenue(spec):
            yield
            return
        gpu_job = is_gpu_avenue(spec)
        gate = self._gpu if gpu_job else self._cpu
        permits = self._gpu_slots if gpu_job and exclusive else 1
        holds_exclusive_guard = False
        if gpu_job:
            if exclusive:
                self._gpu_exclusive.acquire()
                holds_exclusive_guard = True
                for _ in range(permits):
                    gate.acquire()
            else:
                # Do not admit new ordinary jobs while an exclusive retry is
                # draining existing leases.
                with self._gpu_exclusive:
                    gate.acquire()
        else:
            gate.acquire()
        try:
            if gpu_job and self.executor is not None:
                self.executor.wait_for_gpu(self.config.min_free_gpu_vram_gb)
            yield
        finally:
            for _ in range(permits):
                gate.release()
            if holds_exclusive_guard:
                self._gpu_exclusive.release()
