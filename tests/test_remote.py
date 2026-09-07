"""Optional remote-compute placement and GPU concurrency gates."""

from __future__ import annotations

import threading
import time

import autoprogramming as ap
from autoprogramming.portfolio import ApproachTier, default_avenue
from autoprogramming.remote import (
    RemoteAdmission,
    candidate_placement,
    gpu_environment_prefix,
    is_gpu_avenue,
    record_candidate_placement,
    use_remote_for_avenue,
)


def resources(remote):
    return ap.Resources(
        search=ap.SearchResources(
            allow_package_installs=True,
            allow_model_downloads=True,
            remote_compute=remote,
        ),
        runtime=ap.RuntimeResources(network=False, memory_gb=8),
        data=ap.DataPolicy(external_egress=False),
        confirmed=True,
    )


def test_selected_cuda_device_is_propagated_to_remote_jobs():
    target = ap.RemoteCompute(
        endpoint="gpu-box", transport="ssh", gpu="cuda:3"
    )
    assert gpu_environment_prefix(target) == "CUDA_VISIBLE_DEVICES=3"


def test_remote_gpu_gets_conservative_default_vram_admission_threshold():
    target = ap.RemoteCompute(
        endpoint="gpu-box", transport="ssh", gpu="cuda", gpu_vram_gb=40
    )
    assert target.min_free_gpu_vram_gb == 32


def test_candidate_placement_journal_round_trips(tmp_path):
    workspace = type("Workspace", (), {"root": tmp_path})()
    record_candidate_placement(workspace, "candidate_3", "remote")
    assert candidate_placement(workspace, "candidate_3") == "remote"
    assert candidate_placement(workspace, "candidate_4") is None


def test_remote_placement_is_selective_not_everything_remote():
    target = ap.RemoteCompute(endpoint="gpu-box", transport="ssh", gpu="cuda")
    profile = resources(target)
    assert use_remote_for_avenue(
        default_avenue(ApproachTier.SPECIALIZED_DEEP_MODEL, profile)
    )
    assert use_remote_for_avenue(
        default_avenue(ApproachTier.CLASSICAL_ML, profile)
    )
    assert not use_remote_for_avenue(
        default_avenue(ApproachTier.SINGLE_MODEL_CALL, profile)
    )
    assert not use_remote_for_avenue(
        default_avenue(ApproachTier.CODE_AND_RULES, profile)
    )


def test_only_model_heavy_avenues_take_gpu_lease_by_default():
    target = ap.RemoteCompute(endpoint="gpu-box", transport="ssh", gpu="cuda", gpu_vram_gb=48)
    profile = resources(target)
    assert is_gpu_avenue(
        default_avenue(ApproachTier.SPECIALIZED_DEEP_MODEL, profile)
    )
    assert not is_gpu_avenue(
        default_avenue(ApproachTier.CODE_AND_RULES, profile)
    )


def test_remote_gpu_avenues_are_serialized_by_default(monkeypatch):
    target = ap.RemoteCompute(
        endpoint="unique-gpu-box-for-test",
        transport="ssh",
        gpu="cuda",
        gpu_vram_gb=48,
        max_parallel_gpu_jobs=1,
    )
    profile = resources(target)
    spec = default_avenue(ApproachTier.SPECIALIZED_DEEP_MODEL, profile)
    admission = RemoteAdmission(target)
    monkeypatch.setattr(admission.executor, "wait_for_gpu", lambda *_a, **_k: None)
    active = 0
    maximum = 0
    guard = threading.Lock()

    def work():
        nonlocal active, maximum
        with admission.lease(spec):
            with guard:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.03)
            with guard:
                active -= 1

    threads = [threading.Thread(target=work) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert maximum == 1
