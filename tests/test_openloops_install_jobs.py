"""Unit tests for the async install-job manager.

These tests do NOT actually run an OL build — they monkey-patch
`install_process_library` so the worker thread either succeeds quickly,
fails on demand, or sleeps long enough to be polled while running.
"""
from __future__ import annotations

import time
from unittest.mock import patch

import pytest

import feynman_engine.amplitudes.openloops_install_jobs as jobs


@pytest.fixture(autouse=True)
def _reset_jobs():
    """Clear the module-level job dict between tests."""
    with jobs._LOCK:
        jobs._JOBS.clear()
    yield
    with jobs._LOCK:
        jobs._JOBS.clear()


def _fake_install_ok(library: str) -> None:
    """Fast no-op stand-in for install_process_library."""
    print(f"[fake-install] {library} ok")


def _fake_install_slow(library: str) -> None:
    """Sleep so we can observe the running state."""
    time.sleep(0.3)
    print(f"[fake-install] {library} ok after delay")


def _fake_install_fail(library: str) -> None:
    raise RuntimeError(f"compile failed for {library}")


def test_already_installed_short_circuits_without_starting_thread():
    """If installed_processes() reports the library, no thread is spawned."""
    with patch(
        "feynman_engine.amplitudes.openloops_bridge.installed_processes",
        return_value=["pptt"],
    ) as _:
        snap = jobs.start_install("pptt")
    assert snap["state"] == "already-installed"
    assert jobs.get_status("pptt") is None  # no job tracked


def test_install_succeeds_and_completes():
    with patch(
        "feynman_engine.amplitudes.openloops_bridge.installed_processes",
        return_value=[],
    ), patch(
        "feynman_engine.openloops.install_process_library",
        side_effect=_fake_install_ok,
    ):
        snap = jobs.start_install("pphh2")
        assert snap["state"] == "running"
        # Wait for the worker thread (≤ 1s)
        for _ in range(50):
            time.sleep(0.05)
            cur = jobs.get_status("pphh2")
            if cur["state"] != "running":
                break
        cur = jobs.get_status("pphh2")
        assert cur["state"] == "completed"
        assert cur["error"] is None
        assert cur["finished_at"] is not None
        assert cur["elapsed_s"] >= 0.0
        assert "[fake-install] pphh2 ok" in cur["log_tail"]


def test_install_failure_records_error():
    with patch(
        "feynman_engine.amplitudes.openloops_bridge.installed_processes",
        return_value=[],
    ), patch(
        "feynman_engine.openloops.install_process_library",
        side_effect=_fake_install_fail,
    ):
        jobs.start_install("ppvvjj")
        for _ in range(50):
            time.sleep(0.05)
            cur = jobs.get_status("ppvvjj")
            if cur["state"] != "running":
                break
        cur = jobs.get_status("ppvvjj")
        assert cur["state"] == "failed"
        assert "compile failed" in (cur["error"] or "")
        assert "Traceback" in cur["log_tail"]


def test_running_state_observable_during_install():
    with patch(
        "feynman_engine.amplitudes.openloops_bridge.installed_processes",
        return_value=[],
    ), patch(
        "feynman_engine.openloops.install_process_library",
        side_effect=_fake_install_slow,
    ):
        snap = jobs.start_install("ppwjj")
        assert snap["state"] == "running"
        # Poll once mid-flight: should still be running
        time.sleep(0.05)
        mid = jobs.get_status("ppwjj")
        assert mid["state"] == "running"
        assert mid["elapsed_s"] > 0
        # And eventually finishes.
        for _ in range(50):
            time.sleep(0.05)
            cur = jobs.get_status("ppwjj")
            if cur["state"] != "running":
                break
        assert jobs.get_status("ppwjj")["state"] == "completed"


def test_double_install_returns_same_running_job():
    """Two POSTs to the same library while install is running → same job."""
    with patch(
        "feynman_engine.amplitudes.openloops_bridge.installed_processes",
        return_value=[],
    ), patch(
        "feynman_engine.openloops.install_process_library",
        side_effect=_fake_install_slow,
    ):
        first = jobs.start_install("eett_ew")
        second = jobs.start_install("eett_ew")
        # Same job (started_at equal), still running
        assert first["state"] == "running"
        assert second["state"] == "running"
        assert first["started_at"] == second["started_at"]


def test_list_jobs_returns_all_tracked():
    with patch(
        "feynman_engine.amplitudes.openloops_bridge.installed_processes",
        return_value=[],
    ), patch(
        "feynman_engine.openloops.install_process_library",
        side_effect=_fake_install_ok,
    ):
        jobs.start_install("a_lib")
        jobs.start_install("b_lib")
        # Wait for both to finish
        for _ in range(50):
            time.sleep(0.05)
            if all(j["state"] != "running" for j in jobs.list_jobs()):
                break
        snaps = jobs.list_jobs()
        names = sorted(s["library"] for s in snaps)
        assert names == ["a_lib", "b_lib"]
