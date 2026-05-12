"""Background-thread manager for ``install-process`` API jobs.

Single-process design: all state lives in a module-level dict guarded by a
lock.  This is fine for a single-worker uvicorn deployment (the typical
"local calculator" workflow); multi-worker deployments would need a
shared store (Redis, file lock, etc.) which is out of scope.

Public API
----------
``start_install(library)``  → kicks off install in a background thread,
                               returns the job snapshot.
``get_status(library)``     → returns current snapshot (running / completed
                               / failed) including elapsed time and log tail.
``list_jobs()``             → all current job snapshots.
"""
from __future__ import annotations

import io
import threading
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class _Job:
    """Internal job state (mutable, guarded by ``_LOCK``)."""
    library:     str
    state:       str = "running"   # running | completed | failed | already-installed
    started_at:  float = 0.0
    finished_at: Optional[float] = None
    error:       Optional[str] = None
    log:         io.StringIO = field(default_factory=io.StringIO)
    thread:      Optional[threading.Thread] = None


_LOCK = threading.Lock()
_JOBS: dict[str, _Job] = {}


def _snapshot(job: _Job) -> dict:
    """Convert a _Job into a JSON-friendly dict."""
    log_text = job.log.getvalue()
    return {
        "library":     job.library,
        "state":       job.state,
        "started_at":  job.started_at,
        "finished_at": job.finished_at,
        "elapsed_s":   (
            (job.finished_at if job.finished_at else time.time()) - job.started_at
            if job.started_at else 0.0
        ),
        "error":       job.error,
        "log_tail":    log_text[-4000:],   # last ~4 KB of stdout/stderr
        "log_chars":   len(log_text),
    }


def _run_install(library: str) -> None:
    """Worker: invoke the OpenLoops installer and capture its output."""
    from feynman_engine.openloops import install_process_library
    job = _JOBS[library]
    try:
        # Capture stdout+stderr so the user can see compile progress in the
        # status response.  install_process_library shells out to scons; the
        # subprocess output won't be redirected by Python's redirect_stdout,
        # but Python-level prints from the wrapper will be.
        with redirect_stdout(job.log), redirect_stderr(job.log):
            install_process_library(library)
        with _LOCK:
            job.state = "completed"
            job.finished_at = time.time()
    except Exception as exc:
        with _LOCK:
            job.state = "failed"
            job.finished_at = time.time()
            job.error = f"{type(exc).__name__}: {exc}"
            job.log.write("\n" + traceback.format_exc())


def start_install(library: str) -> dict:
    """Begin (or rejoin) an install job for ``library``.

    Returns the current snapshot — for an already-running or completed job
    the snapshot reports that state instead of starting a new install.
    """
    from feynman_engine.amplitudes.openloops_bridge import installed_processes

    # Already installed → no-op, return short-circuit snapshot.
    if library in installed_processes():
        return {
            "library":   library,
            "state":     "already-installed",
            "elapsed_s": 0.0,
            "started_at": time.time(),
            "finished_at": time.time(),
            "error":     None,
            "log_tail":  "",
            "log_chars": 0,
        }

    with _LOCK:
        existing = _JOBS.get(library)
        if existing and existing.state == "running":
            return _snapshot(existing)
        # Either no prior job, or prior job finished — start a fresh one.
        job = _Job(library=library, started_at=time.time())
        _JOBS[library] = job
        thread = threading.Thread(
            target=_run_install,
            args=(library,),
            name=f"ol-install-{library}",
            daemon=True,
        )
        job.thread = thread
        thread.start()
        return _snapshot(job)


def get_status(library: str) -> Optional[dict]:
    """Snapshot of a job, or None if no job exists."""
    with _LOCK:
        job = _JOBS.get(library)
        if job is None:
            return None
        return _snapshot(job)


def list_jobs() -> list[dict]:
    """All current job snapshots (running + finished)."""
    with _LOCK:
        return [_snapshot(j) for j in _JOBS.values()]
