"""Local runtime patches loaded automatically by Python's `site` module.

This repository is frequently run under Python 3.12 with the third-party
`multiprocess` package (pulled in via `datasets`). `multiprocess==0.70.18`
contains a CPython-compatibility bug where it calls `threading.RLock._recursion_count()`
(which does not exist on Python 3.12), causing noisy shutdown-time tracebacks.

This file applies a small monkey patch to keep exits clean.
"""

from __future__ import annotations

import os
import sys


def _patch_multiprocess_resource_tracker() -> None:
    if sys.version_info < (3, 12):
        return

    try:
        import multiprocess  # type: ignore
        import multiprocess.resource_tracker as mrt  # type: ignore
    except Exception:
        return

    # Only patch versions known to be affected in this environment.
    version = getattr(multiprocess, "__version__", "")
    if version not in {"0.70.18"}:
        return

    ResourceTracker = getattr(mrt, "ResourceTracker", None)
    if ResourceTracker is None:
        return

    # Idempotency: don't patch multiple times.
    if getattr(ResourceTracker._stop_locked, "__name__", "") == "_stop_locked_compat":
        return

    def _stop_locked_compat(
        self,
        close=os.close,
        waitpid=os.waitpid,
        waitstatus_to_exitcode=os.waitstatus_to_exitcode,
    ):
        lock = getattr(self, "_lock", None)
        recursion_count = 1
        if lock is not None:
            rc_attr = getattr(lock, "_recursion_count", None)
            if callable(rc_attr):
                try:
                    recursion_count = int(rc_attr())
                except Exception:
                    recursion_count = 1

        if recursion_count > 1:
            return self._reentrant_call_error()

        if getattr(self, "_fd", None) is None:
            return
        if getattr(self, "_pid", None) is None:
            return

        close(self._fd)
        self._fd = None

        waitpid(self._pid, 0)
        self._pid = None

    ResourceTracker._stop_locked = _stop_locked_compat


_patch_multiprocess_resource_tracker()


def _patch_wandb_service_teardown() -> None:
    """Silence noisy BrokenPipeError during W&B atexit teardown.

    In some multi-process / Ray-style runs, the wandb-core service can be gone
    by interpreter shutdown time. W&B's atexit hook then calls
    `ServiceConnection.teardown()`, which may raise `BrokenPipeError: [Errno 32]`.

    The exception is explicitly "ignored" by atexit, but Python still prints a
    traceback to stderr. This patch makes teardown best-effort and quiet.
    """

    try:
        from wandb.sdk.lib.service import service_connection as sc  # type: ignore
    except Exception:
        return

    ServiceConnection = getattr(sc, "ServiceConnection", None)
    if ServiceConnection is None:
        return

    original_teardown = getattr(ServiceConnection, "teardown", None)
    if original_teardown is None:
        return

    # Idempotency: don't patch multiple times.
    if getattr(original_teardown, "__name__", "") == "teardown_compat":
        return

    def teardown_compat(self, exit_code: int):
        try:
            return original_teardown(self, exit_code)
        except BrokenPipeError:
            # Service already down; try to join if we own the process.
            try:
                proc = getattr(self, "_proc", None)
                return proc.join() if proc else None
            except Exception:
                return None
        except OSError as e:
            if getattr(e, "errno", None) == 32:
                try:
                    proc = getattr(self, "_proc", None)
                    return proc.join() if proc else None
                except Exception:
                    return None
            raise

    ServiceConnection.teardown = teardown_compat


_patch_wandb_service_teardown()
