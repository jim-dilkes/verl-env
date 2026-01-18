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
