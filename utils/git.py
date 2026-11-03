"""Lightweight git inspection helpers for checkpoint metadata.

This module exposes a single best-effort helper that records the git SHA of
the working tree at training time. It is intentionally non-fatal: any git
failure (no repo, no executable, detached HEAD, network drive, etc.) yields
``None`` rather than raising, so a checkpoint can still be saved on machines
without git installed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def get_git_sha(repo_root: Path | str | None = None) -> str | None:
    """Return the git SHA of *repo_root*'s HEAD, or ``None`` on any failure.

    The returned value is suffixed with ``"-dirty"`` when the working tree
    has uncommitted modifications, matching common conventions used by build
    systems for reproducibility tags.

    Args:
        repo_root (Path | str | None): Repository root. When ``None`` the
            current working directory is used.

    Returns:
        str | None: ``"<40-char-sha>"`` or ``"<sha>-dirty"`` when extraction
        succeeds; ``None`` when git is unavailable or the directory is not a
        git repository.
    """
    cwd = str(repo_root) if repo_root is not None else None
    try:
        sha_bytes = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=cwd,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None

    sha = sha_bytes.decode().strip()
    if not sha:
        return None

    try:
        status_bytes = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            cwd=cwd,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return sha

    is_dirty = bool(status_bytes.decode().strip())
    return f"{sha}-dirty" if is_dirty else sha
