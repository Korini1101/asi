"""Fail-closed source identity checks for the RiverSwim diagnostic."""

from __future__ import annotations

import subprocess
from pathlib import Path


def require_source_commit(root: Path, expected: str) -> str:
    """Return the checkout commit when it matches the pinned plan commit."""

    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if actual != expected:
        raise RuntimeError(
            f"source commit {actual} does not match pinned plan commit {expected}"
        )
    return actual


def require_record_source_commit(actual: object, expected: str, label: str) -> None:
    """Reject a raw record whose source identity differs from the plan."""

    if actual != expected:
        raise ValueError(
            f"{label} source commit {actual} does not match pinned plan commit {expected}"
        )
