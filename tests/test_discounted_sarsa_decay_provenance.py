"""The RiverSwim decay diagnostic must bind records to its pinned source."""

import importlib.util
from pathlib import Path

import pytest


def _provenance_module():
    path = (
        Path(__file__).parents[1]
        / "outputs/reference_life_control_calibration"
        / "discounted-sarsa-decay-2026-10-15/provenance.py"
    )
    spec = importlib.util.spec_from_file_location("riverswim_decay_provenance", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load RiverSwim provenance module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_commit_guard_rejects_mismatched_checkout(monkeypatch, tmp_path: Path) -> None:
    provenance = _provenance_module()
    monkeypatch.setattr(
        provenance.subprocess,
        "check_output",
        lambda *args, **kwargs: "actual\n",
    )

    with pytest.raises(RuntimeError, match="does not match pinned plan commit"):
        provenance.require_source_commit(tmp_path, "expected")


def test_source_commit_guard_accepts_pinned_checkout(monkeypatch, tmp_path: Path) -> None:
    provenance = _provenance_module()
    monkeypatch.setattr(
        provenance.subprocess,
        "check_output",
        lambda *args, **kwargs: "expected\n",
    )

    assert provenance.require_source_commit(tmp_path, "expected") == "expected"


def test_record_source_commit_guard_rejects_drift() -> None:
    provenance = _provenance_module()

    with pytest.raises(ValueError, match="source commit drifted.json"):
        provenance.require_record_source_commit("actual", "expected", "drifted.json")
