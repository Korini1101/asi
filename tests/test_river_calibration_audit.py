"""The RiverSwim audit must fail closed on missing source provenance."""

import importlib.util
import json
from pathlib import Path

import pytest


def _audit_module():
    path = (
        Path(__file__).parents[1]
        / "outputs/reference_life_scorecard_development"
        / "river-calibration-audit-2026-10-14/analyze.py"
    )
    spec = importlib.util.spec_from_file_location("river_calibration_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load RiverSwim audit module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_records_rejects_missing_source_identity(tmp_path: Path) -> None:
    audit = _audit_module()
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    payload = {
        "schema": "asi.reference_life_scorecard.run.v1",
        "status": "completed",
        "failure": None,
        "environment_kind": "riverswim",
        "arm": "random",
        "seed": 1,
        "plan_sha256": audit.EXPECTED_PLAN_SHA256,
        "evidence_policy": {"permanently_nonpromoting": True},
        "outcome": {"accepted_events": 20000},
    }
    path = shard_dir / "riverswim__random__1.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="invalid source identity"):
        audit.load_records(run=tmp_path, arms=("random",), seeds=(1,))
