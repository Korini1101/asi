"""Read-only validation of the retained hosted 144-shard campaign."""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from alberta_framework.benchmarks.reference_life_scorecard import (
    ReferenceLifeDevelopmentPlan, load_json_strict, summarize_shard_files,
    validate_scorecard_artifact,
)

root = Path(sys.argv[1])
receipt = load_json_strict(root / 'run-receipt.v1.json')
assert receipt['schema'] == 'asi.reference_life_scorecard.github_run.v1'
assert receipt['policy'] == 'development_only_permanently_nonpromoting'
assert receipt['run']['github_run_id'] == '38023231911'
assert receipt['run']['github_run_attempt'] == '1'
launch = '30c6f016d864ba4017ecd9b43e83fb5476098463'
assert receipt['source']['commit'] == receipt['source']['workflow_commit'] == launch
assert receipt['source']['repository'] == 'SlopDotCash/asi'
assert receipt['source']['dispatch_ref'] == 'refs/heads/main'
assert receipt['source']['tree'] == subprocess.check_output(
    ['git', 'rev-parse', launch + '^{tree}'], text=True).strip()
workflow = '.github/workflows/reference-life-scorecard-dev.yml'
assert receipt['source']['workflow_blob_sha1'] == subprocess.check_output(
    ['git', 'rev-parse', launch + ':' + workflow], text=True).strip()
assert receipt['source']['uv_lock_sha256'] == hashlib.sha256(
    subprocess.check_output(['git', 'show', launch + ':uv.lock'])).hexdigest()
expected_shards = sorted(
    f'{env}__{arm}__{seed}.json'
    for env in ('riverswim', 'switching_two_state')
    for arm in ('differential_sarsa', 'privileged_oracle', 'prototype',
                'prototype_frozen', 'random', 'sarsa')
    for seed in range(70000, 70012)
)
shards = sorted((root / 'shards').glob('*.json'))
assert [p.name for p in shards] == expected_shards
inventory = receipt['result']['inventory']
expected_paths = ['scorecard/plan.json', 'scorecard/artifact.json'] + [
    'scorecard/shards/' + name for name in expected_shards]
assert [item['path'] for item in inventory] == expected_paths
for item in inventory:
    path = root / item['path'].removeprefix('scorecard/')
    raw = path.read_bytes()
    assert len(raw) == item['bytes'], path
    assert hashlib.sha256(raw).hexdigest() == item['sha256'], path
canonical_inventory = json.dumps(inventory, sort_keys=True, separators=(',', ':'),
                                ensure_ascii=True).encode()
assert hashlib.sha256(canonical_inventory).hexdigest() == receipt['result']['campaign_inventory_sha256']
assert receipt['result']['file_count'] == 146
assert receipt['result']['shard_count'] == receipt['run']['job_count'] == 144
assert receipt['result']['plan_sha256'] == inventory[0]['sha256']
assert receipt['result']['artifact_sha256'] == inventory[1]['sha256']
plan = ReferenceLifeDevelopmentPlan.from_payload(load_json_strict(root / 'plan.json'))
artifact = load_json_strict(root / 'artifact.json')
validation = validate_scorecard_artifact(artifact)
recomputed = summarize_shard_files(shards, plan=plan)
assert recomputed == artifact, 'Independent aggregate differs from hosted artifact'
print(json.dumps({'validation': validation, 'inventory_files_verified': 146,
                  'shards_verified': 144, 'independent_aggregate_exact_match': True,
                  'github_run_id': '38023231911', 'launch_sha': launch},
                 sort_keys=True, indent=2))
