"""Execute the prospectively fixed, nonpromoting corrected-clock CPR panel."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARMS = ('cpr_off', 'cpr_ipmnist', 'cpr_utility_free', 'cpr_l2_init', 'cpr_hard_reset')
SEEDS = range(15639100, 15639105)

if __name__ == '__main__':
    # Reserve the entire roster before any learner dispatch. Existing reservation
    # and result files prohibit retries through this retained driver.
    with (HERE / 'reservation.json').open('x') as f:
        json.dump({'arms': ARMS, 'seeds': list(SEEDS), 'tasks': 8,
                   'steps_per_task': 5000, 'development_only': True,
                   'scientific_promotion_allowed': False}, f, indent=2)
    env = dict(os.environ, JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1', PYTHONPATH=str(ROOT))
    affinity = sorted(os.sched_getaffinity(0))[:4]
    os.sched_setaffinity(0, affinity)
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            stem = f'{arm}-{seed}'
            command = [sys.executable, '-m', 'alberta_framework.benchmarks.ipmnist_screening',
                       'run', '--config-name', arm, '--seed', str(seed), '--n-tasks', '8',
                       '--task-length', '5000', '--data-home',
                       '/tmp/asi-issue-completion/openml-cache', '--out',
                       str(HERE / (stem + '.json')), '--progress-every', '1']
            started = time.monotonic()
            with (HERE / (stem + '.log')).open('x') as log:
                try:
                    completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, timeout=300)
                    status = completed.returncode
                except subprocess.TimeoutExpired:
                    status = 'timeout'
            row = {'arm': arm, 'seed': seed, 'status': status,
                   'wall_seconds_telemetry': time.monotonic()-started,
                   'cpu_affinity': affinity, 'command': command}
            with (HERE / (stem + '.exit.json')).open('x') as f:
                json.dump(row, f, indent=2)
            rows.append(row)
            print(json.dumps(row), flush=True)
    with (HERE / 'dispatch.json').open('x') as f:
        json.dump(rows, f, indent=2)
    if any(row['status'] != 0 for row in rows):
        raise SystemExit(1)
