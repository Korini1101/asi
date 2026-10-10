"""Bounded real-data qualification through existing ASI runners."""
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = Path('/home/shaw/asi/outputs/datasets')
MNIST = DATA / 'mnist-cvdf-tfds-24211da'
CIFAR = DATA / 'cifar100-official-binary-20260905/cifar-100-binary-completed.tar.gz'
MNIST_HASHES = {
    'train-images-idx3-ubyte.gz': '440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609',
    'train-labels-idx1-ubyte.gz': '3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c',
}


def publish(name, value):
    with (HERE / name).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def arrays(kind):
    if kind == 'cifar':
        archive = CIFAR.read_bytes()
        assert hashlib.md5(archive).hexdigest() == '03b5dce01913d631647c71ecec9e9cb8'
        assert hashlib.sha256(archive).hexdigest() == '58a81ae192c23a4be8b1804d68e518ed807d710a4eb253b1f2a199162a40d8ec'
        with tarfile.open(CIFAR, 'r:gz') as archive:
            member = archive.getmember('cifar-100-binary/train.bin')
            assert member.isfile() and member.size == 50000 * 3074
            with archive.extractfile(member) as f:
                rows = np.frombuffer(f.read(member.size), np.uint8).reshape(50000, 3074)
        x, y = rows[:, 2:].astype(np.float32) / np.float32(255), rows[:, 1].astype(np.int32)
        assert np.array_equal(np.bincount(y), np.full(100, 500))
    else:
        raw = {}
        for name, expected in MNIST_HASHES.items():
            payload = (MNIST / name).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == expected
            raw[name] = gzip.decompress(payload)
        images, labels = raw['train-images-idx3-ubyte.gz'], raw['train-labels-idx1-ubyte.gz']
        assert struct.unpack('>IIII', images[:16]) == (2051, 60000, 28, 28)
        assert struct.unpack('>II', labels[:8]) == (2049, 60000)
        assert len(images) == 16 + 60000 * 784 and len(labels) == 8 + 60000
        x = np.frombuffer(images, np.uint8, offset=16).reshape(60000, 28, 28).astype(np.float32) / np.float32(255)
        y = np.frombuffer(labels, np.uint8, offset=8).astype(np.int32)
        assert np.all((y >= 0) & (y < 10))
    return x, y


def identity():
    from alberta_framework.reference_life_checkpoint import _source_identity, _runtime_identity, _dependency_identity
    return {'source': _source_identity(), 'runtime': _runtime_identity(),
            'dependencies': _dependency_identity(),
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def worker(kind, seed, label):
    if kind == 'native':
        from alberta_framework.benchmarks import native_supervised_suite as lane
        rows = []
        for benchmark in lane.BENCHMARK_IDS:
            x, y = arrays('cifar' if benchmark == 'split_cifar100' else 'mnist')
            result = lane.run_native_suite(benchmark, x, y, seed=seed,
                                          examples_per_task=64, replay_capacity=16)
            lane.validate_result(result)
            rows.append(asdict(result))
            del x, y
        publish(f'native-{seed}-{label}.json', {'identity': identity(), 'results': rows,
                                              'scientific_promotion_allowed': False})
    else:
        from alberta_framework.benchmarks.plasticity_diagnostics import run_diagnostic
        x, y = arrays('mnist')
        result = run_diagnostic(x.reshape(60000, 784), y, seed=seed,
                                profile_id='bounded-development')
        publish(f'diagnostic-{seed}.json', {'identity': identity(), 'result': asdict(result),
                                           'scientific_promotion_allowed': False})


if __name__ == '__main__':
    if len(sys.argv) > 1:
        worker(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    else:
        publish('reservation.json', {'native_seeds': list(range(15780, 15784)),
                                    'diagnostic_seeds': list(range(15830, 15834)),
                                    'plan_sha256': hashlib.sha256((HERE / 'PLAN.md').read_bytes()).hexdigest()})
        cpus = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, cpus[4:8] if len(cpus) >= 8 else cpus[:4])
        env = dict(os.environ, JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1', PYTHONPATH=str(ROOT))
        jobs = [('native', seed, label) for seed in range(15780, 15784) for label in ('run', 'replay')]
        jobs += [('diagnostic', seed, 'run') for seed in range(15830, 15834)]
        exits = []
        for kind, seed, label in jobs:
            with (HERE / f'{kind}-{seed}-{label}.log').open('x') as f:
                try:
                    code = subprocess.run([sys.executable, str(Path(__file__).resolve()), kind, str(seed), label],
                                          cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=300).returncode
                except subprocess.TimeoutExpired:
                    code = 'timeout'
            exits.append({'kind': kind, 'seed': seed, 'label': label, 'exit': code})
            print(json.dumps(exits[-1]), flush=True)
        publish('dispatch.json', exits)
        if any(row['exit'] != 0 for row in exits):
            raise SystemExit(1)
