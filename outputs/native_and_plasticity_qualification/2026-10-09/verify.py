import json
import hashlib
import math
import statistics
from pathlib import Path
from alberta_framework.benchmarks import native_supervised_suite as native
from alberta_framework.benchmarks import plasticity_diagnostics as diagnostic

root=Path('outputs/native_and_plasticity_qualification/2026-10-09')
summary={'schema':'asi.native-plasticity.issue-completion.v1','development_only':True,
         'scientific_promotion_allowed':False,'native':{},'diagnostic':{},'file_sha256':{}}
rows={name:[] for name in native.BENCHMARK_IDS}
for seed in native.FROZEN_SEEDS:
    pair=[]
    for label in ('run','replay'):
        path=root/f'native-{seed}-{label}.json'
        raw=json.loads(path.read_text());summary['file_sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        for payload in raw['results']:
            value=dict(payload);value['runtime_identity']=tuple(value['runtime_identity']);arms=[]
            for entry in value['arms']:
                arm=dict(entry);arm['task_accuracies']=tuple(arm['task_accuracies'])
                arm['receipt']=native.ResourceReceipt(**arm['receipt']);arms.append(native.ArmResult(**arm))
            value['arms']=tuple(arms);result=native.validate_result(native.SuiteResult(**value))
            assert result.seed==seed
            assert result.examples_per_task==64 and result.replay_capacity==16
            if label=='run': rows[result.benchmark_id].append(result)
            for arm in payload['arms']:arm['receipt']['elapsed_ns']=0
        pair.append(raw)
    assert pair[0]==pair[1],seed
for name, results in rows.items():
    assert len(results)==4
    accuracies={arm:[next(x.online_accuracy for x in r.arms if x.arm_id==arm) for r in results] for arm in native.ARM_IDS}
    delta=[a-b for a,b in zip(accuracies['replay_sgd'],accuracies['online_sgd'])]
    summary['native'][name]={'accuracy_by_seed':accuracies,'paired_replay_minus_online':delta,
        'mean_paired_delta':statistics.mean(delta),'stderr':statistics.stdev(delta)/2,
        'outcome':'development_candidate' if statistics.mean(delta)>0 else 'development_rejection',
        'mean_accuracy':{a:statistics.mean(v) for a,v in accuracies.items()},
        'fresh_process_replays':4}
for seed in diagnostic.FROZEN_SEEDS:
    path=root/f'diagnostic-{seed}.json';raw=json.loads(path.read_text());summary['file_sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    value=dict(raw['result']);value['runtime_identity']=tuple(value['runtime_identity'])
    value['profile']=diagnostic.DiagnosticProfile(**value['profile']);arms=[]
    for entry in value['arms']:
        arm=dict(entry)
        for field in ('task_accuracy','task_loss','dead_unit_fraction','effective_rank'):arm[field]=tuple(arm[field])
        receipt=dict(arm['receipt'])
        receipt['replacements_by_layer']=tuple(receipt['replacements_by_layer'])
        arm['receipt']=diagnostic.ResourceReceipt(**receipt);arms.append(diagnostic.ArmResult(**arm))
    value['arms']=tuple(arms);result=diagnostic.validate_result(diagnostic.DiagnosticResult(**value))
    summary['diagnostic'][str(seed)]={arm.arm_id:{'mean_task_accuracy':statistics.mean(arm.task_accuracy),'final_state_sha256':arm.final_state_sha256} for arm in result.arms}
assert summary == json.loads((root/'summary.json').read_text())
print('16 native records, four exact fresh-process panel replays and four diagnostic records accepted')
