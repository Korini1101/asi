import json,statistics,math,hashlib
from pathlib import Path
from alberta_framework.benchmarks.ipmnist_screening import load_shard,merge_shards,partial_reset_development_record,validate_partial_reset_development_record
root=Path('outputs/cpr_corrected_clock/2026-10-09-stable')
seeds=range(15639100,15639105);arms=['cpr_off','cpr_ipmnist','cpr_utility_free','cpr_l2_init','cpr_hard_reset']
paths=[root/f'{arm}-{seed}.json' for seed in seeds for arm in arms]
summary=merge_shards(paths,control_name='cpr_off',slope_window=4)
with (root/'merged.json').open('x') as f:json.dump(summary,f,indent=2,sort_keys=True,allow_nan=False)
rows={}
for path in paths:
    result=load_shard(path)
    receipt=partial_reset_development_record(result)
    validate_partial_reset_development_record(receipt)
    rows[(result.config_name,result.seed)]=result
accuracies={arm:[float(rows[arm,s].per_task_accuracy.mean()) for s in seeds] for arm in arms}
deltas={arm:[a-b for a,b in zip(accuracies['cpr_ipmnist'],accuracies[arm])] for arm in arms if arm!='cpr_ipmnist'}
primary=[deltas[a] for a in ['cpr_off','cpr_utility_free','cpr_l2_init']]
outcome='supported' if all(all(d>0 for d in v) for v in primary) else ('rejected' if all(statistics.mean(v)<=0 for v in primary) else 'inconclusive')
audit={'development_only':True,'scientific_promotion_allowed':False,'outcome':outcome,
'accuracy_by_seed':accuracies,'mean_accuracy':{a:statistics.mean(v) for a,v in accuracies.items()},
'cpr_minus_control':{a:{'per_seed':v,'mean':statistics.mean(v),'stderr':statistics.stdev(v)/math.sqrt(5)} for a,v in deltas.items()},
'file_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
with (root/'audit.json').open('x') as f:json.dump(audit,f,indent=2,sort_keys=True,allow_nan=False)
print(json.dumps(audit,indent=2))
