"""Compare full RMI parity checkpoints with the pinned reference, in order."""
import argparse
import json
from pathlib import Path

import numpy as np


def metrics(actual,reference):
    actual=np.asarray(actual,dtype=np.float64).reshape(-1)
    reference=np.asarray(reference,dtype=np.float64).reshape(-1)
    if actual.shape!=reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('checkpoint shape or finite-value mismatch')
    error=actual-reference
    rmse=float(np.sqrt(np.mean(error*error)))
    norm=float(np.linalg.norm(actual)*np.linalg.norm(reference))
    cosine=float(np.dot(actual,reference)/norm) if norm else float(np.array_equal(actual,reference))
    return {'max_abs':float(np.max(np.abs(error))),'rmse':rmse,'relative_rmse':rmse/max(float(np.sqrt(np.mean(reference*reference))),1e-12),'cosine':cosine}


def compare(trace,reference,output,quantized=False):
    thresholds=json.loads((Path(__file__).parent/'thresholds.json').read_text())
    limits=thresholds['quantized_implementation' if quantized else 'floating_implementation']
    manifest=json.loads((reference/'manifest.json').read_text())
    expected={}
    for item in manifest['checkpoints']:expected.setdefault(item['name'],[]).append(item)
    actual={};token_ids=None
    for line in trace.read_text().splitlines():
        item=json.loads(line)
        if 'token_ids' in item:
            if item['name']=='fastframes.token_ids':token_ids=item['token_ids']
        else:actual.setdefault(item['name'],[]).append(item)
    if token_ids!=json.loads((reference/'tokens.json').read_text()):raise ValueError('token IDs differ')
    if not manifest['checkpoints']:raise ValueError('reference has no checkpoints (benchmark mode cannot prove parity)')
    results=[];passed=True
    for name,items in expected.items():
        if name=='fastframes.pixels':continue
        if name not in actual or len(actual[name])!=len(items):
            raise ValueError(f'checkpoint occurrence count differs: {name}')
    for name in set(actual)-set(expected):raise ValueError(f'unexpected checkpoint: {name}')
    occurrences={}
    for right in manifest['checkpoints']:
        name=right['name']
        if name=='fastframes.pixels':continue
        occurrence=occurrences.get(name,0);occurrences[name]=occurrence+1
        left=actual[name][occurrence]
        binary=left.get('binary_path')
        if binary is None:raise ValueError(f'missing full checkpoint path in {left}')
        a=np.fromfile(binary,dtype='<f4');r=np.fromfile(reference/right['file'],dtype='<f4')
        if name=='fastframes.noise' and not np.array_equal(a,r):
            raise ValueError('shared noise differs; implementation tolerances require identical inputs')
        values=metrics(a,r)
        okay=values['relative_rmse']<=limits['relative_rmse_max'] and values['cosine']>=limits['cosine_min']
        results.append({'name':name,'occurrence':occurrence,'passed':okay,**values});passed &=okay
    report={'passed':passed,'first_failure':next((r for r in results if not r['passed']),None),'thresholds':limits,'reference_recipe':manifest['recipe'],'checkpoints':results}
    output.write_text(json.dumps(report,indent=2)+'\n')
    return passed


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quantized',action='store_true')
    args=parser.parse_args()
    if not compare(args.trace,args.reference,args.output,args.quantized):raise SystemExit(1)

if __name__=='__main__':main()
