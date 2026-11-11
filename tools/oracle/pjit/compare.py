"""Finite, shape-strict numerical comparison; no bitwise checks for lossy paths."""
import argparse
import json
from pathlib import Path
import numpy as np

def metrics(actual,expected):
    actual=np.asarray(actual,dtype=np.float64);expected=np.asarray(expected,dtype=np.float64)
    if actual.shape!=expected.shape or actual.size==0:raise ValueError('shape mismatch or empty output')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():raise ValueError('non-finite output')
    error=actual-expected;rmse=float(np.sqrt(np.mean(error*error)))
    norm=float(np.linalg.norm(actual)*np.linalg.norm(expected))
    return dict(max_abs=float(np.max(np.abs(error))),rmse=rmse,nrmse=rmse/max(float(np.sqrt(np.mean(expected*expected))),1e-12),cosine=float(np.sum(actual*expected)/norm) if norm else float(np.array_equal(actual,expected)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('actual',type=Path);p.add_argument('expected',type=Path)
    p.add_argument('--max-nrmse',type=float,default=.02);p.add_argument('--min-cosine',type=float,default=.999)
    a=p.parse_args();result=metrics(np.fromfile(a.actual,dtype='<f4'),np.fromfile(a.expected,dtype='<f4'))
    result['passed']=result['nrmse']<=a.max_nrmse and result['cosine']>=a.min_cosine
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 1)
