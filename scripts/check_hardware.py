#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the GPU/NPU checks and record the actual results for this machine."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--full',action='store_true',help='include MNIST and longer GPU stability runs')
a=p.parse_args();root=Path(__file__).resolve().parent.parent
out=root/'results/local';out.mkdir(parents=True,exist_ok=True)
cases=['gpu_ops.py','gpu_models.py','npu.py','unified_api.py','legacy_npu.py','unified_stress.py']
if a.full:cases+=['gpu_training.py','gpu_stability.py']
env=os.environ.copy();env['PYTHONPATH']=str(root)+os.pathsep+env.get('PYTHONPATH','')
results=[]
for name in cases:
    print('RUN',name,flush=True);start=time.monotonic()
    r=subprocess.run([sys.executable,str(root/'tests'/name)],cwd=root,env=env,capture_output=True,text=True)
    (out/(name+'.log')).write_text(r.stdout+r.stderr)
    results.append(dict(test=name,passed=r.returncode==0,seconds=round(time.monotonic()-start,3),stdout=r.stdout))
    (out/'hardware-checks.json').write_text(json.dumps(results,indent=2)+'\n')
    print('PASS' if r.returncode==0 else 'FAIL',name,flush=True)
    if r.returncode:
        print(r.stdout[-3000:]+r.stderr[-3000:],flush=True)
        raise SystemExit(r.returncode)
print('ALL HARDWARE CHECKS PASS',flush=True)
