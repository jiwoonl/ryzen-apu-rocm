# SPDX-License-Identifier: MIT
"""Verify actual XRT launches, repeated changed inputs, and BF16 parity."""
from pathlib import Path
Path("results/local").mkdir(parents=True,exist_ok=True)
import json
import numpy as np
import torch
import aie.utils as utils
from npu_hybrid import FusedNpuMLP, TwoDispatchNpuMLP
from ml_dtypes import bfloat16

torch.set_num_threads(16)
rng = np.random.default_rng(20261111)
make = lambda: (rng.normal(size=(512,512)) * .04).astype('float32')
w1,w2=make(),make()
new,old=FusedNpuMLP(w1,w2,config=dict(k=32,n=32)),TwoDispatchNpuMLP(w1,w2)
runtime=utils.DefaultNPURuntime
original_run=runtime.run
launches=[]
class CountKernel:
 def __init__(self,delegate): self.delegate=delegate
 def __getattr__(self,name): return getattr(self.delegate,name)
 def __call__(self,*args,**kwargs):
  launches.append(1)
  return self.delegate(*args,**kwargs)
def counted_run(handle,*args,**kwargs):
 original=handle.kernel
 handle.kernel=CountKernel(original)
 try:return original_run(handle,*args,**kwargs)
 finally:handle.kernel=original
runtime.run=counted_run
errors=[]
try:
 for i in range(100):
  x=make() if i else np.zeros((512,512),dtype='float32')
  start=len(launches);y=new(x).astype('float32')
  assert len(launches)-start == 1, 'not exactly one pyxrt kernel invocation'
  if i<12:
   start=len(launches);reference=old(x).astype('float32')
   assert len(launches)-start == 2
   np.testing.assert_array_equal(y,reference)
   q=lambda a:a.astype(bfloat16).astype('float32')
   reference=np.maximum(q(x)@q(w1),0).astype(bfloat16).astype('float32')@q(w2)
   rel=float(np.linalg.norm(y-reference)/max(float(np.linalg.norm(reference)),1e-20))
   errors.append(rel)
  assert np.isfinite(y).all()
 print('100 changed-input runs: PASS; exactly one actual XRT launch per call',flush=True)
 print('12 outputs bitwise identical to two-dispatch implementation',flush=True)
 for bad in (np.zeros((4,4)),torch.zeros((512,512),requires_grad=True)):
  start=len(launches)
  try:new(bad)
  except ValueError:pass
  else:raise AssertionError('invalid input accepted')
  assert len(launches)==start
 out=dict(runs=100,bitwise_comparisons=12,actual_xrt_launches_per_forward=1,old_xrt_launches_per_forward=2,max_relative_l2_vs_cpu=max(errors),invalid_inputs_rejected=True)
 json.dump(out,open('results/local/legacy-verification.json','w'),indent=2)
 print(json.dumps(out,indent=2),flush=True)
finally:runtime.run=original_run
