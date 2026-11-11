# SPDX-License-Identifier: MIT
from pathlib import Path
Path("results/local").mkdir(parents=True,exist_ok=True)
import json,torch,numpy as np
import aie.utils as utils
from npu_hybrid import FusedNpuMLP

torch.set_num_threads(16)
rng=np.random.default_rng(120)
make=lambda shape:(rng.normal(size=shape)*.04).astype('float32')
w1,w2=make((512,512)),make((512,512))
dram=FusedNpuMLP(w1,w2);sram=FusedNpuMLP(w1,w2,backend='sram')
runtime=utils.DefaultNPURuntime;original_run=runtime.run;launches=[]
class CountKernel:
 def __init__(self,k):self.k=k
 def __getattr__(self,key):return getattr(self.k,key)
 def __call__(self,*a,**kw):launches.append(1);return self.k(*a,**kw)
def counted_run(handle,*a,**kw):
 original=handle.kernel;handle.kernel=CountKernel(original)
 try:return original_run(handle,*a,**kw)
 finally:handle.kernel=original
runtime.run=counted_run
report={}
try:
 for batch in (1,2,4,8,16):
  print('checking batch',batch,flush=True)
  x=make((batch,512,512));ref=np.stack([dram(item) for item in x])
  for name,model in [('dram',dram),('sram',sram)]:
   print('backend',name,flush=True)
   before=len(launches);out=model(x)
   assert len(launches)-before==1
   np.testing.assert_array_equal(out,ref)
   saved=out.copy()
   out2=model.forward_batch(x[::-1])
   np.testing.assert_array_equal(out2,ref[::-1]);np.testing.assert_array_equal(out,saved)
  report[str(batch)]='both backends bitwise equal to individual executions; one XRT launch'
 for i in range(100):
  x=make((512,512));ref=dram(x)
  np.testing.assert_array_equal(sram(x),ref)
 for model in (dram,sram):
  for x in (np.zeros((3,512,512)),np.zeros((2,8,8)),torch.zeros((2,512,512),requires_grad=True)):
   before=len(launches)
   try:model.forward_batch(x)
   except ValueError:pass
   else:raise AssertionError('invalid input accepted')
   assert len(launches)==before
  for batch in (0,3,True,2.0):
   try:model.prepare_batch(batch)
   except ValueError:pass
   else:raise AssertionError('invalid batch accepted')
 report['changed_input_stress_runs']=100;report['invalid_inputs_rejected']=True
 assert all(len(plan[1])==4 for plan in sram._plans.values())
 report['sram_runtime_tensor_arguments']=4
 report['sram_hidden_bytes_per_core']=32*512*2
 report['sram_hidden_compute_cores']=16
 json.dump(report,open('results/local/npu-verification.json','w'),indent=2)
 print(json.dumps(report,indent=2),flush=True)
finally:runtime.run=original_run
