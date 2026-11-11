# SPDX-License-Identifier: MIT
"""Warm end-to-end comparison; identical host FP32 boundaries and resident weights."""
import json,time,statistics
from pathlib import Path
Path('results/local').mkdir(parents=True,exist_ok=True)
import numpy as np,torch
from npu_hybrid import FusedNpuMLP,SharedMemoryNpuMLP

torch.set_num_threads(16)
rng=np.random.default_rng(51)
make=lambda shape:(rng.normal(size=shape)*.04).astype('float32')
x=make((16,512,512));w1,w2=make((512,512)),make((512,512))
old=SharedMemoryNpuMLP(w1,w2);dram=FusedNpuMLP(w1,w2);sram=FusedNpuMLP(w1,w2,backend='sram')
w1g=torch.from_numpy(w1).to('cuda',torch.bfloat16);w2g=torch.from_numpy(w2).to('cuda',torch.bfloat16)
def gpu():
 a=torch.from_numpy(x[0]).to('cuda',torch.bfloat16)
 return (torch.relu(a@w1g)@w2g).cpu().float().numpy()
cases={'prior_npu_b1':(1,lambda:old(x[0]).astype('float32')),'gpu_b1':(1,gpu)}
for batch in (1,2,4,8,16):
 for name,model in [('dram',dram),('sram',sram)]:
  model.prepare_batch(batch)
  cases[f'{name}_b{batch}']=(batch,lambda b=batch,m=model:m(x[:b]).astype('float32'))
result={k:[] for k in cases}
for r in range(5):
 for name in (list(cases) if r%2==0 else list(cases)[::-1]):
  batch,fn=cases[name]
  # Cached hardware contexts may have been evicted while another case ran.
  # Warm them before timing, consistently excluding compilation/loading.
  for _ in range(8):fn()
  for _ in range(60):
   t=time.perf_counter_ns();fn();result[name].append((time.perf_counter_ns()-t)/1e6)
 print('round',r+1,'PASS',flush=True)
out={}
for name,v in result.items():
 mean=statistics.mean(v);batch=cases[name][0]
 out[name]=dict(batch=batch,mean_ms=mean,p50_ms=statistics.median(v),p95_ms=float(np.percentile(v,95)),amortized_ms_per_input=mean/batch,pipelines_per_s=batch*1000/mean,samples=len(v))
report=dict(operation='ReLU(A@B)@D',shape=[512,512],dtype='bf16',boundary='host FP32 input/output; resident BF16 weights; warm cache; torch CPU threads=16',dram_config=dram.config,sram_config=sram.config,results=out)
json.dump(report,open('results/local/pipeline.json','w'),indent=2)
print(json.dumps(report,indent=2),flush=True)
