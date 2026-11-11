# SPDX-License-Identifier: MIT
"""Isolated compile/run/verify/benchmark of an NPU configuration."""
import argparse,json,time,statistics,os
import torch,numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron.device import NPU1
from aie.utils.npukernel import NPUKernel
from npu_hybrid._sram import sram_mlp
from npu_hybrid._two_matmul import two_matmul
from npu_hybrid._two_matmul_baseline import two_matmul as baseline
p=argparse.ArgumentParser();p.add_argument('kind',choices=['sram','dram']);p.add_argument('--m',type=int,default=32);p.add_argument('--k',type=int,default=32);p.add_argument('--n',type=int,default=32);p.add_argument('--depth',type=int,default=2);p.add_argument('--dma',type=int,default=4);p.add_argument('--batch',type=int,default=1);p.add_argument('--iters',type=int,default=100);p.add_argument('--output',default='results/local/tuning.jsonl');args=p.parse_args()
from pathlib import Path
Path(args.output).parent.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(16);iron.set_current_device(NPU1())
rng=np.random.default_rng(11)
a=(rng.normal(size=(args.batch*512,512))*.04).astype(bfloat16)
b,d=[(rng.normal(size=(512,512))*.04).astype(bfloat16) for _ in range(2)]
kw=dict(M=args.batch*512,K=512,N=512,m=args.m,k=args.k,n=args.n,n_aie_cols=4,dtype_in_str='bf16',dtype_out_str='bf16',fifo_depth=args.depth,dma_rows=args.dma)
if args.kind=='sram':
 assert args.k==args.n
 fn=sram_mlp;kw=dict(batch=args.batch,m=args.m,tile=args.n,depth=args.depth)
else:fn=two_matmul
start=time.perf_counter();kernel=NPUKernel(*fn.specialize(**kw).compile());compile_s=time.perf_counter()-start
bufs=[iron.tensor(v,dtype=bfloat16,device='npu') for v in (a,b,d)]
if args.kind=='dram':bufs.append(iron.zeros(a.shape,dtype=bfloat16,device='npu'))
bufs.append(iron.zeros(a.shape,dtype=bfloat16,device='npu'))
# Match native NPU arithmetic using independently executed stages.
kernel(*bufs);y=np.asarray(bufs[-1].numpy()).astype('float32').reshape(a.shape)
if args.k==32:
 basekw=dict(M=512,K=512,N=512,m=32,k=32,n=32,n_aie_cols=4,dtype_in_str='bf16',dtype_out_str='bf16')
 bk=NPUKernel(*baseline.specialize(**basekw).compile())
 scratch=iron.zeros((512,512),dtype=bfloat16,device='npu');out=iron.zeros((512,512),dtype=bfloat16,device='npu')
 for i in range(args.batch):
  inp=iron.tensor(a[i*512:(i+1)*512],dtype=bfloat16,device='npu');bk(inp,bufs[1],bufs[2],scratch,out)
  np.testing.assert_array_equal(y[i*512:(i+1)*512],np.asarray(out.numpy()).reshape(512,512).astype('float32'))
 reference_kind='bitwise baseline'
else:
 from npu_hybrid._matmul import whole_array as standalone_mm
 basekw=dict(M=512,K=512,N=512,m=32,k=args.k,n=32,n_aie_cols=4,dtype_in_str='bf16',dtype_out_str='bf16')
 bk=NPUKernel(*standalone_mm.specialize(**basekw).compile())
 scratch=iron.zeros((512,512),dtype=bfloat16,device='npu');out=iron.zeros((512,512),dtype=bfloat16,device='npu')
 for i in range(args.batch):
  inp=iron.tensor(a[i*512:(i+1)*512],dtype=bfloat16,device='npu');bk(inp,bufs[1],scratch)
  hidden=np.maximum(np.asarray(scratch.numpy()).reshape(512,512).astype('float32'),0).astype(bfloat16)
  hidden=iron.tensor(hidden,dtype=bfloat16,device='npu');bk(hidden,bufs[2],out)
  np.testing.assert_array_equal(y[i*512:(i+1)*512],np.asarray(out.numpy()).reshape(512,512).astype('float32'))
 reference_kind='bitwise independent NPU stages, same K tile'

ref=np.maximum(a.astype('float32')@b.astype('float32'),0)@d.astype('float32')
relative_l2=float(np.linalg.norm(y-ref)/np.linalg.norm(ref))
for _ in range(10):kernel(*bufs)
raw=[];e2e=[]
for _ in range(args.iters):
 t=time.perf_counter_ns();kernel(*bufs);raw.append((time.perf_counter_ns()-t)/1e6)
for _ in range(args.iters):
 t=time.perf_counter_ns()
 np.copyto(bufs[0].torch_view().view(torch.uint16).numpy(),a.view(np.uint16));bufs[0].to('npu')
 kernel(*bufs);np.asarray(bufs[-1].numpy()).reshape(a.shape).copy().astype('float32')
 e2e.append((time.perf_counter_ns()-t)/1e6)
row=vars(args)|dict(compile_s=compile_s,verified=reference_kind,relative_l2_vs_fp32=relative_l2,raw_mean_ms=statistics.mean(raw),raw_p50_ms=statistics.median(raw),mean_ms=statistics.mean(e2e),p50_ms=statistics.median(e2e),p95_ms=float(np.percentile(e2e,95)),pipelines_per_s=args.batch*1000/statistics.mean(e2e),xclbin=str(kernel.xclbin_path))
with open(args.output,'a') as f:f.write(json.dumps(row)+'\n')
print(json.dumps(row),flush=True)
