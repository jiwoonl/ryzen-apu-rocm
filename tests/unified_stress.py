# SPDX-License-Identifier: MIT
import numpy as np
import torch
from npu_hybrid import FusedNpuGpuMLP, FusedNpuMLP
rng=np.random.default_rng(23); m=lambda:(rng.normal(size=(512,512))*.04).astype(np.float32)
w1,w2,x=m(),m(),m(); mixed=FusedNpuGpuMLP(w1,w2); npu=FusedNpuMLP(w1,w2)
first=mixed(x).cpu().float();first_npu=npu(x).astype(np.float32)
for i in range(100):
 out=mixed(x).cpu().float();npu_out=npu(x).astype(np.float32)
 if not torch.isfinite(out).all() or not np.isfinite(npu_out).all():raise RuntimeError('nonfinite output')
 if (i+1)%20==0:
  torch.testing.assert_close(out,first,rtol=0,atol=0);np.testing.assert_array_equal(npu_out,first_npu)
  print('iterations',i+1,'PASS',flush=True)
print('UNIFIED STRESS PASS',flush=True)
