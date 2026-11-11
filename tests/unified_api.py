# SPDX-License-Identifier: MIT
import torch
import numpy as np
from npu_hybrid import FusedNpuGpuMLP,FusedNpuMLP
from ml_dtypes import bfloat16
rng=np.random.default_rng(991)
w1=(rng.normal(size=(512,512))*.04).astype('float32')
w2=(rng.normal(size=(512,512))*.04).astype('float32')
mixed=FusedNpuGpuMLP(w1,w2); npu=FusedNpuMLP(w1,w2)
for i in range(3):
 x=(rng.normal(size=(512,512))*.04).astype('float32')
 q=lambda t:torch.from_numpy(np.asarray(t).astype(bfloat16).astype('float32'))
 reference=torch.relu(q(x)@q(w1))@q(w2)
 a=mixed(x).float().cpu(); b=torch.from_numpy(npu(torch.from_numpy(x)).astype('float32'))
 torch.testing.assert_close(a,reference,rtol=.06,atol=.01);torch.testing.assert_close(b,reference,rtol=.06,atol=.01)
 print('input',i+1,'NPU+GPU max error',float((a-reference).abs().max()),'NPU-only max error',float((b-reference).abs().max()),flush=True)
try:mixed(np.zeros((4,4),dtype=np.float32));raise AssertionError('invalid shape accepted')
except ValueError:print('invalid shape rejected',flush=True)
try:mixed(torch.zeros((512,512),requires_grad=True));raise AssertionError('autograd input accepted')
except ValueError:print('autograd input rejected',flush=True)
print('UNIFIED API PASS',flush=True)
