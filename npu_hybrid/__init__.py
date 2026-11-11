# SPDX-License-Identifier: MIT
"""Fixed-shape 512x512 BF16 inference on XDNA1 and Radeon GPU.

FusedNpuMLP executes ReLU(A @ B) @ D with ONE XRT hardware dispatch.
The default uses a shared-memory scratch buffer; backend="sram" keeps hidden
activations in NPU core SRAM. Batches also run in a single NPU command. FusedNpuGpuMLP splits the two
stages between NPU and GPU. Inference only; approximate BF16 arithmetic.
"""
import numpy as np
import torch
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron.device import NPU1
from aie.utils.npukernel import NPUKernel
from ._fused_matmul_relu import whole_array as _matmul_relu
from ._matmul import whole_array as _matmul
from ._two_matmul import two_matmul as _two_matmul
from ._sram import sram_mlp as _sram_mlp

_N=512
_KW=dict(M=_N,K=_N,N=_N,m=32,k=32,n=32,n_aie_cols=4,dtype_in_str='bf16',dtype_out_str='bf16')

def _array(value,name):
    if isinstance(value,torch.Tensor):
        if value.requires_grad: raise ValueError(f'{name} has requires_grad=True; only inference is supported')
        value=value.detach().to(device='cpu',dtype=torch.float32).numpy()
    value=np.asarray(value)
    if value.shape!=(_N,_N): raise ValueError(f'{name} must have shape ({_N}, {_N}), got {value.shape}')
    return np.ascontiguousarray(value,dtype=bfloat16)

def _torch_bf16(value):
    return torch.from_numpy(np.asarray(value).view(np.uint16)).view(torch.bfloat16)

class _Base:
    def __init__(self,first_weight,second_weight):
        iron.set_current_device(NPU1())
        self._first=iron.tensor(_array(first_weight,'first_weight'),dtype=bfloat16,device='npu')
        self._second_np=_array(second_weight,'second_weight')
        self._input=iron.zeros((_N,_N),dtype=bfloat16,device='npu')
        self._intermediate=iron.zeros((_N,_N),dtype=bfloat16,device='npu')
        self._first_kernel=NPUKernel(*_matmul_relu.specialize(**_KW).compile())
    def _first_stage(self,x):
        self._input.torch_view().copy_(_torch_bf16(_array(x,'input')))
        self._input.to('npu')
        self._first_kernel(self._input,self._first,self._intermediate)
    def __call__(self,x): return self.forward(x)

class FusedNpuGpuMLP(_Base):
    """One Python call for NPU matmul+ReLU followed by GPU matmul.

    Construct once with two 512x512 weights; call with a 512x512 input. Returns
    a GPU torch.bfloat16 tensor. Not thread-safe and not autograd-compatible.
    """
    def __init__(self,first_weight,second_weight):
        if not torch.cuda.is_available(): raise RuntimeError('ROCm GPU unavailable')
        super().__init__(first_weight,second_weight)
        self._second_gpu=_torch_bf16(self._second_np).cuda()
        self._gpu_intermediate=torch.empty((_N,_N),device='cuda',dtype=torch.bfloat16)
    def forward(self,x):
        self._first_stage(x)
        intermediate=np.asarray(self._intermediate.numpy()).reshape(_N,_N)
        self._gpu_intermediate.copy_(_torch_bf16(intermediate))
        return self._gpu_intermediate @ self._second_gpu

class TwoDispatchNpuMLP(_Base):
    """One Python call for two NPU dispatches; returns a NumPy BF16 array.

    The first dispatch fuses matmul with ReLU. The second dispatch performs the
    last matmul. Not thread-safe and not autograd-compatible.
    """
    def __init__(self,first_weight,second_weight):
        super().__init__(first_weight,second_weight)
        self._second=iron.tensor(self._second_np,dtype=bfloat16,device='npu')
        self._output=iron.zeros((_N,_N),dtype=bfloat16,device='npu')
        self._second_kernel=NPUKernel(*_matmul.specialize(**_KW).compile())
    def forward(self,x):
        self._first_stage(x)
        self._second_kernel(self._intermediate,self._second,self._output)
        return np.asarray(self._output.numpy()).reshape(_N,_N).copy()


class SharedMemoryNpuMLP(_Base):
    """ReLU(input @ first_weight) @ second_weight in one NPU dispatch.

    Weights and input must be 512x512. Returns an owned NumPy BF16 array.
    A 512 KiB shared-memory intermediate is written and reread by NPU DMA,
    with an on-device completion barrier between stages. No intermediate
    host synchronization or second XRT launch. Not thread-safe; no autograd.
    """
    def __init__(self, first_weight, second_weight):
        iron.set_current_device(NPU1())
        self._first = iron.tensor(_array(first_weight, 'first_weight'), dtype=bfloat16, device='npu')
        self._second = iron.tensor(_array(second_weight, 'second_weight'), dtype=bfloat16, device='npu')
        self._input = iron.zeros((_N, _N), dtype=bfloat16, device='npu')
        self._intermediate = iron.zeros((_N, _N), dtype=bfloat16, device='npu')
        self._output = iron.zeros((_N, _N), dtype=bfloat16, device='npu')
        self._kernel = NPUKernel(*_two_matmul.specialize(**_KW).compile())

    def forward(self, x):
        self._input.torch_view().copy_(_torch_bf16(_array(x, 'input')))
        self._input.to('npu')
        self._kernel(self._input, self._first, self._second, self._intermediate, self._output)
        return np.asarray(self._output.numpy()).reshape(_N, _N).copy()


class FusedNpuMLP(_Base):
    """One-dispatch two-matmul inference with optional SRAM intermediates.

    backend='auto' uses the measured faster shared-memory implementation.
    backend='sram' keeps every hidden row block in NPU compute-tile SRAM.
    forward accepts (512,512) or (B,512,512); forward_batch accepts the latter for
    B in {1,2,4,8,16}; both return owned NumPy BF16 arrays. Weights are shared
    across batch items. Each batch is ONE actual XRT launch, without padding.
    Compilation and buffer allocation happen once per batch size. Use
    prepare_batch(B) ahead of timed execution. Inference only; not thread-safe.
    """
    _batches = (1, 2, 4, 8, 16)

    def __init__(self, first_weight, second_weight, *, backend='auto', config=None):
        if backend not in ('auto', 'dram', 'sram'):
            raise ValueError("backend must be 'auto', 'dram', or 'sram'")
        self.backend = 'dram' if backend == 'auto' else backend
        self.config = dict(m=32, k=64, n=64, depth=2, dma_rows=4) if self.backend == 'dram' else dict(m=32, k=64, n=64, depth=1)
        if config:
            if set(config) - set(self.config):
                raise ValueError('unknown configuration keys')
            self.config.update(config)
        cfg = self.config
        if cfg['m'] not in (8,16,32,64) or cfg['k'] not in (16,32,64) or cfg['n'] not in (16,32,64) or cfg['depth'] not in (1,2,3):
            raise ValueError('unsupported tile dimensions or buffer depth')
        if self.backend == 'sram' and (cfg['m'] > 32 or cfg['k'] != cfg['n']):
            raise ValueError('SRAM backend requires m <= 32 and k == n')
        if self.backend == 'dram' and cfg['dma_rows'] not in (2,4,8):
            raise ValueError('dma_rows must be 2, 4, or 8')
        iron.set_current_device(NPU1())
        self._first = iron.tensor(_array(first_weight, 'first_weight'), dtype=bfloat16, device='npu')
        self._second = iron.tensor(_array(second_weight, 'second_weight'), dtype=bfloat16, device='npu')
        self._plans = {}
        self.prepare_batch(1)

    def prepare_batch(self, batch_size):
        if isinstance(batch_size, bool) or batch_size not in self._batches or not isinstance(batch_size, (int, np.integer)):
            raise ValueError(f'batch size must be one of {self._batches}')
        if batch_size in self._plans:
            return self
        cfg = self.config
        iron.set_current_device(NPU1())
        shape = (batch_size * _N, _N)
        inp = iron.zeros(shape, dtype=bfloat16, device='npu')
        out = iron.zeros(shape, dtype=bfloat16, device='npu')
        if self.backend == 'sram':
            kernel = NPUKernel(*_sram_mlp.specialize(batch=batch_size, m=cfg['m'], tile=cfg['n'], depth=cfg['depth']).compile())
            args = [inp, self._first, self._second, out]
        else:
            kw = _KW | dict(M=shape[0], m=cfg['m'], k=cfg['k'], n=cfg['n'], fifo_depth=cfg['depth'], dma_rows=cfg['dma_rows'])
            kernel = NPUKernel(*_two_matmul.specialize(**kw).compile())
            hidden = iron.zeros(shape, dtype=bfloat16, device='npu')
            args = [inp, self._first, self._second, hidden, out]
        # The CPU view aliases the existing input BO; np.copyto avoids a CPU
        # thread-pool launch while staging input for its DMA upload.
        host_view = inp.torch_view().view(torch.uint16).numpy().reshape(shape)
        self._plans[batch_size] = (kernel, args, host_view)
        return self

    def _execute(self, value, batch_size):
        self.prepare_batch(batch_size)
        kernel, args, host_view = self._plans[batch_size]
        # Mark the BO CPU-dirty on EVERY write; a cached NumPy view alone
        # does not notify XRT after its first upload.
        args[0].torch_view()
        np.copyto(host_view, value.view(np.uint16).reshape(host_view.shape))
        args[0].to('npu')
        kernel(*args)
        return np.asarray(args[-1].numpy()).reshape(batch_size, _N, _N).copy()

    def forward(self, x):
        ndim = x.ndim if hasattr(x, 'ndim') else np.asarray(x).ndim
        if ndim == 3:
            return self.forward_batch(x)
        return self._execute(_array(x, 'input'), 1)[0]

    def forward_batch(self, x):
        if isinstance(x, torch.Tensor):
            if x.requires_grad:
                raise ValueError('input has requires_grad=True; only inference is supported')
            x = x.detach().to(device='cpu', dtype=torch.float32).numpy()
        x = np.asarray(x)
        if x.ndim != 3 or x.shape[1:] != (_N, _N) or x.shape[0] not in self._batches:
            raise ValueError(f'input must have shape (B,512,512), B in {self._batches}')
        return self._execute(np.ascontiguousarray(x, dtype=bfloat16), x.shape[0])
