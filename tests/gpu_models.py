# SPDX-License-Identifier: MIT
import gc
import torch
from torch import nn
from torchvision.models import resnet18, mobilenet_v3_small, squeezenet1_1

torch.manual_seed(2026)
torch.set_num_threads(4)
assert torch.cuda.is_available(), 'ROCm GPU unavailable'
print('PyTorch:',torch.__version__,'device:',torch.cuda.get_device_name(0),flush=True)

def compare(name, factory, input_shape, training=False, fp16=False):
    print(f'Running {name} ...',flush=True)
    model=factory().eval(); data=torch.randn(*input_shape)
    if training: model.train()
    cpu_x=data.clone().requires_grad_(training)
    cpu_y=model(cpu_x)
    if training: cpu_y.square().mean().backward()
    cpu_result=cpu_y.detach(); cpu_grad=cpu_x.grad.detach() if training else None
    gpu_model=factory().eval(); gpu_model.load_state_dict(model.state_dict())
    if training: gpu_model.train()
    gpu_model=gpu_model.cuda()
    if fp16: gpu_model=gpu_model.half()
    gpu_x=data.cuda().to(dtype=torch.float16 if fp16 else torch.float32).requires_grad_(training)
    gpu_y=gpu_model(gpu_x)
    if training: gpu_y.float().square().mean().backward()
    gpu_result=gpu_y.detach().float().cpu()
    tol=0.03 if fp16 else 0.004
    torch.testing.assert_close(gpu_result,cpu_result,atol=tol,rtol=tol)
    if training:
        grad=gpu_x.grad.detach().float().cpu()
        torch.testing.assert_close(grad,cpu_grad,atol=0.002,rtol=0.03)
    print(name,'PASS',f'output max error={(gpu_result-cpu_result).abs().max().item():.6g}',flush=True)
    del model,gpu_model,cpu_x,cpu_y,gpu_x,gpu_y; gc.collect(); torch.cuda.empty_cache()

compare('ResNet18 FP32 inference',lambda:resnet18(weights=None),(1,3,96,96))
compare('MobileNetV3 small FP32 inference',lambda:mobilenet_v3_small(weights=None),(1,3,96,96))
compare('SqueezeNet FP16 inference',lambda:squeezenet1_1(weights=None),(1,3,96,96),fp16=True)
compare('ResNet18 FP32 training',lambda:resnet18(weights=None),(2,3,64,64),training=True)
compare('Transformer encoder FP32 training',lambda:nn.TransformerEncoderLayer(d_model=64,nhead=4,dim_feedforward=128,dropout=0,batch_first=True),(2,24,64),training=True)
print('MODEL TESTS PASS',flush=True)
