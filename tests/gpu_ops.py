# SPDX-License-Identifier: MIT
import torch, torchvision
import torch.nn.functional as F
print('torch',torch.__version__,'HIP',torch.version.hip,'device',torch.cuda.get_device_name(0),flush=True)
torch.manual_seed(19)
def check(name, cpu, gpu, rtol=1e-3, atol=1e-3):
    actual=gpu.detach().cpu(); expected=cpu.detach().cpu(); err=(actual-expected).abs().max().item()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    print(f'{name}: PASS max error={err:.6g}',flush=True)
a=torch.randn(384,320); b=torch.randn(320,256)
check('FP32 matrix',a@b,a.cuda()@b.cuda())
check('FP16 matrix',(a.half()@b.half()).float(),(a.cuda().half()@b.cuda().half()).float(),rtol=0.03,atol=0.03)
x=torch.randn(2,3,32,32,requires_grad=True); w=torch.randn(8,3,3,3,requires_grad=True)
xg=x.detach().cuda().requires_grad_(); wg=w.detach().cuda().requires_grad_()
y=F.conv2d(x,w,padding=1); yg=F.conv2d(xg,wg,padding=1)
check('convolution',y,yg,rtol=0.002,atol=0.002)
y.square().mean().backward(); yg.square().mean().backward()
check('backprop input',x.grad,xg.grad,rtol=0.003,atol=0.0001)
check('backprop weights',w.grad,wg.grad,rtol=0.003,atol=0.0001)
f=torch.randn(1024); check('FFT',torch.fft.rfft(f).abs(),torch.fft.rfft(f.cuda()).abs(),rtol=0.002,atol=0.002)
boxes=torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.],[20.,20.,30.,30.]]); scores=torch.tensor([0.9,0.8,0.7]); actual=torchvision.ops.nms(boxes.cuda(),scores.cuda(),0.5).cpu(); assert actual.tolist()==[0,2],actual
print('torchvision NMS: PASS',flush=True)
print('ALL PASS',flush=True)
