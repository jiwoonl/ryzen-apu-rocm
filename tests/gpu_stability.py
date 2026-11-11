# SPDX-License-Identifier: MIT
import gc, time, traceback
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import (resnet18, mobilenet_v3_small, efficientnet_b0,
    densenet121, shufflenet_v2_x0_5, vgg11, vit_b_16)

torch.set_num_threads(4)
torch.manual_seed(20261111)
assert torch.cuda.is_available(), 'GPU unavailable'
print('PyTorch',torch.__version__,'HIP',torch.version.hip,'GPU',torch.cuda.get_device_name(0),flush=True)
failures=[]
def run(name,fn):
    print('START',name,flush=True)
    start=time.monotonic()
    try:
        fn(); torch.cuda.synchronize()
        print('PASS',name,'seconds',round(time.monotonic()-start,2),flush=True)
    except Exception as e:
        failures.append(name); print('FAIL',name,repr(e),flush=True); traceback.print_exc()
    finally:
        gc.collect(); torch.cuda.empty_cache()

def vit_model():
    model=vit_b_16(weights=None)
    nn.init.normal_(model.heads.head.weight, std=0.02)
    nn.init.normal_(model.heads.head.bias, std=0.02)
    return model

def inference(factory, shape, half=False):
    model=factory().eval(); x=torch.randn(*shape)
    with torch.inference_mode():
        expected=model(x); model=model.cuda(); x=x.cuda()
        if half: model=model.half(); x=x.half()
        out=model(x).float().cpu()
    torch.testing.assert_close(out,expected,rtol=0.04 if half else 0.004,atol=0.025 if half else 0.002)
    print('  max_abs_error',float((out-expected).abs().max()),flush=True)

for name,factory,shape,half in [
 ('EfficientNet-B0 FP32',lambda:efficientnet_b0(weights=None),(1,3,96,96),False),
 ('DenseNet121 FP32',lambda:densenet121(weights=None),(1,3,96,96),False),
 ('ShuffleNetV2 FP32',lambda:shufflenet_v2_x0_5(weights=None),(1,3,96,96),False),
 ('VGG11 FP32',lambda:vgg11(weights=None),(1,3,64,64),False),
 ('MobileNetV3 FP16',lambda:mobilenet_v3_small(weights=None),(1,3,96,96),True),
 ('ViT-B/16 FP32',vit_model,(1,3,224,224),False),
]: run(name,lambda f=factory,s=shape,h=half:inference(f,s,h))

def lstm_test():
    m=nn.LSTM(32,64,2,batch_first=True).eval(); x=torch.randn(2,32,32)
    with torch.inference_mode():
        ref=m(x)[0]; m=m.cuda(); out=m(x.cuda())[0].cpu()
    torch.testing.assert_close(out,ref,rtol=0.003,atol=0.001)
run('LSTM inference',lstm_test)

def attention_test():
    q=torch.randn(2,4,32,32); k=torch.randn_like(q); v=torch.randn_like(q)
    ref=F.scaled_dot_product_attention(q,k,v); out=F.scaled_dot_product_attention(q.cuda(),k.cuda(),v.cuda()).cpu()
    torch.testing.assert_close(out,ref,rtol=0.004,atol=0.002)
run('Scaled dot product attention',attention_test)

def stress_inference():
    m=mobilenet_v3_small(weights=None).cuda().eval(); x=torch.randn(2,3,128,128,device='cuda')
    with torch.inference_mode():
        y=m(x); torch.cuda.synchronize(); first=y.clone(); start=torch.cuda.memory_allocated(); peak=0
        for i in range(2000):
            y=m(x)
            if (i+1)%400==0:
                torch.cuda.synchronize(); torch.testing.assert_close(y,first,rtol=1e-4,atol=1e-5)
                peak=max(peak,torch.cuda.memory_allocated()); print('  iteration',i+1,'allocated MiB',round(torch.cuda.memory_allocated()/2**20,1),flush=True)
        print('  allocated change MiB',round((peak-start)/2**20,2),flush=True)
run('MobileNet repeated inference x2000',stress_inference)

def stress_train():
    m=resnet18(weights=None,num_classes=10).cuda().train(); opt=torch.optim.SGD(m.parameters(),lr=0.001); x=torch.randn(4,3,64,64,device='cuda'); target=torch.randint(10,(4,),device='cuda')
    for i in range(200):
        opt.zero_grad(set_to_none=True); y=m(x); loss=F.cross_entropy(y,target); loss.backward()
        if not torch.isfinite(loss).item() or any(not torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None): raise RuntimeError('non-finite loss or gradient')
        opt.step()
        if (i+1)%50==0: torch.cuda.synchronize(); print('  step',i+1,'loss',round(loss.item(),5),'allocated MiB',round(torch.cuda.memory_allocated()/2**20,1),flush=True)
run('ResNet18 training x200',stress_train)
print('FAILED:',failures,flush=True)
if failures: raise SystemExit(1)
print('ALL STABILITY TESTS PASS',flush=True)
