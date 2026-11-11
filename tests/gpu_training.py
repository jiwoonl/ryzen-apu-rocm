# SPDX-License-Identifier: MIT
import os, time, copy, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

torch.manual_seed(42); random.seed(42); np.random.seed(42); torch.set_num_threads(4)
assert torch.cuda.is_available(), 'ROCm GPU not available'
device=torch.device('cuda'); print('GPU:',torch.cuda.get_device_name(0),'torch:',torch.__version__,flush=True)
root=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
train=datasets.MNIST(root,train=True,download=True,transform=transforms.ToTensor())
test=datasets.MNIST(root,train=False,download=True,transform=transforms.ToTensor())
generator=torch.Generator().manual_seed(42)
train_ids=torch.randperm(len(train),generator=generator)[:12000].tolist()
test_ids=torch.randperm(len(test),generator=generator)[:2000].tolist()
train_loader=DataLoader(Subset(train,train_ids),batch_size=128,shuffle=True,num_workers=0,generator=generator)
test_loader=DataLoader(Subset(test,test_ids),batch_size=256,shuffle=False,num_workers=0)
class CNN(nn.Module):
    def __init__(self):
        super().__init__(); self.net=nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Flatten(),nn.Linear(32*7*7,64),nn.ReLU(),nn.Linear(64,10))
    def forward(self,x):return self.net(x)
model=CNN().to(device); optimizer=torch.optim.Adam(model.parameters(),lr=.001); start=time.monotonic()
for epoch in range(1,4):
    model.train(); loss_sum=0.; seen=0
    for x,y in train_loader:
        x=x.to(device); y=y.to(device); optimizer.zero_grad(set_to_none=True); logits=model(x); loss=nn.functional.cross_entropy(logits,y)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite MNIST loss')
        loss.backward(); optimizer.step(); loss_sum += loss.item()*len(x); seen+=len(x)
    model.eval(); correct=0; total=0
    with torch.inference_mode():
        for x,y in test_loader:
            pred=model(x.to(device)).argmax(1).cpu(); correct+=(pred==y).sum().item(); total+=len(y)
    torch.cuda.synchronize(); print(f'MNIST epoch {epoch}: train_loss={loss_sum/seen:.4f} test_acc={correct/total:.2%} ({correct}/{total}) elapsed={time.monotonic()-start:.1f}s',flush=True)
    if epoch==1: first_loss=loss_sum/seen
last_loss=loss_sum/seen
x,y=next(iter(test_loader)); model.eval(); cpu_model=copy.deepcopy(model).cpu()
with torch.inference_mode():
    gpu_logits=model(x.to(device)).cpu(); cpu_logits=cpu_model(x)
maxerr=(gpu_logits-cpu_logits).abs().max().item(); torch.testing.assert_close(gpu_logits,cpu_logits,rtol=0.003,atol=0.003)
print('MNIST GPU-CPU max abs error:',maxerr,flush=True)
assert last_loss < first_loss and correct/total>=0.90, 'MNIST learning threshold failed'
# Synthetic regression: known linear mapping, noise-free.
rng=torch.Generator().manual_seed(10); X=torch.randn(4096,16,generator=rng); W=torch.randn(16,4,generator=rng); Y=X@W+0.2
X=X.to(device); Y=Y.to(device); reg=nn.Linear(16,4).to(device); opt=torch.optim.Adam(reg.parameters(),lr=.05)
for step in range(1,201):
    opt.zero_grad(set_to_none=True); output=reg(X); loss=nn.functional.mse_loss(output,Y); loss.backward(); opt.step()
    if step in (1,50,100,200): print(f'Synthetic regression step {step}: mse={loss.item():.8f}',flush=True)
    if not torch.isfinite(loss): raise RuntimeError('nonfinite regression loss')
assert loss.item()<1e-5, 'Regression convergence failed'
torch.cuda.synchronize(); print('TRAINING TESTS PASSED',flush=True)
