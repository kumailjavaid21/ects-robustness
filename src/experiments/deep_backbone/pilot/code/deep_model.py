"""Strictly causal TCN and shared/progressive/independent checkpoint heads."""
import torch
from torch import nn
from torch.nn import functional as F

class CausalConv(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__();self.pad=2*dilation
        self.conv=nn.Conv1d(channels,channels,3,dilation=dilation)
    def forward(self,x):return self.conv(F.pad(x,(self.pad,0)))

class CausalBlock(nn.Module):
    def __init__(self,channels,dilation,dropout):
        super().__init__()
        self.c1=CausalConv(channels,dilation);self.c2=CausalConv(channels,dilation)
        # LayerNorm across channels at ONE timestep; never normalize over time.
        self.n1=nn.LayerNorm(channels);self.n2=nn.LayerNorm(channels)
        self.drop=nn.Dropout(dropout)
    def forward(self,x):
        z=self.c1(x);z=self.n1(z.transpose(1,2)).transpose(1,2)
        z=self.drop(F.gelu(z));z=self.c2(z)
        z=self.n2(z.transpose(1,2)).transpose(1,2)
        return F.gelu(x+self.drop(z))

class AnytimeTCN(nn.Module):
    def __init__(self,classes,mode,checkpoints=9,channels=64,dropout=.1):
        super().__init__();self.mode=mode;self.channels=channels
        self.input=nn.Conv1d(1,channels,1)
        self.blocks=nn.Sequential(*[CausalBlock(channels,d,dropout) for d in (1,2,4,8,16)])
        self.heads=nn.ModuleList([nn.Linear(channels*8,classes) for _ in range(1 if mode=='A' else checkpoints)])
    def encode(self,x,times):
        h=self.blocks(self.input(x[:,None,:])).float()
        c=F.pad(torch.cumsum(h,dim=-1),(1,0))
        features=[]
        for t in times:
            t=int(t);bins=[]
            # A causal temporal pyramid (1,2,4 bins) retains prefix shape/order.
            for n in (1,2,4):
                for i in range(n):
                    a=i*t//n;b=(i+1)*t//n
                    bins.append((c[:,:,b]-c[:,:,a])/max(b-a,1))
            bins.append(h[:,:,t-1])
            features.append(torch.cat(bins,dim=1))
        return torch.stack(features,dim=1)
    def forward(self,x,times):
        f=self.encode(x,times)
        if self.mode=='A':return self.heads[0](f)
        return torch.stack([self.heads[j](f[:,j]) for j in range(len(times))],1)
    def inherit(self,j):
        if self.mode=='B' and j>0:self.heads[j].load_state_dict(self.heads[j-1].state_dict())

def objective(logits,y,lam):
    k=logits.shape[1]
    ce=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),y[:,None].expand(-1,k).reshape(-1))
    consistency=logits.new_tensor(0.,dtype=torch.float32)
    if k>1 and lam:
        p=logits.float().softmax(-1);a=p[:,:-1];b=p[:,1:];m=(a+b)/2
        consistency=.5*((a*(a.clamp_min(1e-8).log()-m.clamp_min(1e-8).log())).sum(-1).mean()+
                         (b*(b.clamp_min(1e-8).log()-m.clamp_min(1e-8).log())).sum(-1).mean())
    return ce+lam*consistency,ce,consistency
