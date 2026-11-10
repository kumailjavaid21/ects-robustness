"""Causal running normalization and a prefix-augmented anytime GRU."""
import random
import numpy as np
import torch
from torch import nn

def normalize(X,mode):
    x=np.asarray(X,dtype=np.float64)
    if x.ndim!=2 or not np.isfinite(x).all():raise ValueError('Expected finite [series,time] array')
    if mode=='standard':return x.astype(np.float32)
    if mode!='causal':raise ValueError('Unknown normalization mode')
    n=np.arange(1,x.shape[1]+1,dtype=np.float64)[None,:]
    mean=np.cumsum(x,axis=1)/n
    var=np.maximum(np.cumsum(x*x,axis=1)/n-mean*mean,0)
    std=np.sqrt(var)
    # At a constant prefix, no scale is identifiable; emit zero.
    z=np.divide(x-mean,std,out=np.zeros_like(x),where=std>1e-8)
    return z.astype(np.float32)

class AnytimeGRU(nn.Module):
    def __init__(self,classes,hidden=32):
        super().__init__()
        self.gru=nn.GRU(input_size=1,hidden_size=hidden,batch_first=True)
        self.head=nn.Linear(hidden,classes)
    def forward(self,x):
        h,_=self.gru(x.unsqueeze(-1))
        return self.head(h)

def train(X,y,device,seed=42,epochs=60,batch_size=64):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    classes,encoded=np.unique(y,return_inverse=True)
    Xt=torch.tensor(np.ascontiguousarray(X),dtype=torch.float32)
    yt=torch.tensor(encoded,dtype=torch.long)
    model=AnytimeGRU(len(classes)).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    generator=torch.Generator().manual_seed(seed);epoch_losses=[]
    for epoch in range(epochs):
        model.train();order=torch.randperm(len(Xt),generator=generator);losses=[]
        for ix in order.split(batch_size):
            batch=Xt[ix].to(device);labels=yt[ix].to(device);logits=model(batch)
            t=torch.randint(0,batch.shape[1],(len(ix),),generator=generator).to(device)
            augmented=nn.functional.cross_entropy(logits[torch.arange(len(ix),device=device),t],labels)
            # Include complete trajectories as well as one randomly sampled prefix per series.
            loss=augmented+.5*nn.functional.cross_entropy(logits[:,-1,:],labels)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),2.0);optimizer.step();losses.append(float(loss.detach().cpu()))
        epoch_losses.append(float(np.mean(losses)))
    return model,classes,epoch_losses

def probabilities(model,X,device,batch_size=128):
    out=[];model.eval()
    with torch.inference_mode():
        for ix in range(0,len(X),batch_size):
            x=torch.as_tensor(np.ascontiguousarray(X[ix:ix+batch_size]),dtype=torch.float32,device=device)
            out.append(model(x).softmax(-1).cpu().numpy())
    return np.concatenate(out,axis=0)
