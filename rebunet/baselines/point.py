"""Architecture-level reimplementations; original paper tuning is unavailable.
DLinear is per-target decomposition; PatchTST uses target patches; iTransformer
uses variates as tokens. Covariate adaptations are disclosed in documentation.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PointForecaster(nn.Module):
    probabilistic=False
    def __init__(self,cfg,dims):
        super().__init__();self.cfg=cfg;self.name=cfg['name'];l,h,d=cfg['lookback'],cfg['horizon'],cfg['hidden']
        self.future=nn.Linear(dims['known_dim'],1)
        if self.name=='dlinear':
            self.seasonal=nn.Linear(l,h);self.trend=nn.Linear(l,h)
        elif self.name=='patchtst':
            self.patch=min(16,l);self.stride=max(1,self.patch//2);n=(l-self.patch)//self.stride+1
            self.input=nn.Linear(self.patch,d);self.pos=nn.Parameter(torch.randn(1,n,d)*.01)
            self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,d*4,cfg['dropout'],batch_first=True),2)
            self.head=nn.Linear(n*d,h)
        elif self.name=='itransformer':
            self.input=nn.Linear(l,d)
            self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(d,4,d*4,cfg['dropout'],batch_first=True),2)
            self.head=nn.Linear(d,h)
        else:raise ValueError(self.name)
    def forward(self,b):
        x=torch.expm1(b['history'][...,0]);mask=b['history'][...,1]
        count=mask.sum(1,keepdim=True).clamp_min(1)
        mean=(x*mask).sum(1,keepdim=True)/count
        std=(((x-mean)**2*mask).sum(1,keepdim=True)/count+1).sqrt()
        z=(x-mean)/std;z=torch.where(mask.bool(),z,torch.zeros_like(z))
        if self.name=='dlinear':
            kernel=min(25,self.cfg['lookback']);kernel=kernel if kernel%2 else kernel-1
            trend=F.avg_pool1d(F.pad(z[:,None],(kernel//2,kernel//2),mode='replicate'),kernel,1)[:,0]
            pred=self.trend(trend)+self.seasonal(z-trend)
        elif self.name=='patchtst':
            patches=z.unfold(1,self.patch,self.stride)
            pred=self.head(self.encoder(self.input(patches)+self.pos).flatten(1))
        else:
            # Inverted tokens: target + availability + known/historical covariates.
            hist=b['history'].clone();hist[:,:,0]=z
            pred=self.head(self.encoder(self.input(hist.transpose(1,2))))[:,0]
        return (pred*std+mean+self.future(b['future']).squeeze(-1)).clamp_min(0)
