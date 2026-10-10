"""Independently implemented DeepAR and Switch-Hurdle references.
They are not the unpublished baseline configurations used in ReBuNet's tables.
"""
import torch
from torch import nn
from torch.nn import functional as F
from rebunet.models.encoders import StaticEmbedding
from rebunet.models.distributions import Forecast,positive_mean


class DeepAR(nn.Module):
    probabilistic=True
    def __init__(self,cfg,dims):
        super().__init__();self.cfg=cfg;d=cfg['hidden']
        self.static=StaticEmbedding(dims['static_cardinalities'],cfg.get('static_dim',8))
        self.encoder=nn.GRU(2+dims['known_dim']+dims['historical_dim']+self.static.out_dim,d,batch_first=True)
        self.decoder=nn.GRUCell(1+dims['known_dim']+self.static.out_dim,d)
        self.head=nn.Linear(d,2)
    def forward(self,b):
        stat=self.static(b['static']);hist=b['history'];l=hist.shape[1]
        inp=torch.cat([hist,stat[:,None].expand(-1,l,-1)],-1);_,hh=self.encoder(inp);hidden=hh[-1]
        previous=torch.expm1(hist[:,-1,0]);future=b['future'];B,H=future.shape[:2]
        # Autoregressive forecasts integrate future trajectories by Monte Carlo.
        samples=1 if self.training else self.cfg.get('mc_samples',64)
        hidden=hidden.repeat_interleave(samples,0);stat=stat.repeat_interleave(samples,0)
        previous=previous.repeat_interleave(samples,0);mu_list=[];k_list=[]
        for h in range(H):
            x=torch.cat([torch.log1p(previous)[:,None],future[:,h].repeat_interleave(samples,0),stat],-1)
            hidden=self.decoder(x,hidden);pars=F.softplus(self.head(hidden).double())+1e-5
            mu,k=pars.unbind(-1);mu_list.append(mu.reshape(B,samples));k_list.append(k.reshape(B,samples))
            if self.training:
                previous=torch.where(b['mask'][:,h],b['y'][:,h],mu.float())
            else:
                distribution=torch.distributions.NegativeBinomial(total_count=k,logits=torch.log(mu/k))
                previous=distribution.sample().float()
        mu=torch.stack(mu_list,1);k=torch.stack(k_list,1)
        return Forecast(torch.ones(B,H,device=mu.device),torch.full_like(mu,1/samples),mu,k,'nb')


class SwiGLU(nn.Module):
    def __init__(self,d):
        super().__init__();self.a=nn.Linear(d,4*d);self.b=nn.Linear(d,4*d);self.c=nn.Linear(4*d,d)
    def forward(self,x):return self.c(F.silu(self.a(x))*self.b(x))


class SwitchLayer(nn.Module):
    def __init__(self,d,experts=4):
        super().__init__();self.router=nn.Linear(d,experts);self.experts=nn.ModuleList([SwiGLU(d) for _ in range(experts)])
        self.norm=nn.LayerNorm(d);self.aux=None
    def forward(self,x):
        probs=torch.softmax(self.router(x),-1);choice=probs.argmax(-1)
        if self.training:
            gate=F.one_hot(choice,len(self.experts)).to(probs)+probs-probs.detach()
            yy=torch.stack([expert(x) for expert in self.experts],-2)
            result=(yy*gate[...,None]).sum(-2)
        else:
            flat=x.reshape(-1,x.shape[-1]);which=choice.reshape(-1);result=torch.zeros_like(flat)
            for j,expert in enumerate(self.experts):
                selected=which==j
                if selected.any():result[selected]=expert(flat[selected])
            result=result.reshape_as(x)
        mean=probs.mean(dim=(0,1));self.aux=(mean*(mean.clamp_min(1e-12)*len(self.experts)).log()).sum()
        return self.norm(x+result)


class SwitchHurdle(nn.Module):
    probabilistic=True
    def __init__(self,cfg,dims):
        super().__init__();self.cfg=cfg;d=cfg['hidden']
        self.static=StaticEmbedding(dims['static_cardinalities'],cfg.get('static_dim',8))
        self.input=nn.Linear(2+dims['known_dim']+dims['historical_dim']+self.static.out_dim,d)
        self.position=nn.Parameter(torch.randn(1,cfg['lookback'],d)*.01)
        self.layers=nn.ModuleList([SwitchLayer(d,cfg.get('experts',4)) for _ in range(2)])
        self.previous=nn.Linear(1,d);self.future=nn.Linear(dims['known_dim'],d)
        self.horizon=nn.Embedding(cfg['horizon'],d);self.cross=nn.MultiheadAttention(d,4,batch_first=True)
        self.norm=nn.LayerNorm(d);self.head=nn.Linear(d,3);self.teacher_ratio=0.0
    def forward(self,b):
        stat=self.static(b['static']);hist=b['history'];B,L=hist.shape[:2]
        memory=self.input(torch.cat([hist,stat[:,None].expand(-1,L,-1)],-1))+self.position
        for layer in self.layers:memory=layer(memory)
        previous=torch.expm1(hist[:,-1,0]);ps=[];mus=[];ks=[]
        for h in range(b['future'].shape[1]):
            query=self.previous(torch.log1p(previous)[:,None])+self.future(b['future'][:,h])+self.horizon.weight[h]
            attended,_=self.cross(query[:,None],memory,memory,need_weights=False)
            p_raw,mu_raw,k_raw=self.head(self.norm(query+attended[:,0])).unbind(-1)
            p=torch.sigmoid(p_raw.double());mu=F.softplus(mu_raw.double())+1e-8;k=F.softplus(k_raw.double())+1e-4
            ps.append(p);mus.append(mu[:,None]);ks.append(k[:,None])
            previous=(p*positive_mean(mu,k)).float()
            if self.training and self.teacher_ratio>0:
                use=(torch.rand(B,device=hist.device)<self.teacher_ratio)&b['mask'][:,h]
                previous=torch.where(use,b['y'][:,h],previous)
        mu=torch.stack(mus,1)
        return Forecast(torch.stack(ps,1),torch.ones_like(mu),mu,torch.stack(ks,1),'ztnb')
    def auxiliary_loss(self):return self.cfg.get('balance_weight',.01)*sum(layer.aux for layer in self.layers)
