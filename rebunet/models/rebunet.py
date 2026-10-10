from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from rebunet.models.encoders import DualTimeEncoder
from rebunet.models.distributions import Forecast,inverse_mean,positive_mean


def mlp(inputs,d,outputs=1): return nn.Sequential(nn.Linear(inputs,d),nn.SiLU(),nn.Linear(d,outputs))


class MarkHead(nn.Module):
    def __init__(self,inputs,d,mode,delta):
        super().__init__();self.mode,self.delta=mode,delta
        self.a=mlp(inputs,d);self.b=mlp(inputs,d)
        self.k0=mlp(inputs,d);self.k1=mlp(inputs,d);self.gate=mlp(inputs,d)
    def warmup(self,active):
        # The paper leaves the precise freeze scope ambiguous; freeze high component dispersion.
        for p in self.k1.parameters(): p.requires_grad_(not active)
        for p in self.gate.parameters(): p.requires_grad_(not active)
        self.is_warm=active
    def forward(self,q):
        a,b=self.a(q).squeeze(-1),self.b(q).squeeze(-1)
        k0,k1=F.softplus(self.k0(q).squeeze(-1)).double()+1e-4,F.softplus(self.k1(q).squeeze(-1)).double()+1e-4
        mode=self.mode;kind='ztnb'
        if mode=='ordered':
            # Conditional means: m0=1+softplus(a), m1=m0+softplus(b)+delta.
            # Invert M(mu,k)=m separately; ordering raw NB means is insufficient.
            m0=1+F.softplus(a.double());m1=m0+F.softplus(b.double())+self.delta
            mu0,mu1=inverse_mean(m0,k0),inverse_mean(m1,k1)
        elif mode in ['single','unordered','raw_ordered']:
            mu0=F.softplus(a.double())+1e-8
            mu1=mu0+F.softplus(b.double())+self.delta if mode=='raw_ordered' else F.softplus(b.double())+1e-8
        elif mode=='shifted_nb':
            kind='shifted_nb';mu0=F.softplus(a.double())+1e-8;mu1=mu0+F.softplus(b.double())+self.delta
        elif mode=='ztpoisson':
            kind='ztpoisson';mu0=F.softplus(a.double())+1e-8;mu1=mu0;k0=k1=torch.ones_like(mu0)
        elif mode=='dlognormal':
            kind='dlognormal';mu0=a.double();mu1=mu0;k0=F.softplus(self.k0(q).squeeze(-1).double())+.05;k1=k0
        else:raise ValueError(mode)
        single=mode in ['single','ztpoisson','dlognormal']
        if single:
            return torch.ones_like(mu0)[...,None],mu0[...,None],k0[...,None],kind
        g=torch.full_like(mu0,.01) if getattr(self,'is_warm',False) else torch.sigmoid(self.gate(q).squeeze(-1).double())
        return torch.stack([1-g,g],-1),torch.stack([mu0,mu1],-1),torch.stack([k0,k1],-1),kind


class ReBuNet(nn.Module):
    probabilistic=True
    def __init__(self,cfg,dims):
        super().__init__();self.cfg=cfg;d=cfg['hidden'];self.state=cfg.get('state','scalar');self.cap=cfg.get('cap',128)
        self.encoder=DualTimeEncoder(cfg,dims);self.horizon=nn.Embedding(cfg['horizon'],cfg.get('horizon_dim',16))
        self.adim=cfg.get('age_dim',16)
        if self.state!='direct':self.age_encoder=nn.Sequential(nn.Linear(1,self.adim),nn.Tanh(),nn.Linear(self.adim,self.adim))
        qdim=d+dims['known_dim']+self.horizon.embedding_dim+(0 if self.state=='direct' else self.adim)
        self.hazard=mlp(qdim,d);self.mark=MarkHead(qdim,d,cfg['mark'],cfg.get('delta',1e-3))
        if self.state=='grs':self.grs=nn.GRUCell(1+dims['known_dim']+self.horizon.embedding_dim,self.adim)
    def warmup(self,active): self.mark.warmup(active)
    def age_features(self,age):return self.age_encoder(torch.log1p(age.to(self.horizon.weight.dtype))[...,None])
    def forward(self,batch,state_override=None,return_state=False):
        state=state_override or self.state
        if state=='direct' and self.state!='direct':raise ValueError('Direct decoder has different head dimensions; fit it separately')
        c=self.encoder(batch);b=c.shape[0];r=batch['age'].double();initial=r
        ages=r[:,None];mass=torch.ones_like(ages)
        if state=='cad':
            mass=torch.zeros((b,self.cap+1),device=c.device,dtype=torch.float64)
            mass.scatter_(1,r.long().clamp_max(self.cap)[:,None],1)
            ages=torch.arange(self.cap+1,device=c.device,dtype=torch.float64)[None].expand(b,-1)
        if state=='grs':hidden=torch.zeros(b,self.adim,device=c.device)
        ps=[];ws=[];mus=[];ks=[];age_history=[]
        for h in range(batch['future'].shape[1]):
            he=self.horizon.weight[h][None].expand(b,-1);future=batch['future'][:,h]
            base=torch.cat([c,future,he],-1)
            if state in ['ead','cad']:
                r=(mass*ages).sum(-1)
                qa=torch.cat([base[:,None].expand(-1,ages.shape[1],-1),self.age_features(ages)],-1)
                hazards=torch.sigmoid(self.hazard(qa).squeeze(-1).double())
                p=(mass*hazards).sum(-1)
                q=torch.cat([base,self.age_features(r)],-1)
            elif state=='direct':q=base;p=torch.sigmoid(self.hazard(q).squeeze(-1).double())
            elif state=='grs':q=torch.cat([base,hidden],-1);p=torch.sigmoid(self.hazard(q).squeeze(-1).double())
            else:
                q=torch.cat([base,self.age_features(r)],-1);p=torch.sigmoid(self.hazard(q).squeeze(-1).double())
            w,mu,k,kind=self.mark(q);ps.append(p);ws.append(w);mus.append(mu);ks.append(k);age_history.append(r)
            # EAD: reset mass pi_next(0)=sum_a pi(a)*hazard(a);
            # surviving mass pi(a)*(1-hazard(a)) moves to age a+1.
            if state=='ead':
                mass=torch.cat([p[:,None],mass*(1-hazards)],-1)
                ages=torch.cat([torch.zeros_like(ages[:,:1]),ages+1],-1)
            elif state=='cad':
                # CAD merges overflow into the final age bucket to conserve mass.
                survival=mass*(1-hazards)
                mass=torch.cat([p[:,None],survival[:,:-2],(survival[:,-2]+survival[:,-1])[:,None]],-1)
            elif state=='grs':hidden=self.grs(torch.cat([p.float()[:,None],future,he],-1),hidden)
            elif state=='fixed_age':r=initial
            elif state=='no_feedback':r=initial+h+1
            elif state=='detached':r=((1-p)*(r+1)).detach()
            # Mean-state approximation: r_next=(1-p)*(r+1); this is not EAD.
            elif state=='scalar':r=(1-p)*(r+1)
            elif state!='direct':raise ValueError(state)
        result=Forecast(torch.stack(ps,1),torch.stack(ws,1),torch.stack(mus,1),torch.stack(ks,1),kind)
        return (result,torch.stack(age_history,1)) if return_state else result
