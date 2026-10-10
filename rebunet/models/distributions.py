"""NB(mean, concentration), positive mean inverse, and normalized count marginals.
Numerical probability arithmetic is float64, independently of backbone dtype.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor


def zero_normalizer(mu: Tensor, k: Tensor):
    # Z = P_NB(Y > 0) = 1 - (k / (k + mu))**k; expm1 avoids cancellation.
    return -torch.expm1(-k * torch.log1p(mu / k))


def positive_mean(mu: Tensor, k: Tensor):
    # M(mu, k) = E[Y | Y > 0] = mu / Z; M tends to 1 as mu tends to 0.
    mu, k = mu.double(), k.double()
    z = zero_normalizer(mu, k)
    # Stable expansion around zero, with a relative smallness criterion.
    small = (mu < 1e-6) & (mu / k < 1e-5)
    series = 1 + (k+1)/(2*k)*mu + (k*k-1)/(12*k*k)*mu*mu
    return torch.where(small, series, mu / z.clamp_min(1e-300))


def mean_derivatives(mu, k):
    z = zero_normalizer(mu, k)
    p0 = torch.exp(-k * torch.log1p(mu/k))
    z_mu = p0 * k/(k+mu)
    z_k = p0 * (torch.log1p(mu/k) - mu/(k+mu))
    dm = (z-mu*z_mu) / z.square().clamp_min(1e-300)
    dk = -mu*z_k / z.square().clamp_min(1e-300)
    small = (mu < 1e-6) & (mu/k < 1e-5)
    dm = torch.where(small, (k+1)/(2*k)+(k*k-1)/(6*k*k)*mu, dm)
    dk = torch.where(small, -mu/(2*k*k)+mu*mu/(6*k**3), dk)
    return dm, dk


class _InverseMean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, m, k):
        dtype_m, dtype_k = m.dtype, k.dtype
        m, k = torch.broadcast_tensors(m.double(), k.double())
        if not bool(((m > 1) & (k > 0) & torch.isfinite(m) & torch.isfinite(k)).all()):
            raise ValueError('ZTNB inverse requires finite m>1 and kappa>0')
        lo, hi = torch.zeros_like(m), m.clone()
        mu = ((m-1)*2*k/(k+1)).minimum(m*0.9).clamp_min(1e-300)
        for _ in range(100):
            residual = positive_mean(mu, k) - m
            if residual.abs().max() <= 1e-10: break
            lo = torch.where(residual < 0, mu, lo)
            hi = torch.where(residual >= 0, mu, hi)
            derivative, _ = mean_derivatives(mu, k)
            newton = mu - residual/derivative.clamp_min(1e-300)
            valid = (newton > lo) & (newton < hi) & torch.isfinite(newton)
            mu = torch.where(valid, newton, (lo+hi)/2)
        if (positive_mean(mu,k)-m).abs().max() > 1e-5:
            raise FloatingPointError('Positive-mean inverse did not satisfy 1e-5 residual')
        ctx.save_for_backward(mu,k)
        ctx.dtypes = dtype_m, dtype_k
        return mu

    @staticmethod
    def backward(ctx, grad):
        mu,k = ctx.saved_tensors
        # Implicit differentiation of M(mu,k)=m: dmu/dm=1/M_mu; dmu/dk=-M_k/M_mu.
        dm,dk = mean_derivatives(mu,k)
        return (grad/dm).to(ctx.dtypes[0]), (-grad*dk/dm).to(ctx.dtypes[1])


def inverse_mean(m, k):
    # Expand outside the Function so autograd correctly reduces broadcast axes.
    m,k = torch.broadcast_tensors(m,k)
    return _InverseMean.apply(m,k)


def nb_log_prob(y, mu, k):
    y,mu,k = y.double(),mu.double(),k.double()
    return (torch.lgamma(y+k)-torch.lgamma(k)-torch.lgamma(y+1)
            -k*torch.log1p(mu/k)+y*(torch.log(mu.clamp_min(1e-300))-torch.log(k+mu)))


@dataclass
class Forecast:
    """p: B,H; w,mu,k: B,H,C. kind applies to all mixture components.
    'nb' uses p=1 and no hurdle; occurrence_probability accounts for NB zeros.
    'dlognormal' is floor(LogNormal)+1, where mu is location and k is scale.
    """
    p: Tensor
    w: Tensor
    mu: Tensor
    k: Tensor
    kind: str = 'ztnb'
    point_override: Tensor | None = None

    def component_log_prob(self, y):
        y = y.double().unsqueeze(-1)
        if self.kind == 'ztnb':
            return nb_log_prob(y,self.mu,self.k)-torch.log(zero_normalizer(self.mu,self.k))
        if self.kind == 'shifted_nb': return nb_log_prob((y-1).clamp_min(0),self.mu,self.k)
        if self.kind == 'nb': return nb_log_prob(y,self.mu,self.k)
        if self.kind == 'ztpoisson':
            mu=self.mu.double()
            return y*mu.log()-mu-torch.lgamma(y+1)-torch.log(-torch.expm1(-mu))
        if self.kind == 'dlognormal':
            loc,scale = self.mu.double(),self.k.double()
            high=(y.clamp_min(1).log()-loc)/scale
            low=((y-1).clamp_min(1e-300).log()-loc)/scale
            log_hi=torch.special.log_ndtr(high)
            log_lo=torch.special.log_ndtr(low)
            # Symmetric survival representation in the upper tail.
            a=torch.where(low>0,torch.special.log_ndtr(-low),log_hi)
            b=torch.where(low>0,torch.special.log_ndtr(-high),log_lo)
            return a+torch.log(-torch.expm1((b-a).clamp_max(-1e-15)))
        raise ValueError(self.kind)

    def log_prob(self, y):
        # Hurdle: P(0)=1-p; P(y>0)=p * sum_c w_c f_c(y).
        mix=torch.logsumexp(self.w.double().clamp_min(1e-300).log()+self.component_log_prob(y.clamp_min(1) if self.kind!='nb' else y),-1)
        if self.kind=='nb': return mix
        p=self.p.double().clamp(1e-12,1-1e-12)
        return torch.where(y>0,p.log()+mix,torch.log1p(-p))

    def mean(self):
        if self.point_override is not None: return self.point_override
        if self.kind=='ztnb': means=positive_mean(self.mu,self.k)
        elif self.kind=='shifted_nb': means=1+self.mu
        elif self.kind=='nb': means=self.mu
        elif self.kind=='ztpoisson': means=self.mu/(-torch.expm1(-self.mu))
        elif self.kind=='dlognormal':
            # E[floor(X)+1]=sum_{j>=0}P(X>=j); finite sum with explicit tail guard.
            loc,scale=self.mu.double(),self.k.double()
            cutoff=int(torch.ceil(torch.exp(loc+7*scale)).max().clamp(2,200000).item())
            means=torch.ones_like(loc)
            for start in range(1,cutoff+1,256):
                grid=torch.arange(start,min(start+256,cutoff+1),device=loc.device,dtype=loc.dtype)
                means=means+torch.special.ndtr((loc[...,None]-grid.log())/scale[...,None]).sum(-1)
            tail=torch.exp(loc+scale.square()/2)*torch.special.ndtr((loc+scale.square()-torch.log(torch.tensor(float(cutoff),device=loc.device)))/scale)
            if tail.max()>1e-5: raise FloatingPointError('Discretized lognormal mean tail exceeds tolerance')
        else: raise ValueError(self.kind)
        # Total expectation: E[Y] = p * sum_c w_c E[Y | Y>0, component=c].
        return self.p*(self.w*means).sum(-1)

    def occurrence_probability(self):
        if self.kind!='nb': return self.p
        return 1-(self.w*torch.exp(-self.k*torch.log1p(self.mu/self.k))).sum(-1)

    def numpy(self):
        return {key:getattr(self,key).detach().cpu().numpy() for key in ['p','w','mu','k']}


def masked_nll(forecast, y, mask):
    # L = -sum(mask * log P(y)) / sum(mask); unavailable dates are not zero sales.
    losses=-forecast.log_prob(y)
    if not bool(torch.isfinite(losses[mask.bool()]).all()): raise FloatingPointError('Nonfinite supervised NLL')
    return losses.masked_select(mask.bool()).mean() if mask.any() else losses.sum()*0
