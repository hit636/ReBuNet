"""SciPy marginal scoring. DRPS includes an explicit certified omitted-tail bound."""
import numpy as np
from scipy.stats import nbinom,poisson,lognorm


class CountMixture:
    def __init__(self,p,w,mu,k,kind):
        self.p=np.asarray(p,float).reshape(-1);self.w=np.asarray(w,float).reshape(len(self.p),-1)
        self.mu=np.asarray(mu,float).reshape(self.w.shape);self.k=np.asarray(k,float).reshape(self.w.shape);self.kind=kind
        if kind not in ['ztnb','shifted_nb','nb','ztpoisson','dlognormal']:raise ValueError(kind)
        if not np.isfinite(self.mu).all() or not np.isfinite(self.k).all() or (self.k<=0).any():raise ValueError('Invalid distribution parameters')
        if (self.w<0).any() or not np.allclose(self.w.sum(-1),1,atol=1e-7) or ((self.p<0)|(self.p>1)).any():raise ValueError('Unnormalized forecast')
        if kind=='nb' and not np.allclose(self.p,1):raise ValueError('Ordinary NB mixture requires p=1')
        if kind!='dlognormal' and (self.mu<=0).any():raise ValueError('Count mean/rate must be positive')
        self.prob=self.k/(self.k+self.mu) if kind in ['nb','ztnb','shifted_nb'] else None
        self.z=-np.expm1(-self.k*np.log1p(self.mu/self.k)) if kind=='ztnb' else -np.expm1(-self.mu) if kind=='ztpoisson' else 1

    def cdf(self,x):
        x=np.asarray(x);x=x[...,None]
        if self.kind in ['nb','ztnb']:value=nbinom.cdf(x,self.k,self.prob)
        elif self.kind=='shifted_nb':value=nbinom.cdf(x-1,self.k,self.prob)
        elif self.kind=='ztpoisson':value=poisson.cdf(x,self.mu)
        else:value=lognorm.cdf(np.maximum(x,0),self.k,scale=np.exp(self.mu))
        if self.kind in ['ztnb','ztpoisson']:value=np.maximum(0,(value-(1-self.z))/self.z)
        mix=np.sum(self.w*value,axis=-1)
        result=mix if self.kind=='nb' else 1-self.p+self.p*mix
        return np.where(np.squeeze(x,-1)<0,0,result).clip(0,1)

    def log_prob(self,y):
        from scipy.special import logsumexp
        yy=np.asarray(y).reshape(-1,1)
        if self.kind in ['nb','ztnb']:lp=nbinom.logpmf(yy,self.k,self.prob)
        elif self.kind=='shifted_nb':lp=nbinom.logpmf(yy-1,self.k,self.prob)
        elif self.kind=='ztpoisson':lp=poisson.logpmf(yy,self.mu)
        else:
            hi=lognorm.logcdf(yy,self.k,scale=np.exp(self.mu));lo=lognorm.logcdf(yy-1,self.k,scale=np.exp(self.mu))
            a=lognorm.logsf(yy-1,self.k,scale=np.exp(self.mu));b=lognorm.logsf(yy,self.k,scale=np.exp(self.mu))
            upper=yy>np.exp(self.mu);aa=np.where(upper,a,hi);bb=np.where(upper,b,lo)
            with np.errstate(divide='ignore',invalid='ignore'):lp=aa+np.log(-np.expm1(bb-aa))
        if self.kind in ['ztnb','ztpoisson']:lp=lp-np.log(self.z)
        with np.errstate(divide='ignore'):
            mixed=logsumexp(np.log(self.w)+lp,axis=-1)
            if self.kind=='nb':return mixed
            return np.where(yy[:,0]>0,np.log(self.p)+mixed,np.log1p(-self.p))

    def tail_excess_bound(self,K):
        """Upper bound for omitted sum_{j>K}(1-F(j))^2 when K>=observed y.
        <= E[(Y-(K+1))_+]; lognormal uses a looser continuous excess bound.
        """
        c=K+1
        if self.kind in ['nb','ztnb','shifted_nb']:
            shift=1 if self.kind=='shifted_nb' else 0;threshold=c-shift
            excess=self.mu*nbinom.sf(threshold-1,self.k+1,self.prob)-threshold*nbinom.sf(threshold,self.k,self.prob)
            if self.kind=='ztnb':excess=excess/self.z
        elif self.kind=='ztpoisson':
            excess=(self.mu*poisson.sf(c-1,self.mu)-c*poisson.sf(c,self.mu))/self.z
        else:
            # Y=floor(X)+1 => (Y-c)+ <= (X-(c-1))+.
            threshold=max(c-1,1)
            from scipy.special import ndtr
            excess=np.exp(self.mu+self.k**2/2)*ndtr((self.mu+self.k**2-np.log(threshold))/self.k)-threshold*lognorm.sf(threshold,self.k,scale=np.exp(self.mu))
        p=1 if self.kind=='nb' else self.p
        return p*np.sum(self.w*np.maximum(excess,0),-1)

    def quantile(self,q):
        n=len(self.p);lo=np.full(n,-1,dtype=np.int64);hi=np.ones(n,dtype=np.int64)
        for _ in range(40):
            grow=self.cdf(hi)<q
            if not grow.any():break
            hi[grow]*=2
        if (self.cdf(hi)<q).any():raise FloatingPointError('Quantile not bracketed')
        while (hi-lo>1).any():
            mid=(lo+hi)//2;left=self.cdf(mid)>=q
            hi=np.where(left,mid,hi);lo=np.where(left,lo,mid)
        return hi

    def drps(self,y,tolerance=1e-6,max_support=200000,chunk=256):
        y=np.asarray(y).reshape(-1)
        K=max(32,int(y.max()),int(self.quantile(.999).max()))
        K=min(K,max_support)
        while K<max_support and self.tail_excess_bound(K).max()>tolerance:K=min(max_support,2*K)
        bound=self.tail_excess_bound(K)
        if K<y.max() or not np.isfinite(bound).all() or bound.max()>tolerance:
            raise FloatingPointError(f'DRPS support={K} insufficient; tail bound={bound.max()}; increase metrics.max_support')
        total=np.zeros(len(y))
        for start in range(0,K+1,chunk):
            grid=np.arange(start,min(start+chunk,K+1))[:,None]
            total+=np.sum((self.cdf(grid)-(y[None]<=grid))**2,axis=0)
        return total,bound,K
