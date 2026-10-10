"""Classical point algorithms. Missing days advance clocks, never become observed zeros."""
import numpy as np


def forecast(history,available,horizon,method,alpha=.1,beta=.1,aggregation=7):
    y=np.asarray(history,dtype=float);available=np.asarray(available,dtype=bool)
    pos=np.flatnonzero((y>0)&available)
    if not len(pos):return np.zeros(horizon)
    first=pos[0];size=y[first];interval=first+1.;prev=first
    if method in ['croston','sba']:
        for t in pos[1:]:
            size=alpha*y[t]+(1-alpha)*size
            interval=alpha*(t-prev)+(1-alpha)*interval;prev=t
        val=size/interval*(1-alpha/2 if method=='sba' else 1)
    elif method=='tsb':
        probability=1/(first+1)
        for t in range(first+1,len(y)):
            if not available[t]:continue
            occurs=float(y[t]>0);probability=beta*occurs+(1-beta)*probability
            if occurs:size=alpha*y[t]+(1-alpha)*size
        val=probability*size
    elif method=='adida':
        # Fixed aggregation selected on validation; incomplete/unavailable blocks omitted.
        blocks=[]
        for end in range(len(y),aggregation-1,-aggregation):
            start=end-aggregation
            if available[start:end].all():blocks.append(y[start:end].sum())
        blocks=blocks[::-1]
        if not blocks:raise ValueError('ADIDA has no fully observed aggregation blocks')
        level=blocks[0]
        for yy in blocks[1:]:level=alpha*yy+(1-alpha)*level
        val=level/aggregation
    else:raise ValueError(method)
    return np.full(horizon,val)
