import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class StaticEmbedding(nn.Module):
    def __init__(self,cards,width):
        super().__init__();self.tables=nn.ModuleList([nn.Embedding(int(c),width) for c in cards]);self.out_dim=width*len(cards)
    def forward(self,x): return torch.cat([table(x[:,j]) for j,table in enumerate(self.tables)],-1)


class GatedCausalBlock(nn.Module):
    def __init__(self,d,dilation,dropout):
        super().__init__();self.pad=2*dilation
        self.conv=nn.Conv1d(d,2*d,3,padding=self.pad,dilation=dilation)
        self.project=nn.Conv1d(d,d,1);self.drop=nn.Dropout(dropout)
        self.norm=nn.LayerNorm(d)
    def forward(self,x):
        a,b=self.conv(x)[...,:x.shape[-1]].chunk(2,1)
        out=x+self.drop(self.project(torch.tanh(a)*torch.sigmoid(b)))
        return self.norm(out.transpose(1,2)).transpose(1,2)


class DualTimeEncoder(nn.Module):
    def __init__(self,cfg,dims):
        super().__init__();d=cfg['hidden'];self.use_events=cfg.get('use_events',True)
        self.static=StaticEmbedding(dims['static_cardinalities'],cfg.get('static_dim',8))
        self.input=nn.Linear(2+dims['known_dim']+dims['historical_dim']+self.static.out_dim,d)
        self.tcn=nn.Sequential(*[GatedCausalBlock(d,2**j,cfg['dropout']) for j in range(cfg['tcn_layers'])])
        if self.use_events:
            self.event=nn.GRU(2+dims['known_dim']+dims['historical_dim'],d,batch_first=True)
            self.gate=nn.Linear(2*d+self.static.out_dim,d)
    def forward(self,batch):
        static=self.static(batch['static']);hist=batch['history'];l=hist.shape[1]
        inputs=torch.cat([hist,static[:,None].expand(-1,l,-1)],-1)
        cal=self.tcn(self.input(inputs).transpose(1,2))[...,-1]
        if not self.use_events:return cal
        lengths=batch['event_length']
        packed=pack_padded_sequence(batch['events'],lengths.clamp_min(1).cpu(),batch_first=True,enforce_sorted=False)
        _,hidden=self.event(packed);evt=hidden[-1]*((lengths>0).to(hist.dtype)[:,None])
        gamma=torch.sigmoid(self.gate(torch.cat([cal,evt,static],-1)))
        return gamma*cal+(1-gamma)*evt
