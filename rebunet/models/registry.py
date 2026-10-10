import importlib
from rebunet.models.rebunet import ReBuNet
from rebunet.baselines.point import PointForecaster
from rebunet.baselines.neural import DeepAR,SwitchHurdle

CLASSICAL=['croston','sba','tsb','adida']
NATIVE=['rebunet','dlinear','patchtst','itransformer','deepar','switch_hurdle']
EXTERNAL=['robust_rnn','transformer_idf','deep_renewal','deep_state','score_hurdle_nb']


def build(cfg,dims):
    model=cfg['model'];name=model['name']
    if name=='rebunet':return ReBuNet(model,dims)
    if name in ['dlinear','patchtst','itransformer']:return PointForecaster(model,dims)
    if name=='deepar':return DeepAR(model,dims)
    if name=='switch_hurdle':return SwitchHurdle(model,dims)
    # External adapter must implement the documented model interface. No substitute model.
    spec=cfg.get('external',{}).get(name,{}).get('factory')
    if not spec:raise RuntimeError(f'{name}: original implementation/configuration required; see README.md#external-baselines')
    module,function=spec.rsplit(':',1);return getattr(importlib.import_module(module),function)(model,dims)
