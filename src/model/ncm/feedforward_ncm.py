import torch as T
import torch.nn as nn

from src.model.scm import SCM
from src.model.distribution import *
from src.model.ncm.mlp import MLP

"""NOTE: WE MAY NEED TO CHANGE THIS
When initializing FF_NCM and MLP they have things like u_size, v_size, o_size 
These correspond to the "size" of a single node. I think that this means like if node Z={"age","sex","race"} then its size would be 3. 
Right now I am keeping everything as size=1 and we might just have to change that later.
"""

class FF_NCM(SCM):
    def __init__(self, cg, v_size={}, default_v_size=1, u_size={},
                 default_u_size=1, f={}, hyperparams=None, discrete_vals=None, scale={}):
        if hyperparams is None:
            hyperparams = dict()
        self.discrete_vals = discrete_vals if discrete_vals is not None else cg.v

        self.cg = cg
        # vassign = cg.assignments.values()

        self.u_size = {k: u_size.get(k, default_u_size) for k in self.cg.c2}
        self.v_size = {k: v_size.get(k, default_v_size) for k in self.cg}
        self.scale = {k: scale.get(k, (lambda x: x)) for k in self.cg}
        super().__init__(
            v=cg.v,
            f=nn.ModuleDict({
                v: f[v] if v in f else MLP(
                    {k: self.v_size[k] for k in self.cg.pa[v]},
                    {k: self.u_size[k] for k in self.cg.v2c2[v]},
                    self.v_size[v],
                    h_size=hyperparams.get('h-size', 128)
                )
                for v in cg}),
            pu=UniformDistribution(self.cg.c2, self.u_size))

    def convert_evaluation(self, samples):
        ret = {}
        for k in samples:
            x = samples[k]
            scaled = T.tensor([[self.scale[k][i](x[j][i]).item() for i in range(len(x[0]))] for j in range(len(x))])
            if k in self.discrete_vals:
                ret[k] = T.round(scaled)
            else:
                ret[k] = scaled
        return ret
        # return {k: T.round(samples[k]) if k in self.discrete_vals else samples[k] for k in samples}


