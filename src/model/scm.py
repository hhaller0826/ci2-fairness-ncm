import itertools

import numpy as np
import torch as T
import torch.nn as nn

from src.model.distribution import Distribution

def log(x):
    return T.log(x + 1e-8)

class SCM(nn.Module):
    def __init__(self, v, f, pu: Distribution):
        super().__init__()
        self.v = v
        self.u = list(pu)
        self.f = f
        self.pu = pu
        self.device_param = nn.Parameter(T.empty(0))

    def space(self, v_size, select=None, tensor=True):
        if select is None:
            select = self.v
        for pairs in itertools.product(*([
            (vi, T.LongTensor(value).to(self.device_param.device) if tensor else value)
            for value in itertools.product(*([0, 1] for j in range(v_size[vi])))]
                for vi in select)):
            yield dict(pairs)

    def sample(self, n=None, u=None, do={}, select=None):
        assert not set(do.keys()).difference(self.v)
        assert (n is None) != (u is None)

        for k in do:
            do[k] = do[k].to(self.device_param)

        if u is None:
            u = self.pu.sample(n)
        if select is None:
            select = self.v
        v = {}
        remaining = set(select)
        for k in self.v:
            v[k] = do[k] if k in do else self.f[k](v, u)
            remaining.discard(k)
            if not remaining:
                break
        return {k: v[k] for k in select}

    def convert_evaluation(self, samples):
        return samples

    def forward(self, n=None, u=None, do={}, select=None, evaluating=False):
        if evaluating:
            with T.no_grad():
                result = self.sample(n, u, do, select)
                result = self.convert_evaluation(result)
                return {k: result[k].cpu() for k in result}

        return self.sample(n, u, do, select)

    def query_loss(self, input, val):
        if T.is_tensor(val):
            raise NotImplementedError()
        else:
            if val == 1:
                return T.sum(-log(input))
            elif val == 0:
                return T.sum(-log(1 - input))
            else:
                raise ValueError("Comparison to {} of type {} is not allowed.".format(val, type(val)))
