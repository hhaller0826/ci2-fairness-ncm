import torch as T
import torch.nn as nn
import numpy as np

# Code from NCMCounterfactuals

class Distribution(nn.Module):
    def __init__(self, u):
        super().__init__()
        self.u = u
        self.device_param = nn.Parameter(T.empty(0))

    def __iter__(self):
        return iter(self.u)

    def sample(self, n=1, device='cpu'):
        raise NotImplementedError()

    def forward(self, n=1):
        return self.sample(n=n)
    
class UniformDistribution(Distribution):
    def __init__(self, u_names, sizes, seed=None):
        assert set(sizes.keys()).issubset(set(u_names))

        super().__init__(list(u_names))
        self.sizes = {U: sizes[U] if U in sizes else 1 for U in u_names}
        if seed is not None:
            self.rand_state = np.random.RandomState(seed=seed)
        else:
            self.rand_state = np.random.RandomState()

    def sample(self, n=1, device=None):
        if device is None:
            device = self.device_param.device

        u_vals = dict()
        for U in self.sizes:
            u_vals[U] = T.from_numpy(self.rand_state.rand(n, self.sizes[U])).float().to(device)

        return u_vals
    

class NeuralDistribution(Distribution):
    def __init__(self, u_names, sizes, hyperparams, default_module=MLP, seed=None):
        assert set(sizes.keys()).issubset(set(u_names))

        super().__init__(list(u_names))
        self.sizes = {U: sizes[U] if U in sizes else 1 for U in u_names}
        if seed is not None:
            self.rand_state = np.random.RandomState(seed=seed)
        else:
            self.rand_state = np.random.RandomState()

        self.func = T.nn.ModuleDict({
            str(u): default_module(
                {},
                {u: self.sizes[u]},
                self.sizes[u],
                h_layers=hyperparams.get('h-layers', 2),
                h_size=hyperparams.get('h-size', 128),
                use_layer_norm=hyperparams.get('layer-norm', False),
                use_sigmoid=False
            )
            for u in self.sizes})

    def sample(self, n=1, device=None):
        if device is None:
            device = self.device_param.device

        u_vals = dict()
        for U in self.sizes:
            noise = T.randn((n, self.sizes[U])).float().to(device)
            u_vals[U] = self.func[str(U)]({}, {U: noise})

        return u_vals


class BernoulliDistribution(Distribution):
    def __init__(self, u_names, sizes, p, seed=None):
        assert set(sizes.keys()).issubset(set(u_names))

        super().__init__(list(u_names))
        self.sizes = {U: sizes[U] if U in sizes else 1 for U in u_names}
        self.p = p
        if seed is not None:
            self.rand_state = np.random.RandomState(seed=seed)
        else:
            self.rand_state = np.random.RandomState()

    def sample(self, n=1, device=None):
        if device is None:
            device = self.device_param.device

        u_vals = dict()
        for U in self.sizes:
            u_vals[U] = T.from_numpy(self.rand_state.binomial(1, self.p, size=(n, self.sizes[U]))).long().to(device)

        return u_vals