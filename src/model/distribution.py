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