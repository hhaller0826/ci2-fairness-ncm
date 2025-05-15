import torch as T
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    # KEVIN
    def __init__(self, pa_size, u_size, o_size, h_size=128, h_layers=2, use_sigmoid=True, use_layer_norm=True):
        super().__init__()
        self.pa = sorted(pa_size)
        self.set_pa = set(self.pa)
        self.u = sorted(u_size)
        self.pa_size = pa_size
        self.u_size = u_size
        self.o_size = o_size
        self.h_size = h_size

        self.i_size = sum(self.pa_size[k] for k in self.pa_size) + sum(self.u_size[k] for k in self.u_size)

        layers = [nn.Linear(self.i_size, self.h_size)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(self.h_size))
        layers.append(nn.ReLU())
        for l in range(h_layers - 1):
            layers.append(nn.Linear(self.h_size, self.h_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(self.h_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.h_size, self.o_size))
        if use_sigmoid:
            layers.append(nn.Sigmoid())

        self.nn = nn.Sequential(*layers)

        self.device_param = nn.Parameter(T.empty(0))

        self.nn.apply(self.init_weights)
        
    def init_weights(self, m):
        if type(m) == nn.Linear:
            T.nn.init.xavier_normal_(m.weight,
                                     gain=T.nn.init.calculate_gain('relu'))

    def forward(self, pa, u, inp_pa=None, include_inp=False):
        if len(u.keys()) == 0:
            inp = T.cat([pa[k] for k in self.pa], dim=1) if inp_pa is None else inp_pa
        elif len(pa.keys()) == 0 or len(set(pa.keys()).intersection(self.set_pa)) == 0:
            inp = T.cat([u[k] for k in self.u], dim=1)
        else:
            inp_u = T.cat([u[k] for k in self.u], dim=1)
            inp_pa = T.cat([pa[k] for k in self.pa], dim=1) if inp_pa is None else inp_pa
            inp = T.cat((inp_pa, inp_u), dim=1)

        if include_inp:
            return self.nn(inp), inp

        return self.nn(inp)

if __name__ == '__main__':
    s = MLP(dict(v1=2, v2=1), dict(u1=1, u2=2), 3)
    print(s)
    pa = {
        'v1': T.tensor([[1, 2], [3, 4.]]),
        'v2': T.tensor([[5], [6.]])
    }
    u = {
        'u1': T.tensor([[7.], [8]]),
        'u2': T.tensor([[9, 10], [11, 12.]])
    }
    print(s(pa, u))


class MultiMLPConcat(nn.Module):
    def __init__(self, mlps, input_sizes):
        """
        mlps: list of MLPs
        input_sizes: list of integers specifying the input size of each MLP
        """
        super().__init__()
        assert len(mlps) == len(input_sizes), "Each MLP must have a corresponding input size"
        self.mlps = nn.ModuleList(mlps)
        self.input_sizes = input_sizes
        self.split_indices = self._compute_split_indices(input_sizes)

    def _compute_split_indices(self, sizes):
        indices = [0]
        for s in sizes:
            indices.append(indices[-1] + s)
        return indices

    def forward(self, x):
        outputs = []
        for i, mlp in enumerate(self.mlps):
            start, end = self.split_indices[i], self.split_indices[i+1]
            xi = x[:, start:end]
            yi = mlp(xi)
            outputs.append(yi)
        return T.cat(outputs, dim=1)

class VerticalStackMLP(nn.Module):
    def __init__(self, pa, mlps, sorted_output_vars, v_size={}, default_v_size=1, keep_separated=False):
        """
        pa: dict mapping variable name -> list of its parent variable names
        mlps: dict mapping variable name -> MLP (nn.Module)
        """
        super().__init__()
        self.pa = pa
        self.mlps = nn.ModuleDict(mlps)
        self.output_vars = sorted_output_vars
        self.keep_separated = keep_separated

        self.o_size = sum(v_size.get(out, default_v_size) for out in sorted_output_vars)

    def forward(self, pa, u):
        """
        inputs: dict mapping input variable name -> tensor
        """
        computed = {}

        def compute(var):
            if var in computed:
                return computed[var]
            inp_pa = {p: compute(p) for p in self.pa[var]}
            out = self.mlps[var](inp_pa, u)
            computed[var] = out
            return out
        
        if self.keep_separated: 
            return {out_var: compute(out_var) for out_var in self.output_vars}

        return T.cat([compute(out_var) for out_var in self.output_vars], dim=1)

class HorizontalStackMLP(nn.Module):
    def __init__(self, mlps, sorted_output_vars, unproject_map, pa_mlps=None, keep_separated=False):
        super().__init__()
        self.mlps = nn.ModuleDict(mlps)
        self.output_vars = sorted_output_vars
        self.unproject_map = unproject_map
        self.pa_mlps = pa_mlps
        self.keep_separated = keep_separated

    def unproject_pa(self, input):
        values = {}
        for var in input:
            if var in self.unproject_map:
                v_indices = self.unproject_map[var]
                values.update({v: input[var][:, v_indices[v][0]:v_indices[v][1]] for v in v_indices})
            else:
                values[var] = input[var]
        return values

    def forward(self, pa, u):
        inp_sfm_pa = self.unproject_pa(pa)
        inp_ext_pa = self.pa_mlps({}, u) if self.pa_mlps is not None else {}
        v = {**inp_sfm_pa, **inp_ext_pa}

        def compute(var):
            if var in v:
                return v[var]
            out = self.mlps[var](v, u)
            v[var] = out
            return out
        
        if self.keep_separated: 
            return {out_var: compute(out_var) for out_var in self.output_vars}

        return T.cat([compute(out_var) for out_var in self.output_vars], dim=1)


class TwoLayerArchitecture(nn.Module):
    # DRAGO
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerArchitecture, self).__init__()
        # Define the architecture here.
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Define the forward pass here.
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output_layer(x)
        return x