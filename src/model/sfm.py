from collections.abc import Iterable
import torch as T
import itertools

from src.model.scm import SCM

class SFM(SCM):
    def __init__(self, model: SCM, projection, data=None):
        self.v_size = {k:len(data.assignments[k]) for k in data.assignments} if getattr(data, 'assignments', None) is not None else getattr(model, 'v_size')
        # TODO: check valid projection
        # WILL need graph to do that
        # TODO: translate convert_evaluation function
        V, f = project_to_sfm(model, projection, self.v_size)
        super().__init__(v=V, f=f, pu=model.pu)

        self.X = 'X'
        self.Y = 'Y'
        self.Z = 'Z' if 'Z' in self.v else None 
        self.W = 'W' if 'W' in self.v else None 

        self.og_model = model
        self.assignments = {k: list(itertools.chain.from_iterable([data.assignments[v] for v in projection[k]])) for k in projection}
        self.projection = projection

        self.scale = data.get_assigned_scale(self.assignments)
    
    def convert_evaluation(self, samples):
        # return super().convert_evaluation(samples)
        # return self.og_model.convert_evaluation(samples)
        ret = {}
        for k in samples:
            x = samples[k]
            ret[k] = T.tensor([[self.scale[k][i](x[j][i]).item() for i in range(len(x[0]))] for j in range(len(x))])
        return ret
    
    def print_projection(self):
        print(f'Protected Attribute: {self.assignments['X']}')
        print(f'Confounders:         {self.assignments.get('Z', None)}')
        print(f'Mediators:           {self.assignments.get('W', None)}')
        print(f'Outcome:             {self.assignments['Y']}')

def process_projection(projection):
    X = projection['X']
    Z = [*projection.get('Z',[])]
    W = [*projection.get('W',[])]
    Y = projection['Y']
    return X, Z, W, Y

# Confirm that this is the projection you intended:
def project_to_sfm(model: SCM, projection, v_sizes):
    X, Z, W, Y = process_projection(projection)

    V = ['X']
    f = {
        'X': model.f[X]
    }

    if len(Z)>0:
        def fz(v, u, X_proj, Z_proj, model):
            temp_v = {}
            temp_v[X_proj] = v['X']
            for z in Z_proj:
                temp_v[z] = model.f[z](temp_v,u)
            return T.cat([temp_v[z] for z in Z_proj], dim=1)
        
        V.append('Z')
        f['Z'] = (lambda v, u, X=X, Z=Z, model=model: fz(v, u, X, Z, model))

    if len(W)>0:
        def fw(v, u, X_proj, Z_proj, W_proj, model, v_sizes):
            temp_v = {}
            temp_v[X_proj] = v['X']
            i=0
            for z in Z_proj:
                z_size = v_sizes[z]
                temp_v[z] = v['Z'][:, i:i+z_size]
                i+=z_size
            
            for w in W_proj:
                temp_v[w] = model.f[w](temp_v,u)
            return T.cat([temp_v[w] for w in W_proj], dim=1)
        
        V.append('W')
        f['W'] = (lambda v, u, X=X, Z=Z, W=W, model=model, v_sizes=v_sizes: fw(v, u, X, Z, W, model, v_sizes))

    def fy(v, u, X_proj, Z_proj, W_proj, Y_proj, model, v_sizes):
        temp_v = {}
        temp_v[X_proj] = v['X']
        i=0
        for z in Z_proj:
            z_size = v_sizes[z]
            temp_v[z] = v['Z'][:, i:i+z_size]
            i+=z_size
        i=0
        for w in W_proj:
            w_size = v_sizes[w]
            temp_v[w] = v['W'][:, i:i+w_size]
            i+=w_size
        return model.f[Y_proj](temp_v,u)
    
    V.append('Y')
    f['Y'] = (lambda v, u, X=X, Z=Z, W=W, Y=Y, model=model, v_sizes=v_sizes: fy(v, u, X, Z, W, Y, model, v_sizes))

    return V, f
