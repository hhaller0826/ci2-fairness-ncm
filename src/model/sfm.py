import torch as T
import torch.nn as nn
import numpy as np

from src.model.scm import SCM
from src.model.ncm.mlp import *
from src.data import ProcessedData

from src.model.ncm.mlp import TwoLayerArchitecture

class SFM(SCM):
    def __init__(self, assignments, f, pu, scale, v_size={}, default_v_size=1, og_projection=None):
        self.assignments = assignments
        
        super().__init__(v=list(assignments.keys()), f=f, pu=pu)

        self.X = 'X'
        self.Y = 'Y'
        self.Z = 'Z' if 'Z' in self.v else None 
        self.W = 'W' if 'W' in self.v else None 
        self.Yhat = None

        self.scale = scale
        self.og_projection = og_projection
        self.v_size = {k: v_size.get(k, default_v_size) for k in self.v}
    
    def convert_evaluation(self, samples):
        ret = {}
        for k in samples:
            x = samples[k]
            scale_var = k if k != self.Yhat else self.Y
            ret[k] = T.tensor([[self.scale[scale_var][i](x[j][i]).item() for i in range(len(x[0]))] for j in range(len(x))])
        return ret
    
    def print_projection(self):
        print(f'Protected Attribute: {self.assignments['X']}')
        print(f'Confounders:         {self.assignments.get('Z', None)}')
        print(f'Mediators:           {self.assignments.get('W', None)}')
        print(f'Outcome:             {self.assignments['Y']}')

    def add_predictor(self, prediction_model):
        self.v.remove(self.Yhat)
        self.Yhat = 'fair_predictions'

        if type(prediction_model)==TwoLayerArchitecture:
            def fyhat(v, u=None, model=None):
                x_data = v.get('X', T.tensor([]))
                z_data = v.get('Z', T.tensor([]))
                w_data = v.get('W', T.tensor([]))
                fts_eval_t = T.hstack((x_data,z_data,w_data))
                return model(fts_eval_t)
            
            self.v.append(self.Yhat)
            self.f[self.Yhat] = (lambda v, u, model=prediction_model: fyhat(v, u, model))

        else: 
            self.v.append(self.Yhat)
            self.f[self.Yhat] = (lambda v, u, model=prediction_model: prediction_model(v, u))

    def predict(self, data=None, n=100):
        if self.Yhat is None:
            raise ValueError('Predictor has not been initialized on this model.')
        if data is not None:
            if 'X' not in data:
                new_data = {
                    'X':data[self.assignments['X']],
                    'Z':data[self.assignments.get('Z',[])],
                    'W':data[self.assignments.get('W',[])]
                }
            else: new_data = data
            return self.f[self.Yhat](new_data)
        return self.sample(n=n)[self.Yhat]