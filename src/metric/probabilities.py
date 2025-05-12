import torch as T

from src.metric.counterfactual import CTFTerm
from src.metric.queries import get_u_n, sample_ctf, tensor_prob_dist

def _get_conditioned_u(model, u=None, do={}, conditions={}, opposite_conditions={}, n=10000):
    """
    Gets U conditioned on v != opposite_conditions[v] for all v in opposite_conditions
    """
    u, n_new = get_u_n(model, u, n)
    sample = model(u=u, evaluating=True)
    
    indices_to_keep = set(range(n_new))
    for c in conditions:
        itk = T.where(sample[c]==conditions[c])[0].tolist()
        indices_to_keep = indices_to_keep.intersection(set(itk))

    for c in opposite_conditions:
        itk = T.where(sample[c]==opposite_conditions[c])[0].tolist()
        indices_to_keep = indices_to_keep.difference(set(itk))

    return {k:u[k][list(indices_to_keep)] for k in u}, len(indices_to_keep)

class ReusableProbability:
    def __init__(self, model, outcome, attr=None, aval1=None, aval0=None, u=None, n=10000):
        self.model = model
        self.U, self.n = _get_conditioned_u(model, u=u, n=n)

        self.Y = outcome
        self.X = attr
        self._x0_val = aval0
        self._x1_val = aval1
        self._x0, self._x1 = None, None

    def _y(self, do_vals={}):
        return CTFTerm(self.Y, do_vals)

    def _reset_u(self, n=None):
        if n is not None: self.n=n
        self.model.pu.sample(n=self.n)

    @property
    def x0_val(self): return self._x0_val
    @x0_val.setter
    def x0_val(self, new_x0):
        self._x0_val = new_x0
        self._x0 = {self.X: new_x0}

    @property
    def x1_val(self): return self._x1_val
    @x1_val.setter
    def x1_val(self, new_x1):
        self._x1_val = new_x1
        self._x1 = {self.X: new_x1}

    @property
    def x0(self):
        assert not (self.x0_val is None and self.x1_val is None)
        if self.x0_val is None: return {'neq':self.x1}
        if self._x0 is None:
            self._x0 = {self.X:self.x0_val}
        return self._x0
    
    @property
    def x1(self):
        assert not (self.x0_val is None and self.x1_val is None)
        if self.x1_val is None: return {'neq':self.x0}
        if self._x1 is None:
            self._x1 = {self.X:self.x1_val}
        return self._x1

    def _probability(self, val=None, conditions={}, intervention={}):
        """Calculates the probability"""
        eq = {k: conditions[k] for k in set(conditions.keys()).difference({'neq'})}
        temp_u, _ = _get_conditioned_u(self.model, self.U, conditions=eq, opposite_conditions=conditions.get('neq',{}))
        sampleY = sample_ctf(self.model, self._y(do_vals=intervention), u=temp_u)[self.Y]

        if val is None: return tensor_prob_dist(sampleY)
        temp_n = sampleY.numel()
        return 0 if temp_n==0 else (sampleY==val).sum().item() / temp_n
    
    def probability(self, val, evidence={}, intervention={}, print_str=True):
        """Prints & calculates the probability"""
        prob = self._probability(val, evidence, intervention)
        cond = ','.join([f'{k}={evidence[k]}' for k in evidence]+[f'do({k}={intervention[k]})' for k in intervention])
        if print_str: print(f'P({self.Y}={val}{' | '+cond if cond else ''}) = {prob:.4f}')
        return prob
    
    def total_variation(self, val, print_str=True):
        py_condx1 = self._probability(val, conditions=self.x1)
        py_condx0 = self._probability(val, conditions=self.x0)
        tv = py_condx1 - py_condx0
        if print_str: print(f'TV({self.Y}={val}) = {py_condx1:.4f} - {py_condx0:.4f} = {tv:.4f}')
        return tv
    
    def total_effect(self, val, x1=None, x0=None, evidence={}, print_str=True):
        dox1 = self.x1 if x1 is None else {self.X:x1} 
        dox0 = self.x0 if x0 is None else {self.X:x0}
        if 'neq' in dox1 or 'neq' in dox0: raise ValueError(f'Interventions must be explicit.')
        
        py_dox1 = self._probability(val, intervention=dox1, conditions=evidence)
        py_dox0 = self._probability(val, intervention=dox0, conditions=evidence)
        te = py_dox1 - py_dox0
        if print_str:
            z = ','.join([f'{k}={evidence[k]}' for k in evidence])
            print(f'TE({self.Y}={val}{' | '+z if z else ''}) = {py_dox1:.4f} - {py_dox0:.4f} = {te:.4f}')
        return te
    
    def ett(self, val, whatif_treatment=None, actual_treatment=None, print_str=True):
        """
        whatif_treatment (Default=self.x0)
        """
        whatif_treatment = self.x0_val if whatif_treatment is None else whatif_treatment
        if whatif_treatment is None: raise ValueError(f'Interventions must be explicit. Specify a whatif value or set prob.x0_val')
        
        treatment = {'neq':{self.X:whatif_treatment}} if actual_treatment is None else {self.X:actual_treatment}
        ett = self._probability(val, intervention={self.X:whatif_treatment}, conditions=treatment)

        if print_str:
            interv_str = f'actually {actual_treatment}' if actual_treatment else f'not actually {whatif_treatment}'
            print(f'{ett:.4f}: probability that {self.Y}={val} if we had intervened to make {self.X}={whatif_treatment}, given that {self.X} was {interv_str}.')

        return ett
    
    def pnps(self, whatif_outcome, actual_outcome=None, whatif_treatment=None, actual_treatment=None, print_str=True):
        """
        whatif_treatment (Default=self.x0)
        """
        whatif_treatment = self.x0_val if whatif_treatment is None else whatif_treatment
        if whatif_treatment is None: raise ValueError(f'Interventions must be explicit. Specify a whatif_treatment value or set prob.x0_val')

        conditions = {'neq': {self.X:whatif_treatment}} if actual_treatment is None else {self.X:actual_treatment}
        conditions.update({'neq': {**conditions['neq'], self.Y:whatif_treatment}} if actual_outcome is None else {self.Y: actual_outcome})

        pnps = self._probability(whatif_outcome, intervention={self.X:whatif_treatment}, conditions=conditions)

        if print_str:
            interv_strx = f'{self.X} was actually {actual_treatment}' if actual_treatment else f'{self.X} was not actually {whatif_treatment}'
            interv_stry = f'{self.Y} was actually {actual_outcome}' if actual_outcome else f'{self.Y} was not actually {whatif_outcome}'
            print(f'{pnps:.4f}: probability that {self.Y}={whatif_outcome} if we had intervened to make {self.X}={whatif_treatment}, given that {interv_stry} and {interv_strx}.')

        return pnps

