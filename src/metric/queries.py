import torch as T

from src.metric.counterfactual import CTFTerm

def expand_do(val, n):
    """Kevin"""
    if T.is_tensor(val):
        return T.tile(val, (n, 1))
    else:
        return T.unsqueeze(T.ones(n) * val, 1)
    
def expanded_dos(vals,n):
    do_dict = {}
    for k in vals:
        do_dict[k] = expand_do(vals[k],n)
    return do_dict

def check_equal(input, val):
    """Kevin"""
    if T.is_tensor(val):
        return T.all(T.eq(input, T.tile(val, (input.shape[0], 1))), dim=1).bool()
    else:
        return T.squeeze(input == val)
    
def tensor_prob_dist(t):
    """Returns the probability distribution of a 1D tensor with discrete values."""
    unique_vals, counts = T.unique(t, return_counts=True)
    probs = counts.float() / counts.sum()
    return unique_vals, probs

def get_u_n(model, u=None, n=10000):
    if u is None:
        U = model.pu.sample(n=n)
        return U, n
    else:
        n_new = len(u[next(iter(u))])
        return u, n_new

def get_conditioned_u(model, u=None, do={}, conditions=None, n=10000):
    return _get_conditioned_u(model, u=u, do=do, conditions=conditions, n=n)

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

def sample_ctf(model, term: CTFTerm, conditions={}, n=10000, u=None, evaluating=True):
    """
    Samples the given model based on the given counterfactual term. 
    """
    U, n = get_conditioned_u(model, u, conditions=conditions, n=n)

    expanded_do_terms = dict()
    for k in term.do_vals:
        if k == "nested":
            nested_term = term.do_vals[k]
            ctf = sample_ctf(model=model, term=nested_term, n=n, u=U, evaluating=False)
            for var in nested_term.vars:
                expanded_do_terms.update({var: ctf[var]})
        else:
            expanded_do_terms[k] = expand_do(term.do_vals[k],n)

    return model(n=None, u=U, do=expanded_do_terms, select=term.vars, evaluating=evaluating)

def _probability(model, var, val, evidence={}, intervention={}, neq_evidence={}, u=None):
    """ This will calculate the probability that a variable equals a certain value given the dictionary of evidence, and the dictionary of interventions. Both dictionaries are optional.
    - `probability('Y',1,{'X':0},{'Z':1})` will return $P(Y=1 | X=0, do(Z=1))$
    - `probability('Y',1,intervention={'Z':1})` will return $P(Y=1 | do(Z=1))$
    """
    U, _ = _get_conditioned_u(model, u, conditions=evidence, opposite_conditions=neq_evidence)
    sampleY = sample_ctf(model, CTFTerm({var}, do_vals=intervention), u=U)[var]
    n = sampleY.numel()
    return 0 if n==0 else (sampleY == val).sum().item() / n 

def probability(model, var, val, evidence={}, intervention={}):
    """ This will calculate the probability that a variable equals a certain value given the dictionary of evidence, and the dictionary of interventions. Both dictionaries are optional.
    - `probability('Y',1,{'X':0},{'Z':1})` will return $P(Y=1 | X=0, do(Z=1))$
    - `probability('Y',1,intervention={'Z':1})` will return $P(Y=1 | do(Z=1))$
    """
    return _probability(model, var, val, evidence=evidence, intervention=intervention) 

def get_evidence(var, eq, neq, evidence, neq_evidence):
    if eq is not None: 
        evidence[var] = eq
    else: 
        assert neq is not None
        neq_evidence[var] = neq
    return evidence, neq_evidence

def total_variation(model, var, val, attr, aval1, aval0=None):
    U = model.pu.sample(10000)
    x1 = {attr: aval1}
    pY_condx1 = _probability(model, var, val, evidence=x1, u=U)
    
    ev, neq_ev = get_evidence(attr, eq=aval0, neq=aval1, evidence={}, neq_evidence={})
    pY_condx0 = _probability(model, var, val, evidence=ev, neq_evidence=neq_ev, u=U)

    return pY_condx1 - pY_condx0

def total_effect(model, var, val, attr, aval1, aval0, evidence={}):
    U = model.pu.sample(10000)
    x1 = {attr: aval1}
    pY_dox1 = _probability(model, var, val, intervention=x1, evidence=evidence, u=U)

    x0 = {attr: aval0}
    pY_dox0 = _probability(model, var, val, intervention=x0, evidence=evidence, u=U)

    return pY_dox1 - pY_dox0

def ett(model, var, val, treatment_var, treatment_vals):
    assert 'whatif' in treatment_vals
    dox = {treatment_var: treatment_vals['whatif']}
    ev, neq_ev = get_evidence(treatment_var, treatment_vals.get('actual',None), treatment_vals.get('whatif',None), evidence={}, neq_evidence={})
    return _probability(model, var, val, intervention=dox, evidence=ev, neq_evidence=neq_ev)
    
def pnps(model, var, vals, treatment_var, treatment_vals, data=None):
    """
    Return PNPSx,x' = P(Y_x = y | Y=y', X=x')
    """
    U = model.pu.sample(10000)
    yval = vals['whatif']
    xval = treatment_vals['whatif']

    ev, neq_ev = get_evidence(var, vals.get('actual',None), yval, evidence={}, neq_evidence={})
    get_evidence(treatment_var, treatment_vals.get('actual',None), xval, ev, neq_ev)

    return _probability(model, var, yval, intervention={treatment_var:xval}, evidence=ev, neq_evidence=neq_ev, u=U)
