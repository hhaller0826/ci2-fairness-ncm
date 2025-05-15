import warnings 
import pandas as pd
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.model.scm import SCM
from src.model.ncm.feedforward_ncm import FF_NCM
from src.model.ncm.mlp import *
from src.model.sfm import SFM

from src.model.distribution import *
from src.graph.causal_graph import CausalGraph
from src.data import ProcessedData

warnings.simplefilter(action='ignore', category=FutureWarning)

def process_data_assignments(df, assignments, graph: CausalGraph, categorical_vars=[], discrete_vars=[], test_size=0.1):
    check_assignments(assignments=assignments, data=df, graph=graph)
    return ProcessedData(df, assignments, categorical_vars, discrete_vars, test_size)

def check_assignments(data, assignments: dict, graph: CausalGraph):
    """
    Check for assigning data columns to graph nodes.
    """
    # Check that all nodes are being assigned
    assert assignments.keys() <= graph.set_v, f'Node {assignments.keys()-graph.set_v} not in graph'
    
    assigned_features = []
    for features in assignments.values():
        assert features is not None and len(features) > 0, f'All nodes must have an assignment'
        assigned_features.extend(features)
    feature_set = set(assigned_features)
    # check for duplicate features:
    if len(feature_set) < len(assigned_features):
        seen = set()
        duplicates = {x for x in assigned_features if x in seen or seen.add(x)}
        raise ValueError('Feature was assigned to a variable more than once: {}'.format(duplicates))

    # check for unknown features:
    cols = set(data.columns) # assuming this is a DataFrame object rn 
    unknown_features = feature_set - cols
    if len(unknown_features) > 0:
        raise ValueError('Unknown feature assignment: {}'.format(unknown_features))

    # check for missing features (this is OK):
    unassigned_features = cols - feature_set
    if len(unassigned_features) > 0:
        warnings.warn('The following features were not assigned to any variable: {}'.format(unassigned_features), UserWarning)
        print("It is okay to exclude features from the model but they will not be used in the causal analysis.")

    return feature_set


def get_ncm(graph, assignments={}, hyperparams=None, scale={}):
    if hyperparams is None:
        hyperparams = dict()
    dist_type = hyperparams.get('distribution', 'uniform')

    model_choice = hyperparams.get('model_choice','ff')
    v_size = {k:len(assignments[k]) for k in assignments}
    if model_choice == 'ff':
        return FF_NCM(graph, hyperparams=hyperparams, v_size=v_size, scale=scale)

def process_projection(projection):
    X = projection['X']
    Z = set(projection.get('Z',{}))
    W = set(projection.get('W',{}))
    Y = projection['Y']
    return X, Z, W, Y

def check_projection(projection, cg: CausalGraph):
    """
    Check when projecting graph nodes to SFM
    """
    # check attributes in projection 
    assert 'X' in projection, f'Must specify a protected attribute X.'
    assert 'Y' in projection, f'Must specify an outcome Y.'
    if 'Z' not in projection: warnings.warn(f'No confounders assigned.')
    if 'W' not in projection: warnings.warn(f'No mediators assigned.')

    random_keys = projection.keys()-{'X','Y','Z','W'}
    if len(random_keys) > 0: warnings.warn(f'{random_keys} will not be used as variables in the SFM. The following will remain unassigned: {[projection[k] for k in random_keys][0]}', UserWarning)

    X, Z, W, Y = process_projection(projection)
    projected_vars = {X, *Z, *W, Y}

    # Check that every variable along a path within the SFM is assigned (ex.: W1>V>W2)
    unassgned = cg.set_v - projected_vars
    for unassgn in unassgned:
        ancestor_intersection = cg.ancestors({unassgn}).intersection(projected_vars)
        if len(ancestor_intersection)>0:
            grandkid_intersection = cg.grandkids({unassgn}).intersection(projected_vars)
            if len(grandkid_intersection)>0:
                raise ValueError(f'[Path {ancestor_intersection}->{unassgn}->{grandkid_intersection}]: Please explicitly assign {unassgn} to one of the SFM variables.')

    # check assignments
    if len(projected_vars - cg.set_v)>0: raise ValueError(f'Unknown variable(s): {projected_vars.difference(cg.set_v)}')

    # check duplicates
    if len(projected_vars)<len([X, *Z, *W, Y]):
        seen = set()
        duplicates = {x for x in [X, *Z, *W, Y] if x in seen or seen.add(x)}
        raise ValueError('Duplicate variable assignment: {}'.format(duplicates))

    err = ''
    # check bidirected arrows
    xz_cc = set([*(cg.v2cc[xz] for xz in {X,*Z})])
    w_cc = set([*(cg.v2cc[w] for w in W)])
    y_cc = cg.v2cc[Y]
    if len(xz_cc.intersection(w_cc))>0: err += f'Cannot have bidirected arrow path between X or Z and W'
    if len(xz_cc.intersection(y_cc))>0: err += f'Cannot have bidirected arrow path between X or Z and Y'
    if len(w_cc.intersection(y_cc))>0: err += f'Cannot have bidirected arrow path between W and Y'
    
    # check directed arrows
    x_ancestors = cg.ancestors({X})
    z_ancestors = cg.ancestors(Z)
    w_ancestors = cg.ancestors(W)
    
    if Y in {*x_ancestors, *z_ancestors, *w_ancestors}: err += f'Cannot have path from Y to X, Z, or W in Standard Fairness Model.\n'
    if len(W.intersection({*x_ancestors, *z_ancestors}))>0: err += f'Cannot have path from W to X or Z in Standard Fairness Model.\n'
    if len(Z.intersection(x_ancestors))>0: err += f'Cannot have path from Z to X in Standard Fairness Model.\n'
    if X in z_ancestors: err += f'Cannot have path from X to Z in Standard Fairness Model.\n'

    if len(err) > 0: raise ValueError(err)
    return X, Z, W, Y

def index_decomposition(var_set, v_sizes):
    indices = {}
    i=0
    for v in var_set:
        indices[v] = (i,i+v_sizes[v])
    return indices

def project_to_sfm(model: SCM, projection, data: ProcessedData, cg: CausalGraph, assignments=None):
    X, Z, W, Y = check_projection(projection, cg)
    projected_vars = {X, *Z, *W, Y}
    var_assignments = getattr(data, 'assignments', assignments)
    if var_assignments is None: raise ValueError('Original graph node assignments required for SFM projection.')

    v_size = {k:len(var_assignments[k]) for k in var_assignments}
    f = {}

    # assume f's are all MLPs for right now
    x_ancestors = cg.ancestors({X})
    f['X'] = VerticalStackMLP({an:cg.pa[an] for an in x_ancestors}, {an:model.f[an] for an in x_ancestors}, {X}, v_size) if len(x_ancestors)>1 else model.f[X]

    unproject_map = {'X':{X:(0,v_size[X])}}
    if len(Z)>0:
        z_ancestors = cg.ancestors(Z)
        f['Z'] = VerticalStackMLP({an:cg.pa[an] for an in z_ancestors}, {an:model.f[an] for an in z_ancestors}, cg.convert_set_to_sorted(Z), v_size)
        unproject_map['Z'] = index_decomposition(Z, v_size)
    
    if len(W)>0:
        w_parents = {v for w in W for v in cg.pa[w]} - projected_vars
        pa_w_ancestors = cg.ancestors(w_parents)
        fpa_w = VerticalStackMLP({an:cg.pa[an] for an in pa_w_ancestors}, {an:model.f[an] for an in pa_w_ancestors}, cg.convert_set_to_sorted(w_parents), v_size, keep_separated=True) if len(w_parents)>0 else None
        f['W'] = HorizontalStackMLP({w:model.f[w] for w in W}, W, unproject_map=unproject_map, pa_mlps=fpa_w)
        unproject_map['W'] = index_decomposition(W, v_size)

    y_parents = {*cg.pa[Y]} - projected_vars
    pa_y_ancestors = cg.ancestors(y_parents)
    fpa_y = VerticalStackMLP({an:cg.pa[an] for an in pa_y_ancestors}, {an:model.f[an] for an in pa_y_ancestors}, cg.convert_set_to_sorted(y_parents), v_size, keep_separated=True) if len(y_parents)>0 else None
    f['Y'] = HorizontalStackMLP({Y:model.f[Y]}, {Y}, unproject_map=unproject_map, pa_mlps=fpa_y)


    data_assignments = {k: list(itertools.chain.from_iterable([var_assignments[v] for v in projection[k]])) for k in projection}
    scale = data.get_assigned_scale(data_assignments)

    # TODO: Improve SFM code
    return SFM(data_assignments, f=f, pu=model.pu, scale=scale, og_projection=projection, v_size={k: sum(v_size[v] for v in projection[k]) for k in projection})