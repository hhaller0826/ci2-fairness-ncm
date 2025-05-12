import warnings 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from src.model.ncm.feedforward_ncm import FF_NCM
from src.model.distribution import *
from src.graph.causal_graph import CausalGraph
from src.data import ProcessedData

warnings.simplefilter(action='ignore', category=FutureWarning)

def process_data_assignments(df, assignments, graph: CausalGraph, categorical_vars=[], test_size=0.1):
    check_assignments(assignments=assignments, data=df, graph=graph)
    return ProcessedData(df, assignments, categorical_vars, test_size)

# def process_data_assignments(df, assignments, graph: CausalGraph, categorical_vars=[], test_size=0.1):
#     columns = check_assignments(assignments=assignments, data=df, graph=graph)
    
#     abbr_df = pd.DataFrame(df, columns=list(columns))
#     print_df = pd.DataFrame()

#     dat_train, dat_test = train_test_split(abbr_df, test_size=test_size, random_state=42)
#     scale = {}

#     for feat in columns:
#         print_df[feat+'_orig'] = dat_train[feat]
#         encoder = LabelEncoder()
#         if feat in categorical_vars:
#             dat_train[feat] = encoder.fit_transform(dat_train[feat])
#             dat_test[feat] = encoder.fit_transform(dat_test[feat])

#             maxval = abbr_df[feat].nunique()-1
#             minval = 0
#         else:
#             maxval = abbr_df[feat].max()
#             minval = abbr_df[feat].min()
#         scale[feat] = (lambda x, maxval=maxval, minval=minval: (x*(maxval-minval)) + minval)
#         # easiest to use NN with a sigmoid so we need to normalize the values between 0 and 1
#         # TODO: currently just hoping the real max & min values are in the dataset. if this is grades but everyone got B's and C's, then my algo will never predict A or D
#         dat_train[feat] = dat_train[feat].apply(lambda x: (x-minval)/(maxval-minval))
#         dat_test[feat] = dat_test[feat].apply(lambda x: (x-minval)/(maxval-minval))

#         print_df[feat] = dat_train[feat]

#     assigned_scale = {}
#     for v in assignments:
#         assigned_scale[v] = [scale[assignments[v][i]] for i in range(len(assignments[v]))]

#     train_dataloader = NCMDataset(dat_train, assignments).get_dataloader(batch_size=32)
#     test_dataloader = NCMDataset(dat_test, assignments).get_dataloader(batch_size=32)

#     return {'train': train_dataloader, 'test': test_dataloader, 'train-df': dat_train, 'test-df': dat_test, 'scale': assigned_scale, 'print': print_df, 'assignments': assignments}


def check_assignments(data, assignments: dict, graph: CausalGraph):
    # Check that all nodes are being assigned
    assert assignments.keys() <= graph.set_v, f'Node {assignments.keys()-graph.set_v} not in graph'
    assert assignments.keys() | graph.assignments.keys() == graph.set_v, f'All nodes must have an assignment'

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
    cols = set(data.columns) # assuming this is a DataFrame object rn but prob wont be 
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
    

    
