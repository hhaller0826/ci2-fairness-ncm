
import networkx as nx
import matplotlib.pyplot as plt
import warnings
from src.graph.utils import get_positions,plot_causal_graph
from src.causalaibook.fusion import * 
from src.causalaibook.graph.classes.graph import Graph
import numpy as np
from src.causalaibook.utils import plot_causal_diagram

def graph_string(cg):
    lines = ["<NODES>\n"]
    for V in cg.v:
        lines.append("{}\n".format(V))
    lines.append("\n")
    lines.append("<EDGES>\n")
    for V1, V2 in cg.de:
        lines.append("{} -> {}\n".format(V1, V2))
    for V1, V2 in cg.be:
        lines.append("{} <-> {}\n".format(V1, V2))
    return lines

def process_data(d): 
    return {'age': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], 
            'gender': [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0], 
            'race': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
            'degree': [1, 2, 3, 1, 2, 3, 3, 2, 1, 3, 2, 1],
            'num_awards': [0, 10, 4, 6, 2, 1, 0, 0, 0, 3, 2, 1],
            'salary': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
            'skincolor': [1, 1, 0, None, 1, 0],
            'job_title': [0, 1, 0, 1, 0, 1, 1, 0, 0, 1],}

class Model():
    def __init__(self, data, distribution, graph, assignments):
        self.check_assignments(data, assignments)
        self.data = data
        self.distribution = distribution
        self.cg = graph
        self.assignments = assignments

        print(f"Model assignments: {assignments}")
        print()
        
        pass

    def check_assignments(self, data, assignments):
        # pretending "data" is a dictionary
        data_features = list(data.keys())

        assigned_features = []
        for features in assignments.values():
            assigned_features.extend(features)

        # check for duplicate features:
        if len(set(assigned_features)) < len(assigned_features):
            seen = set()
            duplicates = {x for x in assigned_features if x in seen or seen.add(x)}
            raise ValueError('Feature was assigned to a variable more than once: {}'.format(duplicates))

        # check for unknown features:
        unknown_features = assigned_features - data.keys()
        if len(unknown_features) > 0:
            raise ValueError('Unknown feature assignment: {}'.format(unknown_features))

        # check for missing features (this is OK):
        unassigned_features = data.keys() - assigned_features
        if len(unassigned_features) > 0:
            warnings.warn('The following features were not assigned to any variable: {}'.format(unassigned_features), UserWarning)
            print("It is okay to exclude features from the model but they will not be used in the causal analysis.")
        
def run_training(graph, model, hyperparameters):
    print(f'training ran with: \n Graph: {graph} \n Model: {model} \n Params: {hyperparameters}')

def sfm_graph(x_label='X', z_label='Z', w_label='W', y_label='Y'):
    W,X,Y,Z = 'W','X','Y','Z'
    nodes = [
        {'name': Y, 'label':y_label},
        {'name': Z, 'label':z_label},
        {'name': X, 'label':x_label},
        {'name': W, 'label':w_label}
    ]
    edges = [
        {'from_': X, 'to_': Z, 'type_': bidirectedEdgeType},
        {'from_': X, 'to_': W},
        {'from_': X, 'to_': Y},
        {'from_': Z, 'to_': W},
        {'from_': Z, 'to_': Y},
        {'from_': W, 'to_': Y},
    ]
    return Graph(nodes=nodes,edges=edges)




class CausalGraph(Graph):
    def __init__(self,graph=None, nodes=[], edges=[], assignments=[]):
        self.assignments=assignments
        if graph is not None:
            super().__init__(nodes=self.parse_nodes([node['name'] for node in graph.nodes],assignments), edges=graph.edges)
        else:
            super().__init__(nodes=self.parse_nodes(nodes,assignments), edges=self.parse_edges(edges))


    def parse_nodes(self, nodes, assignments=[]):
        ns = []

        # convert tuple to list of dictionaries
        for name in nodes:
            label = name
            if name in assignments:
                label += ": " + str(assignments[name])
            n = {
                'name': name,
                'label': label
            }

            ns.append(n)

        return ns
    
    def parse_edges(self, edges):
        es = []

        # convert tuple to list of dictionaries
        for edge in edges:
            if len(edge) < 2: continue
            type = bidirectedEdgeType.id_ if len(edge)>2 and edge[2]=='bidirected' else 'directed'
            e = {'from_': edge[0], 'to_': edge[1], 'type_': type}
            
            es.append(e)

        return es
    
    def plot(self, scale=1):
        nodes = [node['name'] for node in self.nodes]
        n = len(nodes)
        corners = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            x = np.cos(angle) * scale
            y = np.sin(angle) * scale
            corners.append((x.item(), y.item()))
        positions = {nodes[i]: corners[i] for i in range(n)}
        return plot_causal_diagram(self, node_positions=positions)


def example1():
    nodes = ['a', 'b', 'c', 'd', 'e', 'f']
    edges = [
        ('e', 'a', 'bidirected'),
        ('e', 'f'),
        ('e', 'c'),
        ('e', 'd'),

        ('a','b'),
        ('a', 'f'),
        ('a', 'c'),

        ('b', 'f'),
        ('b', 'c'),
        ('b', 'd'),

        ('c', 'd'),
        ('c', 'f'),

        ('d', 'f'),
    ]
    assignments = {
        'a': ['race'], 
        'b': ['skincolor'], 
        'c': ['degree'], 
        'd': ['job_title','num_awards'], 
        'e': ['gender'], 
        'f': ['salary']
    }
    return CausalGraph(nodes=nodes,edges=edges,assignments=assignments)

def get_sfm_graph():
    return '''<NODES>
Y
Z
X
W

<EDGES>
X -> Y
X -> W
Z -> Y
Z -> W
W -> Y
X -- Z
'''

def project_to_sfm(graph, projection):
    # TODO: check for incorrect projection
    assignments = {
        'X': [*graph.assignments[projection['X']]],
        'Z': sum([graph.assignments[z] for z in projection['Z']],[]),
        'W': sum([graph.assignments[z] for z in projection['W']],[]),
        'Y': [*graph.assignments[projection['Y']]]
    }
    sfm = CausalGraph(graph=parseGraph(get_sfm_graph()), assignments=assignments)
    return sfm 