import numpy as np
from src.graph.causal_graph import CausalGraph

graph_choices = {'backdoor', 'bow', 'sfm'}

def get_predefined_graph(type) -> CausalGraph:
    assert type in graph_choices

    nodes = None
    be = []
    de = []

    if type == 'backdoor':
        X, Z, Y = 'X', 'Z', 'Y'
        nodes = [X, Z, Y]
        de = [
            (Z, X),
            (Z, Y),
            (X, Y)
        ]

    elif type == 'bow':
        X, Y = 'X', 'Y'
        nodes = [X, Y]
        be = [(X, Y)]
        de = [(X, Y)]

    elif type == 'sfm':
        X, Z, W, Y = 'X', 'Z', 'W', 'Y'
        # nodes = [X, Z, W, Y]
        nodes = [Z, Y, X, W]
        be = [(X, Z)]
        de = [
            (X, Y),
            (X, W),
            (Z, Y),
            (Z, W),
            (W, Y)
        ]
    return CausalGraph(nodes, directed_edges=de, bidirected_edges=be)