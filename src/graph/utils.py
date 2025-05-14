import numpy as np
from src.graph.causal_graph import CausalGraph

graph_choices = {'backdoor', 'bow', 'sfm', 'frontdoor', 'napkin', 'simple'}

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
        nodes = [Z, Y, X, W]
        be = [(X, Z)]
        de = [
            (X, Y),
            (X, W),
            (Z, Y),
            (Z, W),
            (W, Y)
        ]

    elif type == 'frontdoor':
        X, Z, Y = 'X', 'Z', 'Y'
        nodes = [X, Z, Y]
        be = [(X, Y)]
        de = [
            (X, Z),
            (Z, Y),
        ]

    elif type == 'napkin':
        X, Z, W, Y = 'X', 'Z', 'W', 'Y'
        nodes = [Z, Y, X, W]
        be = [(X, W), (W, Y)]
        de = [
            (W, Z),
            (Z, X),
            (X, Y),
        ]

    elif type == 'simple':
        X, Y = 'X', 'Y'
        nodes = [X, Y]
        de = [(X, Y)]

    return CausalGraph(nodes, directed_edges=de, bidirected_edges=be)