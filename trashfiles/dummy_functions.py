
import networkx as nx
import matplotlib.pyplot as plt
import warnings


def plot_causal_diagram(nodes, edges):
    # Create a directed graph
    graph = nx.DiGraph()

    # Add nodes
    graph.add_nodes_from(nodes)

    # Add directed edges
    graph.add_edges_from(edges)

    # Draw the graph
    nx.draw(graph, with_labels=True, node_color='skyblue', node_size=1500, 
            arrowstyle='-|>', arrowsize=20)

    # Display the graph
    plt.show()

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
            'incomplete': [1, 1, 0, None, 1, 0]}

class Model():
    def __init__(self, data, distribution, graph, assignments):
        self.check_assignments(data, assignments)
        self.data = data
        self.distribution = distribution
        self.cg = graph
        self.assignments = assignments
        
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
            print("It is okay to exclude some features but they will not be used in the causal analysis.")
        