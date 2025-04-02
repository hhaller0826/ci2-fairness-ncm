
import networkx as nx
import matplotlib.pyplot as plt



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

def process_data(d): pass