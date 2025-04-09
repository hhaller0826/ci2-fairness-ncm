import numpy as np
from src.causalaibook.utils import plot_causal_diagram

def get_positions(nodes, scale=1):
    n = len(nodes)
    corners = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        x = np.cos(angle) * scale
        y = np.sin(angle) * scale
        corners.append((x.item(), y.item()))
    return {nodes[i]: corners[i] for i in range(n)}

def plot_causal_graph(G, scale=1):
   return plot_causal_diagram(G, node_positions=get_positions([node['name'] for node in G.nodes], scale))
