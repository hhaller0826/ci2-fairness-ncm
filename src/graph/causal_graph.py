import itertools
import numpy as np
from graphviz import Source
from collections import deque

# Modified; originally from NCM Counterfactuals
class CausalGraph:
    def __init__(self, nodes, directed_edges=[], bidirected_edges=[]):
        self.de = directed_edges
        self.be = bidirected_edges

        self.v = list(nodes)
        self.set_v = set(nodes)
        self.pa = {v: set() for v in nodes}  # parents (directed edges)
        self.ch = {v: set() for v in nodes}  # children (directed edges)
        self.ne = {v: set() for v in nodes}  # neighbors (bidirected edges)
        self.bi = set(map(tuple, map(sorted, bidirected_edges)))  # bidirected edges

        for v1, v2 in directed_edges:
            self.pa[v2].add(v1)
            self.ch[v1].add(v2)

        for v1, v2 in bidirected_edges:
            self.ne[v1].add(v2)
            self.ne[v2].add(v1)
            self.bi.add(tuple(sorted((v1, v2))))

        self.pa = {v: sorted(self.pa[v]) for v in self.v}
        self.ch = {v: sorted(self.ch[v]) for v in self.v}
        self.ne = {v: sorted(self.ne[v]) for v in self.v}

        self._sort()
        self.v2i = {v: i for i, v in enumerate(self.v)}

        self.assignments = {} # name of each node

        self.cc = self._c_components()
        self.v2cc = {v: next(c for c in self.cc if v in c) for v in self.v} # maps v to the associated c component
        
        self.c2 = self._maximal_cliques()
        self.v2c2 = {v: [c for c in self.c2 if v in c] for v in self.v}

    def __iter__(self):
        return iter(self.v)
    
    def assign(self, names: dict):
        nm = names.keys() & self.set_v
        for v in nm:
            self.assignments[v] = names[v]

    def clear_assignments(self):
        self.assignments = {}

    def _sort(self):  
        """Sort V topologically
        Taken from NCMCounterfactuals
        """
        L = []
        marks = {v: 0 for v in self.v}

        def visit(v):
            if marks[v] == 2:
                return
            if marks[v] == 1:
                raise ValueError('The graph cannot have cycles.')

            marks[v] = 1
            for c in self.ch[v]:
                visit(c)
            marks[v] = 2
            L.append(v)

        for v in marks:
            if marks[v] == 0:
                visit(v)
        self.v = L[::-1]

    def _c_components(self):
        pool = set(self.v)
        cc = []
        while pool:
            cc.append({pool.pop()})
            while True:
                added = {k2 for k in cc[-1] for k2 in self.ne[k]}
                delta = added - cc[-1]
                cc[-1].update(delta)
                pool.difference_update(delta)
                if not delta:
                    break
        return [tuple(sorted(c, key=self.v2i.get)) for c in cc]

    def _maximal_cliques(self):
        """
        Finds all maximal cliques in an undirected graph.
        = All subsets of vertices with the two properties that each pair of vertices in one of the listed subsets is connected by an edge, and no listed subset can have any additional vertices added to it while preserving its complete connectivity
        Tryna find groups with bidirected edges between them

        Taken from NCMCounterfactuals
        """
        # find degeneracy ordering
        o = []
        p = set(self.v)
        while len(o) < len(self.v):
            v = min((len(set(self.ne[v]).difference(o)), v) for v in p)[1]
            o.append(v)
            p.remove(v)

        # brute-force bron_kerbosch algorithm
        c2 = set()

        def bron_kerbosch(r, p, x):
            if not p and not x:
                c2.add(tuple(sorted(r)))
            p = set(p)
            x = set(x)
            for v in list(p):
                bron_kerbosch(r.union({v}),
                              p.intersection(self.ne[v]),
                              x.intersection(self.ne[v]))
                p.remove(v)
                x.add(v)

        # apply brute-force bron_kerbosch with degeneracy ordering
        p = set(self.v)
        x = set()
        for v in o:
            bron_kerbosch({v},
                          p.intersection(self.ne[v]),
                          x.intersection(self.ne[v]))
            p.remove(v)
            x.add(v)

        return c2
    
    def ancestors(self, C):
        """
        Returns the ancestors of set C.
        """
        if len(C)==0: return None
        assert C.issubset(self.set_v)

        frontier = [c for c in C]
        an = {c for c in C}
        while len(frontier) > 0:
            cur_v = frontier.pop(0)
            for par_v in self.pa[cur_v]:
                if par_v not in an:
                    an.add(par_v)
                    frontier.append(par_v)

        return an
    
    def grandkids(self, C):
        """
        Returns the... reverse-ancestors of set C?
        """
        if len(C)==0: return None
        assert C.issubset(self.set_v)

        frontier = [c for c in C]
        ch = {c for c in C}
        while len(frontier) > 0:
            cur_v = frontier.pop(0)
            for ch_v in self.ch[cur_v]:
                if ch_v not in ch:
                    ch.add(ch_v)
                    frontier.append(ch_v)

        return ch
    
    def convert_set_to_sorted(self, C):
        return [v for v in self.v if v in C]

    def plot(self, scale=1):
        n = len(self.v)
        corners = []
        for i in range(n):
            angle = 2 * np.pi * i / n
            x = np.cos(angle) * scale
            y = np.sin(angle) * scale
            corners.append((x.item(), y.item()))
        positions = {self.v[i]: corners[i] for i in range(n)}

        # return plot_causal_diagram(self, node_positions=positions)
        dot_text = self.convert_to_dot(node_positions=positions)
        return Source(dot_text, engine="neato")

    def convert_to_dot(self, path_1=[], path_2=[], nodes=[], node_positions={}):
        dot_str = "digraph G {\n  rankdir=LR;\n"
        # Add nodes with positions
        for node in self.v:
            pos = (
                f'pos="{node_positions[node][0]},{node_positions[node][1]}!"'
                if node in node_positions
                else ""
            )
            fillcolor = "style=filled, fillcolor=lightblue" if node in nodes else ""
            dot_str += f'  {node} [label="{self.assignments.get(node,node)}" {pos} {fillcolor}];\n'

        for a,b in self.de:
            dot_str += f'  {a} -> {b};\n'

        for a,b in self.be:
            arrow_type = (f"[dir=both, style=dashed, constraint=false, splines=curved]")
            dot_str += f'  {a} -> {b} {arrow_type};\n'

        return dot_str + "}"
    
def graph_search(cg, v1, v2=None, edge_type="direct"):
    """
    Uses BFS to check for a path between v1 and v2 in cg. If v2 is None, returns all reachable nodes.
    """
    assert edge_type in ["direct", "bidirect"]
    assert v1 in cg.set_v
    assert v2 in cg.set_v or v2 is None

    q = deque([v1])
    seen = {v1}
    while len(q) > 0:
        cur = q.popleft()
        if edge_type == "direct":
            cur_ne = cg.ch[cur]
        else:
            cur_ne = cg.ne[cur]

        for ne in cur_ne:
            if ne not in seen:
                if v2 is not None and ne == v2:
                    return True
                seen.add(ne)
                q.append(ne)

    if v2 is None:
        return seen

    return False