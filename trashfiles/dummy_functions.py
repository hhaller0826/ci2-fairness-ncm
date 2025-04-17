
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
        # TODO: Checks (inc acyclic check)
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

class SFM(CausalGraph):
    def __init__(self, assignments=[]):
        self.graph = parseGraph(get_sfm_graph())
        super().__init__(self.graph, assignments)

    def parse_nodes(self, nodes, assignments=[]):
        self.X = self.parse_node('X', assignments)
        self.Y = self.parse_node('Y', assignments)
        self.Z = self.parse_node('Z', assignments)
        self.W = self.parse_node('W', assignments)
        return [self.X, self.Y, self.Z, self.W]
    
    def parse_node(self, name, assignments=[]):
        return {
            'name': name,
            'label': name if name not in assignments else name + str(assignments[name])
        }

def project_to_sfm(graph, projection):
    # TODO: check for incorrect projection
    assignments = {
        'X': [*graph.assignments[projection['X']]],
        'Z': sum([graph.assignments[z] for z in projection['Z']],[]),
        'W': sum([graph.assignments[z] for z in projection['W']],[]),
        'Y': [*graph.assignments[projection['Y']]]
    }
    sfm = SFM(assignments=assignments)
    return sfm 

def evaluate_fairness_measures(graph, x0, x1):
    print("GENERAL")
    print(f"\tTE_[x0={x0},x1={x1}](y) = value")
    print(f"\tExp-SE_x(y) = value")
    print(f"\tNDE_[x0={x0},x1={x1}](y) = value")
    print(f"\tNIE_[x0={x0},x1={x1}](y) = value")

    print("\nx-SPECIFIC")
    print(f"\tETT_[x0={x0},x1={x1}](y|x) = value")
    print(f"\tCtf-SE_[x0={x0},x1={x1}](y) = value")
    print(f"\tCtf-DE_[x0={x0},x1={x1}](y|x) = value")
    print(f"\tCtf-IE_[x0={x0},x1={x1}](y|x) = value")

    print("\nz-SPECIFIC")
    print(f"\tz-TE_[x0={x0},x1={x1}](y|x) = value")
    print(f"\tz-DE_[x0={x0},x1={x1}](y|x) = value")
    print(f"\tz-IE_[x0={x0},x1={x1}](y|x) = value")


def fairness_cookbook(data, X, W, Z, Y, x0, x1):
    return (data, X, W, Z, Y, x0, x1)

def evaluate_general_measures(graph, x0, x1):
    print("GENERAL")
    print(f"\tTE_[x0={x0},x1={x1}](y) = value")
    print(f"\tExp-SE_x(y) = value")
    print(f"\tNDE_[x0={x0},x1={x1}](y) = value")
    print(f"\tNIE_[x0={x0},x1={x1}](y) = value")

def evaluate_xspec_measures(graph, x0, x1):
    print("x-SPECIFIC")
    print(f"\tETT_[x0={x0},x1={x1}](y|x) = value")
    print(f"\tCtf-SE_[x0={x0},x1={x1}](y) = value")
    print(f"\tCtf-DE_[x0={x0},x1={x1}](y|x) = value")
    print(f"\tCtf-IE_[x0={x0},x1={x1}](y|x) = value")

def evaluate_zspec_measures(graph, x0, x1):
    print("z-SPECIFIC")
    print(f"\tz-TE_[x0={x0},x1={x1}](y|x) = value")
    print(f"\tz-DE_[x0={x0},x1={x1}](y|x) = value")
    print(f"\tz-IE_[x0={x0},x1={x1}](y|x) = value")

def autoplot(cookbook, decompose="", dataset=None, type=''):
    print("pretend this is a graph")
    (_, _, _, _, _, x0, x1) = cookbook
    if decompose == "gen":
        evaluate_general_measures(None, x0, x1)
    elif decompose == "zspec":
        evaluate_zspec_measures(None, x0, x1)
    else:
        evaluate_xspec_measures(None, x0, x1)

def fair_predictions(data, sfm, x0, x1, bn):
    pass

def fairadapt(graph, data): pass
def predict(fair_pred, data): pass
def fair_decisions(data, sfm, x0, x1, po_transform, po_diff_sign): 
    return fairness_cookbook(data, sfm.X, sfm.W, sfm.Z, sfm.Y, x0=x0, x1=x1)

def probability(y, val, evidence=None, intervention=None):
    p = "P(" + y + "=" + str(val)
    if evidence is not None or intervention is not None: 
        p += " | "
    cond = [x+"="+str(evidence[x]) for x in evidence] if evidence else []
    do = ["do("+x+"="+str(intervention[x])+")" for x in intervention] if intervention else []

    return p + ", ".join(cond+do) + ")"
    
def total_variation(y, val, x, x0, x1=1):
    val,x0,x1 = str(val),str(x0),str(x1)
    tv = "P(" + y + "=" + val + " | " + x + "=" + x1 + ")"
    tv += " - P(" + y + "=" + val + " | " + x + "=" + x0 + ")"
    return probability(y,val,{x:x1})+" - "+probability(y,val,{x:x0})

def evidence(e=[]):
    ev = ""
    for i in range(len(e)):
        if i==0: ev += " | "
        (node,val) = e[i]
        ev += node + "=" + str(val)
        if i < len(e)-1: ev += ", "
    return ev

def condition(c=[]):
    if len(c)==0: return ""
    cond = "_{"
    for i in range(len(c)):
        if i>0: cond += ","
        (node,val) = c[i]
        cond += node + "=" + str(val)
    return cond + "}"

def total_effect(y, e=[], c=[]):
    cond = condition(c)
    te = "P(" + y + cond
    te += evidence([(val[0]+cond,val[1]) for val in e])
    return te + ")"

class CTFQuery():
    def __init__(self):
        pass


def ett(y, c=[], e=[]):
    return "P(" + y + condition(c) + evidence(e) + ")"

def pnps(y, c=[], e=[]):
    return "P(Y_{X=x" + "} = y|X=x', Y=y'"