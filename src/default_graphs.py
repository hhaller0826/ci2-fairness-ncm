
# TODO

def get_predefined_graph(type: str):
    type = type.lower()
    if type=='bow':
        return get_bow_graph()
    elif type=='backdoor':
        return get_backdoor_graph()
    elif type=='sfm':
        return get_sfm_graph()
    else:
        raise ValueError('Unknown graph type: {}'.format(type))


def get_backdoor_graph():
    return '''<NODES>
Z
X
Y

<EDGES>
Z -> X
Z -> Y
X -> Y
'''

def get_bow_graph():
    return '''<NODES>
X
Y

<EDGES>
X -> Y
X <-> Y
'''

def get_sfm_graph():
    return '''<NODES>
Z
X
Y
W

<EDGES>
X -> Y
X -> W
Z -> Y
Z -> W
W -> Y
X <-> Z
'''