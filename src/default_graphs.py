
# TODO

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