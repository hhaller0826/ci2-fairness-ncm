from collections.abc import Iterable

"""Modified Kevin"""
class CTFTerm:
    def __init__(self, vars, do_vals: dict={}):
        self.vars = set(*vars)
        self.do_vals = do_vals

    def __str__(self):
        out = "["
        for i, var in enumerate(self.vars):
            out += var
            if i != len(self.vars) - 1:
                out += ", "
        if len(self.do_vals) > 0:
            out += " | do("
            for i, var in enumerate(self.do_vals):
                out += "{} = {}".format(var, self.do_vals[var])
                if i != len(self.do_vals) - 1:
                    out += ", "
            out += ")"
        out += "]"
        return out