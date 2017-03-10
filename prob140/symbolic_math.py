import sympy as s
import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython


def _run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def declare_symbols(*args, **kwargs):
    assert all([str(a) == a for a in args]), \
        "Must pass in strings" # Shitty way of checking all strings
    assert all([',' not in a and ' ' not in a for a in args]), \
         "Cannot have commas or spaces in variable names"
    return s.symbols(",".join(args), **kwargs)    

def declare(*args, **kwargs):
    """
    Given arguments with variable names, this function creates
    the Symbols in the global namespace

    Issue: Doesn't work with lambda, since it is keyword protected
    """
    assert all([str(a) == a for a in args]), "Must pass in strings"
    assert all([',' not in a and ' ' not in a for a in args]), \
            "Cannot have commas or spaces in variable names"
    assert _run_from_ipython(), "This function only works in iPython; use Symbol() in scripts"
    ishell = get_ipython()
    command = "var(%s, **%s)"%(repr(",".join(args)), str(kwargs))
    ishell.ex(command)

def nicefy(expression):
    """
    Simplifies and factors the Sympy expression passed input
    (Internally calls factor(simplify(expression)))
    """
    return s.factor(s.simplify(expression))

def _nicefy_wrapper(function):
    def _wrapped(*args, **kwargs):
        value = function(*args, **kwargs)
        return nicefy(value)
    return _wrapped

@_nicefy_wrapper
def integrate_and_simplify(*args, **kwargs):
    return s.Integral(*args, **kwargs).doit()

class domain:
    real = (-1*s.oo, s.oo)
    positive = (0, s.oo)
    negative = (-1*s.oo, 0)

    @staticmethod
    def interval(a,b):
        return (a,b)