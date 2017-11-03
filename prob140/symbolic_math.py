import sympy as s
from IPython import get_ipython


def _run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def declare_symbols(*args, **kwargs):
    """
    Creates symbols and returns them as python objects Good idea to use this
    when working inside a function or want to assign different names to similar
    variables
    """
    assert all([str(a) == a for a in args]), 'Must pass in strings'
    assert all([',' not in a and ' ' not in a for a in args]), \
        'Cannot have commas or spaces in variable names'
    return s.symbols(','.join(args), **kwargs)


def declare(*args, **kwargs):
    """
    Given arguments with variable names, this function creates
    the Symbols in the global namespace

    Issue: Doesn't work with lambda, since it is keyword protected
    """
    assert all([str(a) == a for a in args]), 'Must pass in strings'
    assert all([',' not in a and ' ' not in a for a in args]), \
        'Cannot have commas or spaces in variable names'
    assert _run_from_ipython(), \
        'This function only works in iPython; use Symbol() in scripts'
    ishell = get_ipython()
    interval = kwargs.pop('interval', None)
    if interval is not None:
        a, b = interval
        if b < 0:
            kwargs['negative'] = True
        elif b <= 0:
            kwargs['nonpositive'] = True
        elif a > 0:
            kwargs['positive'] = True
        elif a >= 0:
            kwargs['nonnegative'] = True

    command = 'var(%s, **%s)' % (repr(','.join(args)), str(kwargs))
    ishell.ex(command)


def rearrange_1(expression):
    """
    Simplifies and factors the Sympy expression passed input
    (Internally calls factor(simplify(expression)))
    """
    return s.factor(s.simplify(expression))


def _nicefy_wrapper(function):
    def _wrapped(*args, **kwargs):
        value = function(*args, **kwargs)
        return rearrange_1(value)
    return _wrapped


@_nicefy_wrapper
def integrate_and_simplify(*args, **kwargs):
    """
        A wrapper for Integral(...).doit()

        Pass in the same parameters as you would to Integral
    """
    return s.Integral(*args, **kwargs).doit()


class domain:
    real = (-1 * s.oo, s.oo)
    positive = (0, s.oo)
    negative = (-1 * s.oo, 0)

    @staticmethod
    def interval(a, b):
        return (a, b)


def rearrange_2(expression):
    """
    Returns an equivalent version of the expression with unconstrained
    variables. As a result, this will cause expressions to look more like they
    were typed without the simplifications that SymPy makes by default.
    """
    free_variables = list(expression.free_symbols)
    for symbol in free_variables:
        new_symbol = s.Symbol(symbol.name)
        expression = expression.replace(symbol, new_symbol)
    return expression
