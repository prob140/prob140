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

def symbols(*args,**kwargs):
    assert all([str(a)==a for a in args]), "Must pass in strings" # Shitty way of checking all strings
    assert all([',' not in a and ' ' not in a for a in args]), "Cannot have commas or spaces in variable names"
    return s.symbols(",".join(args),**kwargs)    

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
    command = "var(%s, **%s)"%(",".join(map(repr,args)),str(kwargs))
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

class RandomVariable:
    """
    RandomVariable stuff
    """

    def __init__(self, pdf,variable, domain):
        """
            Syntax:
            >>> declare('x')
            >>> X = RandomVariable(1,x,domain(0,1)) # Uniform on 0,1
        """
        self.pdf = pdf
        self.variable = variable
        self.domain = domain
        self.totaldomain = (variable, domain[0], domain[1])
    
    def expectation_of(self, function, var=None):
        """
        Finds the expectation of a function with respect to the random variable

        >>> declare('x','y')
        >>> X = RandomVariable(1,x,domain(0,1)) # Uniform on 0,1
        >>> EX2 = X.expectation_of(x**2)

        If the function has a different variable than whatever the random variable used,
        then use the *var* parameter. Here's an example:
        
        >>> declare('x','y')
        >>> X = RandomVariable(1,x,domain(0,1)) # Uniform on 0,1
        >>> EX2 = X.expectation_of(y**2,var=y)

        You can use this function to compute the main features of a probability distribution
        >>> declare('x')
        >>> X = RandomVariable(1,x,domain(0,1)) # Uniform on 0,1
        >>> 1 == X.expectation_of(1)
        True
        >>> EX = X.expectation_of(x)
        >>> VAR = X.expectation_of(x**2) - EX**2
        """
        if not isinstance(function, s.Expr):
            function = s.Number(function)
        if var:
            function = function.subs(var, self.variable)
        return nicefy(s.Integral(function*self.pdf, self.totaldomain).doit())

    def _repr_latex_(self):
        variable_name = s.latex(str(self.variable).upper())
        variable = s.latex(self.variable)
        pdf = s.latex(self.pdf)
        domain = s.latex(self.domain)
        return "Random variable with $P(%s \\in d%s) = %s$, where $%s \\in %s$"%(variable_name, variable, pdf, variable, domain)
    
    def with_parameters(self, *args):
        """
        usage:
        
            specific_var = RV.with_parameters(mu,0,sigma=1)
        """
        assert len(args) %2 == 0
        new_vals = {args[i]:args[i+1] for i in range(0, len(args), 2)}
        new_pdf = self.pdf.subs(new_vals)
        return RandomVariable(new_pdf, self.variable, self.domain)

def _plot_RV(RV,domain,**kwargs):
    # There should be a checker here which sees if there are any other variables in the expression
    new_domain = (max(domain[0], RV.domain[0]), min(domain[1], RV.domain[1]))
    _plot_expression(RV.pdf, RV.variable, domain,new_domain,**kwargs)


def _plot_expression(expr,variable,domain,actual_domain=None,**kwargs):
    f = lambdify(variable, expr)
    if actual_domain is not None:
        domain_limit = lambda x: f(x) if x < actual_domain[1] and x > actual_domain[0] else 0
    else:
        domain_limit = f
    x_vals = np.linspace(domain[0], domain[1], 200)
    y_vals = [domain_limit(x) for x in x_vals]
    plt.plot(x_vals, y_vals, **kwargs)
    plt.xlim(domain)