from .version import __version__
from .rebinding import Table, Plot, Plots, JointDistribution
from .markov_chains import MarkovChain

from .single_variable import emp_dist

from .plots import (
    Plot_multivariate_normal_cond_exp,
    Plot_continuous,
    Plot_norm,
    Plot_3d,
    Plot_bivariate_normal,
    Scatter_multivariate_normal,
)
from .symbolic_math import *

from .temp import visualize_cr
