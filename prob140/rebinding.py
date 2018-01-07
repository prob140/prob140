from .single_variable import *
from .multi_variable import *
from .markov_chains import MarkovChain, to_markov_chain
from .symbolic_math import *
from .plots import Plot_continuous

from datascience import *


def domain(self, *args):
    if len(args) == 1:
        return single_domain(self, args[0])
    return multi_domain(self, *args)


def states(self, values):
    table = self.with_column('State', values)
    table.move_to_start('State')
    return table

# Binding
Table.values = domain
Table.value = domain
Table.states = states
Table.chart_colors = chart_colors
Table.prob_event = prob_event
Table.event = event
Table.domain = domain
Table.probability = probability
Table.probability_function = probability_function
Table.normalized = normalized
Table.ev = ev
Table.var = var
Table.sd = sd
Table.to_joint = to_joint
Table.sample_from_dist = sample_from_dist
Table.cdf = cdf
Table.remove_zeros = remove_zeros

# Markov Chain stuff
Table.to_markov_chain = to_markov_chain
Table.transition_probability = transition_probability
Table.transition_function = transition_function
