from .single_variable import *
from .multi_variable import *
from .markov_chains import MarkovChain, toMarkovChain
from .symbolic_math import *
from .plots import Plot_continuous

from datascience import *


def ProbabilityTable():
    return Table()


def domain(self, *args):
    if len(args) == 1:
        return single_domain(self, args[0])
    return multi_domain(self, *args)


def states(self, values):
    table = self.with_column('State', values)
    table.move_to_start('State')
    return table

## Binding; still in debate

Table.value = domain
Table.values = domain
Table.states = states
Table.chart_colors = chart_colors
Table.prob_event = prob_event
Table.event = event
Table.domain = domain
Table.probability = probability
Table.probability_function = probability_function
Table.normalized = normalized
Table.expected_value = expected_value
Table.variance = variance
Table.sd = sd
Table.toJoint = toJoint
Table.sample_from_dist = sample_from_dist
Table.cdf = cdf

# Markov Chain stuff
Table.toMarkovChain = toMarkovChain
Table.transition_probability = transition_probability
Table.transition_function = transition_function
