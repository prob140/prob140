from .prob140 import *
from .version import *
from datascience import *

## Binding; still in debate
Table.chart_colors = chart_colors
Table.prob_event = prob_event
Table.event = event
Table.plot_dist = plot_dist
Table.plot_event = plot_event
Table.domain = domain
Table.probability = probability
Table.probability_function = probability_function
Table.normalized = normalized
Table.expected_value = expected_value
Table.variance = variance
Table.sd = sd


def ProbabilityTable():
	return Table()

def to_probability_table(table):
	return Table.copy(table)
