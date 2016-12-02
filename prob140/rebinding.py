from .prob140_tables import *
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


def to_probability_table(table):
	"""
		Returns the table, but with all the probability functionalities builtin
	"""
	table = table.copy()
	table.chart_colors = chart_colors
	table.prob_event = prob_event
	table.event = event
	table.plot_dist = plot_dist
	table.plot_event = plot_event
	table.domain = domain
	table.probability = probability
	table.probability_function = probability_function
	table.normalized = normalized
	table.expected_value = expected_value
	table.variance = variance
	table.sd = sd
	return table