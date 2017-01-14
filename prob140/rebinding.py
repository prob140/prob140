from .single_variable import *
from .multi_variable import *
import inspect

from .version import *
from datascience import *


def ProbabilityTable():
	return Table()

def domain(self,*args):
	if len(args) == 1:
		return single_domain(self,args[0])
	return multi_domain(self,*args)

def probability_function(self, pfunc):

	num_args = len(inspect.signature(pfunc).parameters)

	assert num_args == 1 or num_args == 2, "probability function must have 1 or 2 arguments"
	assert self.num_columns == 1 or self.num_columns == 2, "Table must have 1 or 2 columns"

	if self.num_columns == 1:
		if num_args == 1:
			return single_probability_function(self, pfunc)
		raise ValueError("Probability function must take in one argument")

	elif self.num_columns == 2:
		if num_args == 2:
			return multi_probability_function(self, pfunc)
		raise ValueError("Probability function must take in two arguments")

## Binding; still in debate

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


