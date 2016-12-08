import math
import numpy as np
import matplotlib.pyplot as plt
import collections
import itertools
import abc
import sys
import warnings

import matplotlib

from datascience import *

inf = math.inf
rgb = matplotlib.colors.colorConverter.to_rgb

def check_valid_probability_table(table):
    assert table.num_columns == 2, "In order to run a Prob140 function, your table must have 2 columns: (1) a Values column (2) a Probability column"
    assert all(table.column(1)>=0), "Probabilities must be non-negative"


def prob_event(self, x):
    """
    Finds the probability of an event x

    Parameters
    ----------
    x : float or Iterable
        An event represented either as a specific value in the domain or a
        subset of the domain

    Returns
    -------
    float
        Probability of the event

    Examples
    --------

    >>> dist = FiniteDistribution().domain([1,2,3,4]).probability([1/4,1/4,1/4,1/4])
    >>> dist.prob_event(2)
    0.25

    >>> dist.prob_event([2,3])
    0.5

    >>> dist.prob_event(np.arange(1,5))
    1.0

    """
    check_valid_probability_table(self)
    if isinstance(x, collections.Iterable):
        return sum(self.prob_event(k) for k in x)
    else:
        domain = self.column(0)
        prob = self.column(1)
        return sum(prob[np.where(domain == x)])


def event(self, x):
    """
    Shows the probability that distribution takes on value x or range of
    values x.

    Parameters
    ----------
    x : float or Iterable
        An event represented either as a specific value in the domain or a
        subset of the domain

    Returns
    -------
    FiniteDistribution
        Shows the probabilities of each value in the event


    Examples
    --------
    >>> dist = FiniteDistribution().domain([1,2,3,4]).probability([1/4,1/4,1/4,1/4])
    >>> dist.event(2)
    Domain | Probability
    2      | 0.25

    >>> dist.event([2,3])
    Domain | Probability
    2      | 0.25
    3      | 0.25
    """
    check_valid_probability_table(self)

    if not isinstance(x, collections.Iterable):
        x = [x]    
    probabilities = [self.prob_event(k) for k in x]
    return Table().with_columns('Outcome',x,'Probability',probabilities)


def Plot(dist, width=1, mask=[], event=[], **vargs):
    """
    Plots the histogram for a Distribution

    Parameters
    ----------
    width (optional) : float
        Width of the intervals (default: 1)
    mask (optional) : boolean array or list of boolean arrays
        Colors the parts of the histogram associated with each mask (
        default: no mask)
    vargs
        See pyplot's additional optional arguments


    """
    options = Table.default_options.copy()
    options.update(vargs)

    self = dist
    check_valid_probability_table(dist)

    domain_label = self.labels[0]
    self = self.sort(domain_label)
    domain = self.column(0)
    prob = self.column(1)

    start = min(domain)
    end = max(domain)

    end = (end // width + 1) * width

    if len(event) != 0:

        domain = set(self.column(0))

        def prob(x):
            return np.array([self.prob_event(a) for a in list(x)])

        if isinstance(event[0], collections.Iterable):
            # If event is a list of lists

            colors = list(
                itertools.islice(itertools.cycle(self.chart_colors), len(event) + 1))
            for i in range(len(event)):
                plt.bar(event[i], prob(event[i]) * 100, align="center", color=colors[i], width=1, alpha=0.7, **vargs)
                domain -= set(event[i])

            domain = np.array(list(domain))
            plt.bar(domain, prob(domain) * 100, align="center", color=colors[-1], width=1, alpha=0.7, **vargs)

        else:
            # If event is just a list

            plt.bar(event, prob(event) * 100, align="center", width=1, color="gold", alpha=0.7, **vargs)
            domain = np.array(list(set(self.column(0)) - set(event)))
            plt.bar(domain, prob(domain) * 100, align="center", color="darkblue", width=1, alpha=0.7, **vargs)


    elif len(mask) == 0:
        # no mask or event
        self.hist(counts=domain_label,
                  bins=np.arange(start - width / 2, end + width, width),
                  **vargs)
    else:
        if isinstance(mask[0], collections.Iterable):
            # If mask is a list of lists

            colors = list(
                itertools.islice(itertools.cycle(self.chart_colors), len(mask)))
            for i in range(len(mask)):
                plt.bar(domain[mask[i]], prob[mask[i]] * 100, align="center",
                        color=colors[i], width=1, alpha=0.7, **vargs)

        else:
            # If mask is just a list

            plt.bar(domain[mask], prob[mask] * 100, align="center", color="darkblue", width=1, alpha=0.7, **vargs)
            plt.bar(domain[np.logical_not(mask)], prob[np.logical_not(mask)] * 100,
                    align="center", color="gold", width=1, alpha=0.7, **vargs)

            # dist1 = FiniteDistribution().domain(domain[mask]).probability(prob[mask])
            # dist2 = FiniteDistribution().domain(domain[np.logical_not(mask)]).probability(prob[np.logical_not(mask)])
            # DiscreteDistribution.Plot("1", dist1, "2", dist2, width=width, **vargs)

    plt.xlabel(domain_label)
    plt.ylabel("Percent per unit")

    mindistance = 0.9 * max(min([self.column(0)[i] - self.column(0)[i - 1] for i in range(1, self.num_rows)]),1)

    plt.xlim((min(self.column(0)) - mindistance - width / 2, max(self.column(0))
              + mindistance + width / 2))


def Plots(*labels_and_dists, width=1, **vargs):
    """
    Class method for overlay multiple distributions

    Parameters
    ----------
    labels_and_dists : Even number of alternations between Strings and
    FiniteDistributions
        Each distribution must have a label associated with it
    width (optional) : float
        Width of the intervals (default: 1)
    vargs
        See pyplot's documentation

    """
    # assert len(labels_and_dists) % 2 == 0, 'Even length sequence required'
    options = Table.default_options.copy()
    options.update(vargs)


    i = 0

    domain = set()
    while i < len(labels_and_dists):
        label = labels_and_dists[i]
        dist = labels_and_dists[i + 1]
        check_valid_probability_table(dist)

        domain = domain.union(dist.column(0))
        i += 2

    domain = np.array(list(domain))

    i = 0
    distributions = ["Value", domain]
    while i < len(labels_and_dists):
        distributions.append(labels_and_dists[i])
        dist = labels_and_dists[i + 1]
        probability = np.vectorize(lambda x: prob_event(dist,x), otypes=[np.float])(domain)
        distributions.append(probability)
        i += 2

    result = Table().with_columns(*distributions)

    result.chart_colors = Table.chart_colors

    start = min(domain)
    end = max(domain)
    end = (end // width + 1) * width
    result.hist(counts="Value",
                bins=np.arange(start - width / 2, end + width, width),
                **vargs)
    domain = np.sort(domain)

    mindistance = 0.9 * max(min([domain[i] - domain[i - 1] for i in range(1, len(domain))]), 1)

    plt.xlim((min(domain) - mindistance - width / 2, max(domain) + mindistance +
              width / 2))


def single_domain(self, values):
    """
    Assigns domain values to a FiniteDistribution

    Parameters
    ----------
    values : List or Array
        Values to put into the domain

    Returns
    -------
    FiniteDistibution
        FiniteDistribution with that domain
    """
    table = self.with_column('Value', values)
    table.move_to_start('Value')
    return table


def probability_function(self, pfunc):
    """
    Assigns probabilities to a FiniteDistribution via a probability
    function. The probability function is applied to each value of the
    domain

    Parameters
    ----------
    pfunc : univariate function
        Probability function of the FiniteDistribution

    Returns
    -------
    FiniteDistribution
        FiniteDistribution with those probabilities

    """
    domain_names = self.labels
    values = np.array(self.apply(pfunc, domain_names)).astype(float)
    if any(values < 0):
        warnings.warn("Probability cannot be negative")
    return self.with_column('Probability', values)


def probability(self, values):
    """
    Assigns probabilities to domain values.

    Parameters
    ----------
    values : List or Array
        Values that must correspond to the domain in the same order

    Returns
    -------
    FiniteDistribution
        FiniteDistribution with those probabilities
    """
    if any(np.array(values) < 0):
        warnings.warn("Probability cannot be negative")
    return self.with_column('Probability', values)


def normalized(self):
    """
    Returns the distribution by making the proabilities sum to 1

    Returns
    -------
    FiniteDistribution
        Normalized FiniteDistribution
    """
    column_label = self.labels[-1]
    return self.with_column(column_label,self.column(column_label)/sum(self.column(column_label)))


def expected_value(self):
    """
    Finds expected value of distribution

    Returns
    -------
    float
        Expected value

    """
    check_valid_probability_table(self)
    self = normalized(self)
    ev = 0
    for domain, probability in self.rows:
        ev += domain * probability
    return ev


def variance(self):
    """
    Finds variance of distribution

    Returns
    -------
    float
        Variance
    """
    check_valid_probability_table(self)

    self = normalized(self)
    var = 0
    ev = self.expected_value()
    for domain, probability in self.rows:
        var += (domain - ev) ** 2 * probability
    return var


def sd(self):
    """
    Finds standard deviation of FiniteDistribution

    Returns
    -------
    float
        Standard Deviation
    """
    return math.sqrt(self.variance())

# Brighter colors than the default Table class
chart_colors = (
    rgb("darkblue"),
    rgb("gold"),
    rgb("lime"),
    rgb("red"),
    rgb("darkviolet"),
    rgb("brown"),
    rgb("darkgreen"),
    rgb("black"),
    rgb("cyan"),
)


