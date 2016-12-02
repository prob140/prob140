import math
import numpy as np
import matplotlib.pyplot as plt
import collections
import itertools
import abc
import sys
import numbers
import warnings
import tkinter

import matplotlib

from datascience import *

inf = math.inf
rgb = matplotlib.colors.colorConverter.to_rgb

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
    if isinstance(x, collections.Iterable):
        return sum(self.prob_event(k) for k in x)
    else:
        domain = self._columns["Domain"]
        prob = self._columns["Probability"]
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
    if isinstance(x, collections.Iterable):
        probabilities = [self.prob_event(k) for k in x]
        return Table().domain(x).probability(probabilities)
    else:
        return Table().domain([x]).probability([self.prob_event(x)])


def plot_dist(self, width=1, mask=[], **vargs):
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
    self = self.sort("Domain")
    domain = self["Domain"]
    prob = self["Probability"]

    start = min(domain)
    end = max(domain)

    end = (end // width + 1) * width

    if len(mask) == 0:
        self.hist(counts="Domain",
                  bins=np.arange(start - width / 2, end + width, width),
                  **vargs)
    else:
        if isinstance(mask[0], collections.Iterable):
            colors = list(
                itertools.islice(itertools.cycle(self.chart_colors), len(mask)))
            for i in range(len(mask)):
                plt.bar(domain[mask[i]], prob[mask[i]] * 100, align="center",
                        color=colors[i], width=1, alpha=0.7)
            plt.xlabel("Domain")
            plt.ylabel("Percent per unit")
        else:
            plt.bar(domain[mask], prob[mask] * 100, align="center", color="darkblue", width=1, alpha=0.7)
            plt.bar(domain[np.logical_not(mask)], prob[np.logical_not(mask)] * 100,
                    align="center", color="gold", width=1, alpha=0.7)
            plt.xlabel("Domain")
            plt.ylabel("Percent per unit")
            # dist1 = FiniteDistribution().domain(domain[mask]).probability(prob[mask])
            # dist2 = FiniteDistribution().domain(domain[np.logical_not(mask)]).probability(prob[np.logical_not(mask)])
            # DiscreteDistribution.Plot("1", dist1, "2", dist2, width=width, **vargs)

    mindistance = 0.9 * max(min([self['Domain'][i] - self['Domain'][i - 1] for i in range(1, self.num_rows)]),1)

    plt.xlim((min(self['Domain']) - mindistance - width / 2, max(self['Domain'])
              + mindistance + width / 2))


def plot_event(self, event, width=1, **vargs):
    """

    Parameters
    ----------
    event : List or List of lists
        Each list represents an event which will be colored differently
        by the plot
    width (optional) : float
        Width of the intervals. Actually not implemented right now!
    vargs
        See pyplot's additional optional arguments

    """
    self = self.sort("Domain")
    if len(event) == 0:
        self.plot_dist(width=width, **vargs)

    else:

        mindistance = 0.9 * max(min([self['Domain'][i] - self['Domain'][i - 1] for i in range(1, self.num_rows)]), 1)

        plt.xlim((min(self['Domain']) - mindistance - width / 2, max(self['Domain'])
                  + mindistance + width / 2))

        domain = set(self["Domain"])

        def prob(x):
            return np.array([self.prob_event(a) for a in list(x)])

        if isinstance(event[0], collections.Iterable):
            colors = list(
                itertools.islice(itertools.cycle(self.chart_colors), len(event) + 1))
            for i in range(len(event)):
                plt.bar(event[i], prob(event[i]) * 100, align="center", color=colors[i], width=1, alpha=0.7)
                domain -= set(event[i])

            domain = np.array(list(domain))
            plt.bar(domain, prob(domain) * 100, align="center", color=colors[-1], width=1, alpha=0.7)
            plt.xlabel("Domain")
            plt.ylabel("Percent per unit")
        else:

            plt.bar(event, prob(event) * 100, align="center", width=1, color="gold", alpha=0.7)
            domain = np.array(list(set(self["Domain"]) - set(event)))
            plt.bar(domain, prob(domain) * 100, align="center", color="darkblue", width=1, alpha=0.7)
            plt.xlabel("Domain")
            plt.ylabel("Percent per unit")


def Plot(*labels_and_dists, width=1, **vargs):
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
        domain = domain.union(dist._columns["Domain"])
        i += 2

    domain = np.array(list(domain))

    i = 0
    distributions = ["Domain", domain]
    while i < len(labels_and_dists):
        distributions.append(labels_and_dists[i])
        dist = labels_and_dists[i + 1]
        probability = np.vectorize(dist.prob_event, otypes=[np.float])(domain)
        distributions.append(probability)
        i += 2

    result = Table().with_columns(*distributions)

    result.chart_colors = Table.chart_colors

    start = min(domain)
    end = max(domain)
    end = (end // width + 1) * width
    result.hist(counts="Domain",
                bins=np.arange(start - width / 2, end + width, width),
                **vargs)

    domain = np.sort(domain)

    mindistance = 0.9 * max(min([domain[i] - domain[i - 1] for i in range(1, len(domain))]), 1)

    plt.xlim((min(domain) - mindistance - width / 2, max(domain) + mindistance +
              width / 2))


def domain(self, values):
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
    return self.with_column('Domain', values)


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
    values = np.array(self.apply(pfunc, 'Domain')).astype(float)
    if any(values < 0):
        warnings.warn("Probability cannot be negative")
    return self.with_column('Probability', values).sort("Domain")


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
    return self.with_column('Probability', values).sort("Domain")


def _probability(self, values):
    self['Probability'] = values


def normalize(self):
    """
    Normalizes the distribution by making the proabilities sum to 1

    Returns
    -------
    FiniteDistribution
        Normalized FiniteDistribution
    """
    if 'Probability' not in self.labels:
        self._probability(np.ones(self.num_rows) / self.num_rows)
    else:
        self['Probability'] /= sum(self['Probability'])
    return self


def as_html(self, max_rows=0):
    # self.normalize()
    return super().as_html(max_rows)


def expected_value(self):
    """
    Finds expected value of distribution

    Returns
    -------
    float
        Expected value

    """
    self.normalize()
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

Table.chart_colors = chart_colors
Table.prob_event = prob_event
Table.event = event
Table.plot_dist = plot_dist
Table.plot_event = plot_event
Table.domain = domain
Table.probability = probability
Table.probability_function = probability_function
Table._probability = _probability
Table.normalize = normalize
Table.expected_value = expected_value
Table.variance = variance
Table.sd = sd
