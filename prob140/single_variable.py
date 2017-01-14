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

    >>> dist = Table().domain([1,2,3,4]).probability([1/4,1/4,1/4,1/4])
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
    Shows the probability that distribution takes on value x or list of
    values x.

    Parameters
    ----------
    x : float or Iterable
        An event represented either as a specific value in the domain or a
        subset of the domain

    Returns
    -------
    Table
        Shows the probabilities of each value in the event


    Examples
    --------
    >>> dist = Table().domain([1,2,3,4]).probability([1/4,1/4,1/4,1/4])
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



def _bin(dist, width=1, start=None):
    """
    Helper function that bins a distribution for plotting

    Parameters
    ----------
    dist : Table
        Distribution that needs to be binned
    width (optional) : float
        Width of each bin. Default is 1
    start (optional) : float
        Where to start first bin. Defaults to the minimum value of the domain

    Returns
    -------
    (new_domain, new_prob)
        Domain values of the new bins and the associated probabilities

    Examples
    --------
    >>> x = Table().domain([0, 0.5, 1]).probability([1/3, 1/3, 1/3])
    >>> _bin(x)
    (array([ 0.,  1.]), array([ 0.66666667,  0.33333333]))
    >>> _bin(x, width=0.5)
    (array([ 0. ,  0.5,  1. ]), array([ 0.33333333,  0.33333333,  0.33333333]))

    """
    domain = dist.column(0)
    prob = dist.column(1)

    if start == None:
        start = min(domain)

    num_bins = math.ceil((max(domain) - start) / width) + 1

    new_domain = np.arange(start, start + width * num_bins, width)

    new_prob = np.zeros(num_bins)

    for i in range(len(domain)):
        index = math.ceil((domain[i] - start - width/2) / width)
        new_prob[index] += prob[i]

    return new_domain, new_prob

def Plot(dist, width=1, mask=[], event=[], **vargs):
    """
    Plots the histogram for a single distribution

    Parameters
    ----------
    dist : Table
        A 2-column table representing a probability distribution
    width (optional) : float
        Width of the intervals (default: 1)
    mask (optional) : boolean array or list of boolean arrays
        Colors the parts of the histogram associated with each mask (
        default: no mask)
    vargs
        See pyplot's additional optional arguments

    """

    options = {"width" : width, "lw" : 0, "alpha" : 0.7, "align" : "center"}
    options.update(vargs)

    check_valid_probability_table(dist)

    domain_label = dist.labels[0]
    dist = dist.sort(domain_label)
    domain, prob = _bin(dist, width)

    if len(event) != 0:

        domain = set(dist.column(0))

        def prob(x):
            return np.array([dist.prob_event(a) for a in list(x)])

        if isinstance(event[0], collections.Iterable):
            # If event is a list of lists

            colors = list(itertools.islice(itertools.cycle(dist.chart_colors), len(event) + 1))
            for i in range(len(event)):
                plt.bar(event[i], prob(event[i]) * 100, color=colors[i], **options)
                domain -= set(event[i])

            domain = np.array(list(domain))
            plt.bar(domain, prob(domain) * 100, color=colors[-1], **options)

        else:
            # If event is just a list

            plt.bar(event, prob(event) * 100, color="gold", **options)
            domain = np.array(list(set(dist.column(0)) - set(event)))
            plt.bar(domain, prob(domain) * 100, color="darkblue", **options)


    elif len(mask) == 0:
        # no mask or event
        plt.bar(domain, prob, color="darkblue", **options)
        #dist.hist(counts=domain_label, bins=np.arange(start - width / 2, end + width, width), **vargs)
    else:
        if isinstance(mask[0], collections.Iterable):
            # If mask is a list of lists

            colors = list(itertools.islice(itertools.cycle(dist.chart_colors), len(mask)))
            for i in range(len(mask)):
                plt.bar(domain[mask[i]], prob[mask[i]] * 100, color=colors[i], **options)

        else:
            # If mask is just a list

            plt.bar(domain[mask], prob[mask] * 100, color="darkblue", **options)
            plt.bar(domain[np.logical_not(mask)], prob[np.logical_not(mask)] * 100, color="gold", **options)


    plt.xlabel(domain_label)
    plt.ylabel("Percent per unit")

    mindistance = 0.9 * max(min([dist.column(0)[i] - dist.column(0)[i - 1] for i in range(1, dist.num_rows)]),1)

    plt.xlim((min(dist.column(0)) - mindistance - width / 2, max(dist.column(0))
              + mindistance + width / 2))


def Plots(*labels_and_dists, width=1, **vargs):
    """
    Overlays histograms for multiply probability distributions together.

    Parameters
    ----------
    labels_and_dists : Even number of alternations between Strings and
    Tables
        Each distribution must have a label associated with it
    width (optional) : float
        Width of the intervals (default: 1)
    vargs
        See pyplot's documentation


    Examples
    --------
    >>> dist1 = Table().domain([1,2,3,4]).probability([1/4,1/4,1/4,1/4])
    >>> dist2 = Table().domain([3,4,5,6]).probability([1/2,1/8,1/8,1/4])
    >>> Plots("Distribution1", dist1, "Distribution2", dist2)
    <histogram with dist1 and dist2>

    """

    assert len(labels_and_dists) % 2 == 0, 'Even length sequence required'
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

    plt.xlabel("Value")


def single_domain(self, values):
    """
    Assigns domain values to a single-variable distribution

    Parameters
    ----------
    values : List or Array
        Values to put into the domain

    Returns
    -------
    Table
        Table with those domain values in its first column

    Examples
    --------

    >>> Table().domain([1,2,3])
    Value
    1
    2
    3
    """
    table = self.with_column('Value', values)
    table.move_to_start('Value')
    return table


def single_probability_function(self, pfunc):
    """
    Assigns probabilities to a Distribution via a probability
    function. The probability function is applied to each value of the
    domain. Must have domain values in the first column first.

    Parameters
    ----------
    pfunc : univariate function
        Probability function of the distribution

    Returns
    -------
    Table
        Table with those probabilities in its second column

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
    Table
        A proability distribution with those probabilities
    """
    if any(np.array(values) < 0):
        warnings.warn("Probability cannot be negative")
    return self.with_column('Probability', values)


def normalized(self):
    """
    Returns the distribution by making the proabilities sum to 1

    Returns
    -------
    Table
        A distribution with the probabilities normalized

    Examples
    --------
    >>> Table().domain([1,2,3]).probability([1,1,1])
    Value | Probability
    1     | 1
    2     | 1
    3     | 1
    >>> Table().domain([1,2,3]).probability([1,1,1]).normalized()
    Value | Probability
    1     | 0.333333
    2     | 0.333333
    3     | 0.333333
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
    Finds standard deviation of Distribution

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
    (106/256, 166/256, 53/256), # vivid green
    (234/256, 77/256, 108/256), # rose garden
    rgb("brown"),
    (240/256, 127/256, 80/256), # vivid orange
    (53/256, 148/256, 216/256), # vivid blue
    (122/256, 55/256, 139/256), # purple
    rgb("black"),
    rgb("red"),
)