import collections
import itertools
import math
import warnings

from datascience import (
    are,
    make_array,
    Table,
)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from .multi_variable import multi_domain

inf = math.inf
rgb = matplotlib.colors.colorConverter.to_rgb


def check_valid_probability_table(table):
    assert table.num_columns == 2, 'In order to run a Prob140 function, your '\
        'table must have 2 columns: a Values column and a Probability column'
    assert all(table.column(1) >= 0), 'Probabilities must be non-negative'


def _bin(dist, width=1, start=None, num_bins=-1):
    """
    Helper function that bins a distribution for plotting

    Parameters
    ----------
    dist : Table
        Distribution that needs to be binned.
    width (optional) : float
        Width of each bin. (Default: 1)
    start (optional) : float
        Where to start first bin. (Default: minimum value of the domain.)
    num_bins (optional): int
        Number of bins. (Default: minimum number of bins to span domain)

    Returns
    -------
    (new_domain, new_prob)
        Domain values of the new bins and the associated probabilities

    Examples
    --------
    >>> x = Table().values([0, 0.5, 1]).probability([1 / 3, 1 / 3, 1 / 3])
    >>> _bin(x)
    (array([ 0.,  1.]), array([ 0.66666667,  0.33333333]))
    >>> _bin(x, width=0.5)
    (array([ 0. ,  0.5,  1. ]), array([ 0.33333333,  0.33333333,  0.33333333]))
    """

    domain = dist.column(0)
    prob = dist.column(1)

    if start is None:
        start = min(domain)
    if num_bins < 0:
        num_bins = max(num_bins, math.ceil((max(domain) - start) / width) + 1)

    new_domain = np.arange(start, start + width * num_bins, width)[:num_bins]
    new_prob = np.zeros(num_bins)

    for i in range(len(domain)):
        index = math.ceil((domain[i] - start - width / 2) / width)
        new_prob[index] += prob[i]
    return new_domain, new_prob


def Plot(dist, width=1, event=(), edges=None, show_ev=False, show_ave=False,
         show_sd=False, **vargs):
    """
    Plots the histogram for a single distribution.

    Parameters
    ----------
    dist : Table
        A 2-column table representing a probability distribution.
    width (optional) : float
        Width of the intervals (default: 1)
    edges (optional) : boolean
        If True, there will be a small border around the bars. If False, there
        will be no border. (default: small border unless there more than 75
        bins)
    show_ev (optional) : boolean
        Adds a tick mark at the expected value. (default : False)
    show_ave (optional) : boolean
        Adds a tick mark at the average of an empirical distribution.
        (default : False)
    show_sd (optional) : boolean
        Adds two tick marks one sd above and one sd below the expected value.
        (default : False)
    vargs
        See pyplot's additional optional arguments.
    """
    # Basic sanity checks.
    check_valid_probability_table(dist)
    dist = dist.remove_zeros()

    domain_label = dist.labels[0]
    dist = dist.sort(domain_label)
    domain, prob = _bin(dist, width)

    # Default plot attributes.
    options = {
        'align': 'center',
        'alpha': 0.7,
        'edgecolor': 'w',
        'lw': 0,
        'width': width,
    }

    # Set edges.
    if edges or len(domain) < 75:
        options['lw'] = 0.5
    if not edges and edges is not None:  # edges could be none
        options['lw'] = 0
    options.update(vargs)

    if len(event) != 0:
        # Events.

        domain = set(dist.column(0))
        def prob(x):
            return np.array([dist.prob_event(a) for a in list(x)])

        if isinstance(event[0], collections.Iterable):
            # If event is a list of lists.
            colors = list(itertools.islice(itertools.cycle(dist.chart_colors),
                                           len(event) + 1))
            for i in range(len(event)):
                # Cycle through each event and remove from the events set.
                plt.bar(event[i], prob(event[i]),
                        color=colors[i], **options)
                domain -= set(event[i])

            domain = np.array(list(domain))
            plt.bar(domain, prob(domain), color=colors[-1], **options)
        else:
            # If event is just a list.
            plt.bar(event, prob(event), color='gold', **options)
            domain = np.array(list(set(dist.column(0)) - set(event)))
            plt.bar(domain, prob(domain), color='darkblue', **options)

    else:
        # No event.
        plt.bar(domain, prob, color='darkblue', **options)

    plt.xlabel(domain_label)
    plt.ylabel('Percent per unit')

    # Multiple all y-ticks by 100 to get percent per unit.
    ax = plt.gca()
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 100.))
    ax.yaxis.set_major_formatter(ticks)

    # The minimum distance between any two values.
    min_distance = 0.9 * min([dist.column(0)[i] - dist.column(0)[i - 1]
                             for i in range(1, dist.num_rows)])

    plt.xlim((min(dist.column(0)) - min_distance - width / 2,
              max(dist.column(0)) + min_distance + width / 2))
    plt.ylim(0, max(dist.column(1)) * 1.1)

    # Markers.
    if show_ev or show_ave:
        plt.text(dist.ev(), 0, '^', horizontalalignment='center',
                 verticalalignment='top', size=30, color='red')
    if show_sd:
        plt.text(dist.ev() - dist.sd(), 0, '^',
                 horizontalalignment='center', verticalalignment='top',
                 size=30, color='blue')
        plt.text(dist.ev() + dist.sd(), 0, '^',
                 horizontalalignment='center', verticalalignment='top',
                 size=30, color='blue')


def Plots(*labels_and_dists, width=1, edges=None, **vargs):
    """
    Overlays histograms for multiple probability distributions together.

    Parameters
    ----------
    labels_and_dists : Even number of alternations between Strings and Tables
        Each distribution must have a label associated with it.
    width (optional) : float
        Width of the intervals. (default: 1)
    edges : bool
        If True, there will be a small border around the bars. If False, there
        will be no border. (default: small border unless there more than 75
        bins)
    vargs
        See pyplot's documentation

    Examples
    --------
    >>> dist1 = Table().values([1, 2, 3, 4]).probability([1/4, 1/4, 1/4, 1/4])
    >>> dist2 = Table().values([3, 4, 5, 6]).probability([1/2, 1/8, 1/8, 1/4])
    >>> Plots('Distribution1', dist1, 'Distribution2', dist2)
    <histogram with dist1 and dist2>
    """
    assert len(labels_and_dists) % 2 == 0, 'Even length sequence required'

    # Find the union of the domain of all plotted distributions.
    domain = set()
    distributions = []
    for i in np.arange(1, len(labels_and_dists) + 1, 2):
        dist = labels_and_dists[i]
        assert isinstance(dist, Table), \
            'Argument {} must be a distribution'.format(i)
        check_valid_probability_table(dist)
        distributions.append(dist.remove_zeros())
        domain = domain.union(distributions[-1].column(0))
    domain = np.sort(list(domain))

    # Find the and bin the probabilities corresponding to each value in domain
    # for each distribution.
    probabilities = []
    start = min(domain)
    num_bins = math.ceil((max(domain) - start) / width) + 1
    for dist in distributions:
        domain, prob = _bin(dist, width, start, num_bins)
        probabilities.append(np.array(prob))

    # Set the plot attributes.
    options = {
        'align': 'center',
        'alpha': 0.7,
        'edgecolor': 'w',
        'lw': 0,
        'width': width,
    }
    if edges or len(domain) < 75:
        options['lw'] = 0.5
    if not edges and edges is not None:  # edges could be none
        options['lw'] = 0
    options.update(vargs)

    n = len(distributions)
    colors = list(itertools.islice(itertools.cycle(Table.chart_colors), n))

    for i in range(n):
        plt.bar(domain, probabilities[i], color=colors[i],
                label=labels_and_dists[i * 2], **options)

    plt.legend(loc=2, bbox_to_anchor=(1.05, 1))

    min_distance = 0.9 * max(min([domain[i] - domain[i - 1]
                                 for i in range(1, len(domain))]), 1)

    plt.xlim((min(domain) - min_distance - width / 2,
              max(domain) + min_distance + width / 2))

    plt.xlabel('Value')
    plt.ylabel('Percent per unit')

    # Multiple all y-ticks by 100 to get percent per unit.
    ax = plt.gca()
    ticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x * 100.))
    ax.yaxis.set_major_formatter(ticks)


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

    >>> Table().values([1, 2, 3])
    Value
    1
    2
    3
    """
    table = self.with_column('Value', values)
    table.move_to_start('Value')
    return table


def probability_function(self, pfunc):
    """
    Assigns probabilities to a Distribution via a probability
    function. The probability function is applied to each value of the
    domain. Must have domain values in the first columns.

    Parameters
    ----------
    pfunc : func
        Probability function of the distribution.

    Returns
    -------
    Table
        Table with probabilities in its last column.

    """
    domain_names = self.labels
    values = np.array(self.apply(pfunc, *domain_names)).astype(float)
    if any(values < 0):
        warnings.warn('Probability cannot be negative')
    if round(sum(values), 6) != 1:
        warnings.warn('Probabilities sum to {0}'.format(sum(values)))
    return self.with_column('Probability', values)


def probability(self, values):
    """
    Assigns probabilities to domain values.

    Parameters
    ----------
    values : List or Array
        Values that must correspond to the domain in the same order.

    Returns
    -------
    Table
        A probability distribution with those probabilities
    """
    if any(np.array(values) < 0):
        warnings.warn('Probability cannot be negative')

    if round(sum(values), 6) != 1:
        warnings.warn('Probabilities sum to {0}'.format(sum(values)))
    return self.with_column('Probability', values)


def transition_function(self, pfunc):
    """
    Assigns transition probabilities to a Distribution via a probability
    function. The probability function is applied to each value of the
    domain. Must have domain values in the first column first.

    Parameters
    ----------
    pfunc : variate function
        Conditional probability function of the distribution ( P(Y | X))

    Returns
    -------
    Table
        Table with those probabilities in its final column

    """
    states = self.column(0)

    self = multi_domain(Table(), 'Source', states, 'Target', states)

    domain_names = self.labels
    values = np.array(self.apply(pfunc, *domain_names)).astype(float)
    if any(values < 0):
        warnings.warn('Probability cannot be negative')
    conditioned_var = self.labels[0]
    all_other_vars = ','.join(self.labels[1:])
    return_table = self.with_column('P({} | {})'.format(
        all_other_vars, conditioned_var), values)
    _transition_warn(return_table)
    return return_table


def transition_probability(self, values):
    """
    For a multivariate probability distribution, assigns transition
    probabilities, ie P(Y | X).

    Parameters
    ----------
    values : List or Array
        Values that must correspond to the domain in the same order

    Returns
    -------
    Table
        A probability distribution with those probabilities
    """
    if any(np.array(values) < 0):
        warnings.warn('Probability cannot be negative')

    states = self.column(0)

    self = multi_domain(Table(), 'Source', states, 'Target', states)

    return_table = self.with_column('Probability', values)
    _transition_warn(return_table)
    return return_table


def _transition_warn(table):
    prob_sums = table.group(0, collect=sum)
    for row in prob_sums.rows:
        if round(row[-1], 6) != 1:
            warnings.warn(
                'Transition probabilities for {} sum to {:.04f} not 1'.format(
                    row[0], row[-1])
            )


def prob_event(self, x):
    """
    Finds the probability of an event x.

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

    >>> dist = Table().values([1, 2, 3, 4]).probability([1/4, 1/4, 1/4, 1/4])
    >>> dist.prob_event(2)
    0.25
    >>> dist.prob_event([2, 3])
    0.5
    >>> dist.prob_event(np.arange(1, 5))
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
    >>> dist = Table().values([1 2, 3, 4]).probability([1/4, 1/4, 1/4, 1/4])
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
    return Table().with_columns('Outcome', x, 'Probability', probabilities)


def normalized(self):
    """
    Returns the distribution by making the proabilities sum to 1

    Returns
    -------
    Table
        A distribution with the probabilities normalized

    Examples
    --------
    >>> Table().values([1, 2, 3]).probability([1, 1, 1])
    Value | Probability
    1     | 1
    2     | 1
    3     | 1
    >>> Table().values([1, 2, 3]).probability([1, 1, 1]).normalized()
    Value | Probability
    1     | 0.333333
    2     | 0.333333
    3     | 0.333333
    """
    column_label = self.labels[-1]
    return self.with_column(
        column_label,
        self.column(column_label) / sum(self.column(column_label))
    )


def sample_from_dist(self, n=1):
    """
    Randomly samples from the distribution.

    Note that this function was previously named `sample` but was renamed
    due to naming conflicts with the datascience library.

    Parameters
    ----------
    n : int
        Number of times to sample from the distribution (default: 1)

    Returns
    -------
    float or array
        Samples from the distribution

    Examples
    --------
    >>> dist = Table().with_columns(
    ...    'Value', make_array(2, 3, 4),
    ...    'Probability', make_array(0.25, 0.5, 0.25))
    >>> dist.sample_from_dist()
    3
    >>> dist.sample_from_dist()
    2
    >>> dist.sample_from_dist(10)
    array([3, 2, 2, 4, 3, 4, 3, 4, 3, 3])
    """

    check_valid_probability_table(self)

    domain = self.column(0)
    prob = self.column(1)

    if n == 1:
        return np.random.choice(domain, p=prob)

    return np.random.choice(domain, n, p=prob)


def cdf(self, x):
    """
    Finds the cdf of the distribution

    Parameters
    ----------
    x : float
        Value in distribution

    Returns
    -------
    float
        Finds P(X<=x)

    Examples
    --------
    >>> dist = Table().with_columns(
    ...     'Value', make_array(2, 3, 4),
    ...     'Probability', make_array(0.25, 0.5, 0.25))
    >>> dist.cdf(0)
    0
    >>> dist.cdf(2)
    0.25
    >>> dist.cdf(3.5)
    0.75
    >>> dist.cdf(1000)
    1.0
    """
    check_valid_probability_table(self)
    dist = self.sort(0)
    domain = dist.column(0)
    prob = dist.column(1)
    indices = np.where(domain <= x)
    return sum(prob[indices])


def ev(self):
    """
    Finds expected value of distribution

    Returns
    -------
    float
        Expected value

    Examples
    --------
    >>> dist = Table().values([1, 2, 4]).probability([0.5, 0.4, 0.1])
    >>> dist.ev()
    1.7
    >>> 1 * 0.5 + 2 * 0.4 + 4 * 0.1
    1.7
    """
    check_valid_probability_table(self)
    self = normalized(self)
    ev = 0
    for domain, probability in self.rows:
        ev += domain * probability
    return ev


def var(self):
    """
    Finds variance of distribution

    Returns
    -------
    float
        Variance

    Examples
    --------
    >>> dist = Table().values([1, 2, 4]).probability([0.5, 0.4, 0.1])
    >>> dist.var()
    0.81
    >>> (1 * 0.5 + 4 * 0.4 + 16 * 0.1) - (1.7) ** 2
    0.81
    """
    check_valid_probability_table(self)
    self = normalized(self)
    var = 0
    ev = self.ev()
    for domain, probability in self.rows:
        var += (domain - ev) ** 2 * probability
    return var


def sd(self):
    """
    Finds standard deviation of Distribution.

    Returns
    -------
    float
        Standard Deviation

    Examples
    --------
    >>> dist = Table().values([1, 2, 4]).probability([0.5, 0.4, 0.1])
    >>> dist.sd()
    0.9
    """
    return math.sqrt(self.var())


def remove_zeros(self):
    """
    Removes all values with zero probability from the Distribution.

    Returns
    -------
    Distribution

    Examples
    --------
    >>> dist = Table().values([2, 3, 4, 5]).probability([0.5, 0, 0.5, 0])
    >>> dist
    Value | Probability
    2     | 0.5
    3     | 0
    4     | 0.5
    5     | 0
    >>> dist.remove_zeros()
    Value | Probability
    2     | 0.5
    4     | 0.5
    """
    check_valid_probability_table(self)
    return self.where(1, are.above(0))


def emp_dist(values):
    """
    Takes an array of values and returns an empirical distribution

    Parameters
    ----------
    values : array
        Array of values that will be grouped by the distribution

    Returns
    -------
    Table
        A distribution

    Examples
    --------
    >>> x = make_array(1, 1, 1, 1, 1, 2, 3, 3, 3, 4)
    >>> emp_dist(x)
    Value | Proportion
    1     | 0.5
    2     | 0.1
    3     | 0.3
    4     | 0.1
    """

    total = len(values)

    position_counts = Table().with_column('position', values).group(0)
    new_dist = Table().values(position_counts.column(0))
    return new_dist.with_column(
        'Proportion',
        position_counts.column(1) / total
    )


# Brighter colors than the default Table class
chart_colors = (
    rgb('darkblue'),
    rgb('gold'),
    (106 / 256, 166 / 256, 53 / 256),  # vivid green
    (234 / 256, 77 / 256, 108 / 256),  # rose garden
    rgb('brown'),
    (240 / 256, 127 / 256, 80 / 256),  # vivid orange
    (53 / 256, 148 / 256, 216 / 256),  # vivid blue
    (122 / 256, 55 / 256, 139 / 256),  # purple
    rgb('black'),
    rgb('red'),
)
