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


class DiscreteDistribution(Table):
    """
    Subclass of Table to represent discrete distributions as a 2-column table of
    Domain and Probability.

    For constructing a Distribution, see documentation for FiniteDistribution
    and InfiniteDistribution.
    """

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
            return FiniteDistribution().domain(x).probability(probabilities)
        else:
            return FiniteDistribution().domain([x]).probability([self.prob_event(x)])

    def plot(self, width=1, mask=[], **vargs):
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
                    plt.bar(domain[mask[i]], prob[mask[i]], align="center",
                            color=colors[i], width=1, alpha=0.7)
            else:
                plt.bar(domain[mask], prob[mask], align="center", color="darkblue", width=1, alpha=0.7)
                plt.bar(domain[np.logical_not(mask)], prob[np.logical_not(mask)],
                        align="center", color="gold", width=1, alpha=0.7)
                # dist1 = FiniteDistribution().domain(domain[mask]).probability(prob[mask])
                # dist2 = FiniteDistribution().domain(domain[np.logical_not(mask)]).probability(prob[np.logical_not(mask)])
                # DiscreteDistribution.Plot("1", dist1, "2", dist2, width=width, **vargs)

        mindistance = 0.9 * min(
            [self['Domain'][i] - self['Domain'][i - 1] for i in range(1, self.num_rows)])

        plt.xlim((min(self['Domain']) - mindistance - width/2, max(self['Domain'])
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
        if len(event) == 0:
            self.plot(width=width, **vargs)

        else:

            mindistance = 0.9 * min(
            [self['Domain'][i] - self['Domain'][i - 1] for i in range(1, self.num_rows)])

            plt.xlim((min(self['Domain']) - mindistance - width/2, max(self['Domain'])
                  + mindistance + width / 2))

            domain = set(self["Domain"])

            def prob(x):
                return np.array([self.prob_event(a) for a in list(x)])


            if isinstance(event[0], collections.Iterable):
                colors = list(
                    itertools.islice(itertools.cycle(self.chart_colors), len(event) + 1))
                for i in range(len(event)):
                    plt.bar(event[i], prob(event[i]), align="center", color=colors[i], width=1, alpha=0.7)
                    domain -= set(event[i])

                domain = np.array(list(domain))
                plt.bar(domain, prob(domain), align="center", color=colors[-1], width=1, alpha=0.7)
            else:

                plt.bar(event, prob(event), align="center", width=1, color="gold", alpha=0.7)
                domain = np.array(list(set(self["Domain"]) - set(event)))
                plt.bar(domain, prob(domain), align="center", color="darkblue", width=1, alpha=0.7)


    @classmethod
    def Plot(cls, *labels_and_dists, width=1, **vargs):
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
        options = cls.default_options.copy()
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

        result.chart_colors = DiscreteDistribution.chart_colors

        start = min(domain)
        end = max(domain)
        end = (end // width + 1) * width
        result.hist(counts="Domain",
                    bins=np.arange(start - width / 2, end + width, width),
                    **vargs)

        domain = np.sort(domain)

        mindistance = 0.9 * min(
            [domain[i] - domain[i - 1] for i in range(1, len(domain))])

        plt.xlim((min(domain) - mindistance - width / 2, max(domain) + mindistance +
                  width / 2))


plot = DiscreteDistribution.Plot


class FiniteDistribution(DiscreteDistribution):
    """
    Subclass of DiscreteDistribution to represent a Finite Probability
    Distribution.

    Construct a FiniteDistribution by specifying both the range of the domain
    and the associated probabilities

    Examples
    --------
    >>> FiniteDistribution().domain(make_array(2, 3, 4)).probability(make_array(0.25, 0.5, 0.25))
    Domain | Probability
    2      | 0.25
    3      | 0.5
    4      | 0.25

    The `domain` method takes in an iterable. To associate each value with a
    probability, the `probability` method takes an iterable of the same length

    >>> FiniteDistribution().domain(np.arange(1,11)).probability_function(lambda x:1/10)
    Domain | Probability
    1      | 0.1
    2      | 0.1
    3      | 0.1
    4      | 0.1
    5      | 0.1
    6      | 0.1
    7      | 0.1
    8      | 0.1
    9      | 0.1
    10     | 0.1

    `FiniteDistribution` can also use a function on the domain to assign
    probabilities
    """

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


class InfiniteDistribution(DiscreteDistribution):
    """
    Subclass of DiscreteDistribution to represent an Infinite Probability
    Distribution.

    Construct an `InfiniteDistribution` by specifying both the range of the domain
    and a function for the probabilities

    Examples
    --------

    >>> geometric = InfiniteDistribution().domain(0,inf).probability_function(lambda k:0.4*(0.6)**k)
    >>> geometric
    Domain           | Probability
    0                | 0.4
    1                | 0.24
    2                | 0.144
    3                | 0.0864
    4                | 0.05184
    5                | 0.031104
    6                | 0.0186624
    7                | 0.0111974
    8                | 0.00671846
    9                | 0.00403108
    ... (Infinite rows omitted)

    Use `inf` to specify infinity

    """
    def domain(self, start, end, step=1):
        """
        Assigns domain values to InfiniteDistribution

        Parameters
        ----------
        start : float
            Smallest value in domain
        end : float
            Largest value in domain. Use `inf` for infinity
        step (optional) : float
            Step size in domain (default: 1)

        Returns
        -------
        InfiniteDistribution
            InfiniteDistribution with that domain
        """
        return self.with_column('Domain', [(start, end, step)])

    def probability_function(self, pfunc):
        """
        Assigns probabilities to an InfiniteDistribution via a probability
        function. The probability function is applied to each value of the
        domain

        Parameters
        ----------
        pfunc : univariate function
            Probability function of the InfiniteDistribution

        Returns
        -------
        InfiniteDistribution
            InfiniteDistribution with those probabilities

        """
        return self.with_column('Probability', [pfunc])

    def plot(self, width=1, size=20, **vargs):
        pfunc = self._pfunc
        start = self._start
        step = self._step

        domain = np.arange(start, start + size * step, step)
        probability = np.vectorize(pfunc, otypes=[np.float])(domain)

        FiniteDistribution().domain(domain).probability(probability).plot(
            width=width, **vargs)

    def prob_event(self, x):
        if isinstance(x, collections.Iterable):
            return sum(self.p_event(k) for k in x)
        else:
            # Doesn't check if x is in domain!
            return self._pfunc(x)

    def expected_value(self, delta=1e-25):
        """
        Approximates the expected value of an infinite distribution

        Parameters
        ----------
        delta (optional): float
            Accepted error margin for expected value (default: 1e-25)

        Returns
        -------
        float
            Expected Value
        """
        pfunc = self._pfunc
        start = self._start
        step = self._step

        ev = 0
        diff = pfunc(start)
        while (diff > delta):
            ev += start * diff
            start += step
            diff = pfunc(start)
        return ev

    def variance(self, delta=1e-25):
        """
        Approximates the variance of an InfiniteDistribution

        Parameters
        ----------
        delta (optional): float
            Accepted error margin of variance approximation (default: 1e-25)

        Returns
        -------
        float
            Variance

        """
        pfunc = self._pfunc
        start = self._start
        step = self._step

        ev = self.expected_value()
        var = 0
        diff = pfunc(start)
        while (diff > delta):
            var += (ev - start) ** 2 * diff
            start += step
            diff = pfunc(start)
        return var

    def sd(self):
        """
        Finds the Standard Deviation

        Returns
        -------
        Float
            Standard Deviation
        """
        return math.sqrt(self.variance())

    @property
    def _start(self):
        domain = self._columns["Domain"][0]
        return domain[0]

    @property
    def _step(self):
        domain = self._columns["Domain"][0]
        return domain[2]

    @property
    def _pfunc(self):
        return self._columns["Probability"][0]

    def as_html(self, max_rows=0):
        """Format table as HTML."""
        omitted = "Infinite"
        labels = self.labels
        lines = [
            (0, '<table border="1" class="dataframe">'),
            (1, '<thead>'),
            (2, '<tr>'),
            (3, ' '.join('<th>' + label + '</th>' for label in labels)),
            (2, '</tr>'),
            (1, '</thead>'),
            (1, '<tbody>'),
        ]
        fmts = [self._formats.get(k, self.formatter.format_column(k, v[:max_rows])) for
                k, v in self._columns.items()]
        fmts = [(lambda f: lambda v, label=False: v.as_html() if hasattr(v,
                                                                         'as_html') else f(
            v))(f) for f in fmts]
        for row in itertools.islice(self.rows, max_rows):
            lines += [
                (2, '<tr>'),
                (3, ' '.join('<td>' + fmt(v, label=False) + '</td>' for
                             v, fmt in zip(row, fmts))),
                (2, '</tr>'),
                (1, '</tbody>'),
            ]
        lines.append((0, '</table>'))
        if omitted:
            lines.append((0, '<p>... ({} rows omitted)</p'.format(omitted)))
        return '\n'.join(4 * indent * ' ' + text for indent, text in lines)

    def as_text(self, max_rows=0, sep=" | "):
        """Format table as text."""
        omitted = "Infinite"
        labels = self._columns.keys()
        fmts = [self._formats.get(k, self.formatter.format_column(k, v[:max_rows])) for
                k, v in self._columns.items()]
        rows = [[fmt(label, label=True) for fmt, label in zip(fmts, labels)]]
        for row in itertools.islice(self.rows, max_rows):
            rows.append([f(v, label=False) for v, f in zip(row, fmts)])
        lines = [sep.join(row) for row in rows]
        if omitted:
            lines.append('... ({} rows omitted)'.format(omitted))
        return '\n'.join([line.rstrip() for line in lines])

    class Rows(collections.abc.Sequence):
        """An iterable view over the rows in a table."""

        def __init__(self, table):
            self._table = table
            self._labels = None

        def __getitem__(self, i):
            labels = tuple(self._table.labels)
            if labels != self._labels:
                self._labels = labels
                self._row = type('Row', (Table.Row,), dict(_table=self._table))
            domain = self._table._columns["Domain"][0]
            pfunc = self._table._columns["Probability"][0]
            value = domain[0] + i * domain[2]
            return self._row((value, pfunc(value)))

        def __len__(self):
            return self._table.num_rows

        def __repr__(self):
            return '{0}({1})'.format(type(self).__name__, repr(self._table))


import itertools as it


class JointDistribution(FiniteDistribution):
    def domain(self, **kwargs):
        self.variables = sorted(list(kwargs.keys()))
        tablesetup = sum([[v, []] for v in self.variables], [])
        domains = [kwargs[v] for v in self.variables]
        overall = list(it.product(*domains))
        return self.with_columns(*tablesetup).with_rows(overall)

    def copy(self):
        v = super().copy()
        v.variables = self.variables
        return v

    def probability_function(self, pfunc):
        def function_in_between(row):
            d = {var: row[num] for num, var in enumerate(self.variables)}
            return float(pfunc(**d))

        return self.with_column("Probability", self.apply(function_in_between))

    def expected_value(self, variable):
        if variable not in self.variables:
            return None
        return np.sum(self.column(variable) * self.column("Probability"))

    def sd(self):
        pass

    def plot(self, **kwargs):
        return None

    def marginalize(self, variable):
        pass

    def prob_event(self, **kwargs):
        current = d
        for name, item in kwargs.items():
            current = current.where(name, item)
        return sum(current['Probability'])
