import math
import numpy as np
import matplotlib.pyplot as plt
import collections
import itertools
import abc
import sys
import numbers
import warnings

import matplotlib

from datascience import *

inf = math.inf
rgb = matplotlib.colors.colorConverter.to_rgb


class DiscreteDistribution(Table):
    """Subclass of Table to represent distributions as a 2-column table of Domain and Probability"""

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

    def p(self, x):
        """
        Finds the probability that distribution takes on value x or range of values x. Returns sum.
        """
        if isinstance(x, collections.Iterable):
            return sum(self.p(k) for k in x)
        else:
            domain = self._columns["Domain"]
            prob = self._columns["Probability"]
            return sum(prob[np.where(domain == x)])

    def P(self, x):
        """
        Finds the probability that distribution takes on value x or range of values x. Returns table.
        """
        if isinstance(x, collections.Iterable):
            probabilities = [self.p(k) for k in x]
            return FiniteDistribution().domain(x).probability(probabilities)
        else:
            return FiniteDistribution().domain([x]).probability([self.p(x)])

    def plot(self, binWidth=1, mask=[], **vargs):

        domain = self["Domain"]
        prob = self["Probability"]

        start = min(domain)
        end = max(domain)

        end = (end // binWidth + 1) * binWidth

        if len(mask) == 0:
            self.hist(counts="Domain",
                      bins=np.arange(start - binWidth / 2, end + binWidth, binWidth),
                      **vargs)
        else:
            if isinstance(mask[0], collections.Iterable):
                colors = list(
                    itertools.islice(itertools.cycle(self.chart_colors), len(mask)))
                for i in range(len(mask)):
                    plt.bar(domain[mask[i]], prob[mask[i]], align="center",
                            color=colors[i])
            else:
                plt.bar(domain[mask], prob[mask], align="center", color="darkblue")
                plt.bar(domain[np.logical_not(mask)], prob[np.logical_not(mask)],
                        align="center", color="gold")
                # dist1 = FiniteDistribution().domain(domain[mask]).probability(prob[mask])
                # dist2 = FiniteDistribution().domain(domain[np.logical_not(mask)]).probability(prob[np.logical_not(mask)])
                # DiscreteDistribution.Plot("1", dist1, "2", dist2, binWidth=binWidth, **vargs)

        mindistance = 0.9 * min(
            [self['Domain'][i] - self['Domain'][i - 1] for i in range(1, self.num_rows)])

        plt.xlim((min(self['Domain']) - mindistance - binWidth/2, max(self['Domain'])
                  + mindistance + binWidth / 2))

    @classmethod
    def Plot(cls, *labels_and_dists, binWidth=1, **vargs):
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
            probability = np.vectorize(dist.p, otypes=[np.float])(domain)
            distributions.append(probability)
            i += 2

        result = Table().with_columns(*distributions)

        result.chart_colors = DiscreteDistribution.chart_colors

        start = min(domain)
        end = max(domain)
        end = (end // binWidth + 1) * binWidth
        result.hist(counts="Domain",
                    bins=np.arange(start - binWidth / 2, end + binWidth, binWidth),
                    **vargs)

        domain = np.sort(domain)

        mindistance = 0.9 * min(
            [domain[i] - domain[i - 1] for i in range(1, len(domain))])

        plt.xlim((min(domain) - mindistance - binWidth / 2, max(domain) + mindistance +
                  binWidth / 2))


plot = DiscreteDistribution.Plot


class FiniteDistribution(DiscreteDistribution):
    def domain(self, values):
        return self.with_column('Domain', values)

    def probability_function(self, pfunc):
        values = np.array(self.apply(pfunc, 'Domain')).astype(float)
        if any(values < 0):
            warnings.warn("Probability cannot be negative")
        return self.with_column('Probability', values)

    def probability(self, values):
        if any(np.array(values) < 0):
            warnings.warn("Probability cannot be negative")
        return self.with_column('Probability', values)

    def _probability(self, values):
        self['Probability'] = values

    def normalize(self):
        if 'Probability' not in self.labels:
            self._probability(np.ones(self.num_rows) / self.num_rows)
        else:
            self['Probability'] /= sum(self['Probability'])
        return self

    def as_html(self, max_rows=0):
        # self.normalize()
        return super().as_html(max_rows)

    def expected_value(self):
        self.normalize()
        ev = 0
        for domain, probability in self.rows:
            ev += domain * probability
        return ev

    def variance(self):
        self.normalize()
        var = 0
        ev = self.expected_value()
        for domain, probability in self.rows:
            var += (domain - ev) ** 2 * probability
        return var

    def sd(self):
        return math.sqrt(self.variance())


class InfiniteDistribution(DiscreteDistribution):
    def domain(self, start, end, step=1):
        return self.with_column('Domain', [(start, end, step)])

    def probability_function(self, pfunc):
        return self.with_column('Probability', [pfunc])

    def plot(self, binWidth=1, size=20, **vargs):
        pfunc = self._pfunc
        start = self._start
        step = self._step

        domain = np.arange(start, start + size * step, step)
        probability = np.vectorize(pfunc, otypes=[np.float])(domain)

        FiniteDistribution().domain(domain).probability(probability).plot(
            binWidth=binWidth, **vargs)

    def p(self, x):
        if isinstance(x, collections.Iterable):
            return sum(self.p(k) for k in x)
        else:
            # Doesn't check if x is in domain!
            return self._pfunc(x)

    def expected_value(self):
        pfunc = self._pfunc
        start = self._start
        step = self._step

        ev = 0
        diff = pfunc(start)
        while (diff > 1e-25):
            ev += start * diff
            start += step
            diff = pfunc(start)
        return ev

    def variance(self):
        pfunc = self._pfunc
        start = self._start
        step = self._step

        ev = self.expected_value()
        var = 0
        diff = pfunc(start)
        while (diff > 1e-25):
            var += (ev - start) ** 2 * diff
            start += step
            diff = pfunc(start)
        return var

    def sd(self):
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

    def p(self, **kwargs):
        current = d
        for name, item in kwargs.items():
            current = current.where(name, item)
        return sum(current['Probability'])
