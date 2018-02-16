from collections import OrderedDict
import warnings

from datascience import Table
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


class MarkovChain:
    """
    A class for representing, simulating, and computing Markov Chains.
    """

    def __init__(self, states, transition_matrix):
        transition_matrix = np.array(transition_matrix)
        if not np.all(transition_matrix >= 0):
            warnings.warn('Transition matrix contains negative value(s).')
        if not np.all(np.isclose(np.sum(transition_matrix, axis=1), 1.)):
            warnings.warn('Transition probabilities don\'t sum to 1.')
        self.states = states
        self.matrix = transition_matrix

    def to_pandas(self):
        """
        Returns the Pandas DataFrame representation of the MarkovChain.
        """
        return pd.DataFrame(
            data=self.matrix,
            index=self.states,
            columns=self.states
        )

    def get_transition_matrix(self, steps=1):
        """
        Returns the transition matrix after n steps as a numpy matrix.

        Parameters
        ----------
        steps : int (optional)
            Number of steps. (default: 1)

        Returns
        -------
        Transition matrix
        """
        return np.linalg.matrix_power(self.matrix, steps)

    def transition_matrix(self, steps=1):
        """
        Returns the transition matrix after n steps visually as a Pandas df.

        Parameters
        ----------
        steps : int (optional)
            Number of steps. (default: 1)

        Returns
        -------
        Pandas DataFrame
        """
        return pd.DataFrame(
            data=self.get_transition_matrix(steps),
            index=self.states,
            columns=self.states
        )

    def distribution(self, starting_condition, steps=1):
        """
        Finds the distribution of states after n steps given a starting
        condition.

        Parameters
        ----------
        starting_condition : state or Table
            The initial distribution or the original state.
        n : integer
            Number of transition steps.

        Returns
        -------
        Table
            Shows the distribution after n steps

        Examples
        --------
        >>> states = make_array('A', 'B')
        >>> transition_matrix = np.array([[0.1, 0.9],
        ...                               [0.8, 0.2]])
        >>> mc = MarkovChain.from_matrix(states, transition_matrix)
        >>> mc.distribution(start)
        State | Probability
        A     | 0.24
        B     | 0.76
        >>> mc.distribution(start, 0)
        State | Probability
        A     | 0.8
        B     | 0.2
        >>> mc.distribution(start, 3)
        State | Probability
        A     | 0.3576
        B     | 0.6424
        """
        if isinstance(starting_condition, Table):
            states = list(starting_condition.column(0))

            # datascience Tables store everything in arrays, so iterables get
            # typecast to arrays. Thus, if the states are iterables, we need to
            # typecast it back to its original type.
            if hasattr(states[0], '__iter__'):
                desired_type = type(self.states[0])
                states = list(map(desired_type, states))
            probabilities = starting_condition.column(1)
        else:
            states = [starting_condition]
            probabilities = [1]

        n = len(self.states)
        start = np.zeros((n, 1))
        for i in range(n):
            if self.states[i] in states:
                index = states.index(self.states[i])
                start[i, 0] = probabilities[index]
            else:
                start[i, 0] = 0

        probabilities = start.T.dot(self.get_transition_matrix(steps=steps))
        return Table().states(self.states).probability(probabilities[0])

    def log_prob_of_path(self, starting_condition, path):
        """
        Finds the log-probability of a path given a starting condition.

        May have better precision than `prob_of_path`.

        Parameters
        ----------
        starting_condition : state or Distribution
            If a state, finds the log-probability of the path starting at that
            state. If a Distribution, finds the probability of the path with
            the first element sampled from the Distribution
        path : ndarray
            Array of states

        Returns
        -------
        float
            log of probability

        Examples
        --------
        >>> states = make_array('A', 'B')
        >>> transition_matrix = np.array([[0.1, 0.9],
        ...                               [0.8, 0.2]])
        >>> mc = MarkovChain.from_matrix(states, transition_matrix)
        >>> mc.log_prob_of_path('A', ['A', 'B', 'A'])
        -2.6310891599660815
        >>> start = Table().states(['A', 'B']).probability([0.8, 0.2])
        >>> mc.log_prob_of_path(start, ['A', 'B', 'A'])
        -0.55164761828624576
        """
        states = list(self.states)
        if isinstance(starting_condition, Table):
            first = path[0]
            index = list(starting_condition.column(0)).index(first)
            assert index != -1, 'First path value not found.'
            log_prob = np.log(starting_condition.column(1)[index])
            prev_index = states.index(first)
            i = 1
        else:
            log_prob = np.log(1)
            prev_index = states.index(starting_condition)
            i = 0

        while i < len(path):
            curr_index = states.index(path[i])
            log_prob += np.log(self.matrix[prev_index, curr_index])
            prev_index = curr_index
            i += 1
        return log_prob

    def prob_of_path(self, starting_condition, path):
        """
        Finds the probability of a path given a starting condition.

        Parameters
        ----------
        starting_condition : state or Distribution
            If a state, finds the probability of the path starting at that
            state. If a Distribution, finds the probability of the path with
            the first element sampled from the Distribution.
        path : ndarray
            Array of states

        Returns
        -------
        float
            probability

        Examples
        --------
        >>> states = ['A', 'B']
        >>> transition_matrix = np.array([[0.1, 0.9],
        ...                               [0.8, 0.2]])
        >>> mc = MarkovChain.from_matrix(states, transition_matrix)
        >>> mc.prob_of_path('A', ['A', 'B', 'A'])
        0.072
        >>> 0.1 * 0.9 * 0.8
        0.072
        >>> start = Table().states(['A', 'B']).probability([0.8, 0.2])
        >>> mc.prob_of_path(start, ['A', 'B', 'A'])
        0.576
        >>> 0.8 * 0.9 * 0.8
        0.576
        """
        states = list(self.states)
        if isinstance(starting_condition, Table):
            first = path[0]
            index = list(starting_condition.column(0)).index(first)
            assert index != -1, 'First path value not found.'
            prob = starting_condition.column(1)[index]
            prev_index = states.index(first)
            i = 1
        else:
            prob = 1
            prev_index = states.index(starting_condition)
            i = 0

        while i < len(path):
            curr_index = states.index(path[i])
            prob *= self.matrix[prev_index, curr_index]
            prev_index = curr_index
            i += 1
        return prob

    def simulate_path(self, starting_condition, steps, plot_path=False):
        """
        Simulates a path of n steps with a specific starting condition.

        Parameters
        ----------
        starting_condition : state or Distribution
            If a state, simulates n steps starting at that state. If a
            Distribution, samples from that distribution to find the starting
            state.
        steps : int
            Number of steps to take.
        plot_path : bool
            If True, plots the simulated path.

        Returns
        -------
        ndarray
            Array of sampled states.

        Examples
        --------
        >>> states = ['A', 'B']
        >>> transition_matrix = np.array([[0.1, 0.9],
        ...                               [0.8, 0.2]])
        >>> mc = MarkovChain.from_matrix(states, transition_matrix)
        >>> mc.simulate_path('A', 10)
        array(['A', 'A', 'B', 'A', 'B', 'A', 'B', 'B', 'A', 'B', 'B'])
        """
        states = list(self.states)
        if isinstance(starting_condition, Table):
            start = starting_condition.sample_from_dist()
        else:
            start = starting_condition

        path = [start]
        for i in range(steps):
            index = states.index(path[-1])
            next_state = np.random.choice(states, p=self.matrix[index])
            path.append(next_state)

        if plot_path:
            self.plot_path(path[0], path[1:])
        else:
            return np.array(path)

    def steady_state(self):
        """
        Finds the stationary distribution of the Markov Chain.

        Returns
        -------
        Table
            Distribution.

        Examples
        --------
        >>> states = ['A', 'B']
        >>> transition_matrix = np.array([[0.1, 0.9],
        ...                               [0.8, 0.2]])
        >>> mc = MarkovChain.from_matrix(states, transition_matrix)
        >>> mc.steady_state()
        Value | Probability
        A     | 0.666667
        B     | 0.333333
        """
        # Steady state is the left eigenvector that corresponds to eigenvalue=1.
        w, vl = scipy.linalg.eig(self.matrix, left=True, right=False)

        # Find index of eigenvalue = 1.
        index = np.isclose(w, 1)

        eigenvector = np.real(vl[:, index])[:, 0]
        probabilities = eigenvector / sum(eigenvector)

        # Zero out floating poing errors that are negative.
        indices = np.logical_and(np.isclose(probabilities, 0),
                                 probabilities < 0)
        probabilities[indices] = 0
        return Table().values(self.states).probability(probabilities)

    def expected_return_time(self):
        """
        Finds the expected return time of the Markov Chain (1 / steady state).

        Returns
        -------
        Table
            Expected Return Time

        Examples
        --------
        >>> states = ['A', 'B']
        >>> transition_matrix = np.array([[0.1, 0.9],
        ...                               [0.8, 0.2]])
        >>> mc = MarkovChain.from_matrix(states, transition_matrix)
        >>> mc.expected_return_time()
        Value | Expected Return Time
        A     | 1.5
        B     | 3
        """
        steady = self.steady_state()
        expected_return = steady.column(1)
        return Table().values(self.states).with_column(
            'Expected Return Time',
            1 / expected_return
        )

    def plot_path(self, starting_condition, path):
        """
        Plots a Markov Chain's path.

        Parameters
        ----------
        starting_condition : state
            State to start at.
        path : iterable
            List of valid states.

        Examples
        --------
        >>> states = ['A', 'B']  # Works with all state data types!
        >>> transition_matrix = np.array([[0.1, 0.9],
        ...                               [0.8, 0.2]])
        >>> mc = MarkovChain.from_matrix(states, transition_matrix)
        >>> mc.plot_path(mc.simulate_path('B', 20))
        <Plot of a Markov Chain that starts at 'B' and takes 20 steps>
        """
        assert starting_condition in self.states, 'Start state must be a state.'

        states = list(self.states)
        prev_index = states.index(starting_condition)
        for state in path:
            curr_index = states.index(state)
            assert self.matrix[prev_index, curr_index] != 0, \
                'Path not possible.'
            prev_index = curr_index

        path = [starting_condition] + list(path)
        x = np.arange(len(path))
        y = [states.index(state) for state in path]
        plt.scatter(x, y, color='blue')
        plt.plot(x, y, lw=1, color='black')
        plt.yticks(np.arange(len(states)), states)
        plt.xlim(-0.5, len(path) + 0.5)
        plt.ylim(-0.5, len(states) - 0.5)
        plt.xlabel('Time')
        plt.ylabel('States')

    def _repr_html_(self):
        return self.to_pandas()._repr_html_()

    def __repr__(self):
        return self.to_pandas().__repr__()

    def __str__(self):
        return self.to_pandas().__str__()

    @classmethod
    def from_table(cls, table):
        """
        Constructs a Markov Chain from a Table

        Parameters
        ----------
        table : Table
            A  table with three columns for source state, target state, and
            probability.

        Returns
        -------
        MarkovChain

        Examples
        --------
        >>> table = Table().states(make_array('A', 'B')) \
        ...     .transition_probability(make_array(0.5, 0.5, 0.3, 0.7))
        >>> table
        Source | Target | Probability
        A      | A      | 0.5
        A      | B      | 0.5
        B      | A      | 0.3
        B      | B      | 0.7
        >>> MarkovChain.from_table(table)
             A    B
        A  0.5  0.5
        B  0.3  0.7
        """
        assert table.num_columns == 3, \
            'Must have 3 columns: source, target, probability'
        for prob_sum in table.group(0, collect=sum).column(2):
            assert round(prob_sum, 6) == 1, \
                   'Transition probabilities must sum to 1.'

        # Get a list of the states.
        ordered_set = OrderedDict()
        for row in table.rows:
            ordered_set[row[0]] = 0
        states = list(ordered_set.keys())

        n = len(states)
        transition_matrix = np.zeros((n, n))

        for row in table.rows:
            source = states.index(row[0])
            target = states.index(row[1])
            transition_matrix[source, target] = row[2]
        return cls(states, transition_matrix)

    @classmethod
    def from_transition_function(cls, states, transition_function):
        """
        Constructs a MarkovChain from a transition function.

        Parameters
        ----------
        states : iterable
            List of states.
        transition_function : function
            Bivariate transition function that maps two states to a
            probability.

        Returns
        -------
        MarkovChain

        Examples
        --------
        >>> states = make_array(1, 2)
        >>> def transition(s1, s2):
        ...    if s1 == s2:
        ...        return 0.7
        ...    else:
        ...        return 0.3
        >>> MarkovChain.from_transition_function(states, transition)
             1    2
        1  0.7  0.3
        2  0.3  0.7
        """
        n = len(states)
        transition_matrix = np.zeros((n, n))
        for i in range(n):
            for j in (range(n)):
                transition_matrix[i, j] = transition_function(states[i],
                                                              states[j])
        return cls(states, transition_matrix)

    @classmethod
    def from_matrix(cls, states, transition_matrix):
        """
        Constructs a MarkovChain from a transition matrix.

        Parameters
        ----------
        states : iterable
            List of states.
        transition_matrix : ndarray
            Square transition matrix.

        Returns
        -------
        MarkovChain

        Examples
        --------
        >>> states = [1, 2]
        >>> transition_matrix = np.array([[0.1, 0.9],
        ...                               [0.8, 0.2]])
        >>> MarkovChain.from_matrix(states, transition_matrix)
             1    2
        1  0.1  0.9
        2  0.8  0.2
        """
        return cls(states, transition_matrix)


def to_markov_chain(self):
    """
    Constructs a Markov Chain from the Table.

    Returns
    -------
    MarkovChain
    """
    return MarkovChain.from_table(self)
