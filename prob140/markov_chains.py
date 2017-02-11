import pandas as pd    
from . import pykov
from .single_variable import emp_dist
from datascience import *
import numpy as np

def matrix_to_pandas(matrix):
    all_states =sorted(matrix.states())
    target_states = ['S: '+(str(label)) for label in all_states]
    data = {source: [0]*len(all_states) for source in all_states}
    for (source,target),probability in matrix.items():
        data[target][all_states.index(source)] = probability
    data = {'T: '+str(label): values for label,values in data.items()}
    return pd.DataFrame(data,index=target_states)
    
def matrix_to_table(matrix):
    t = Table().with_columns('Source',[],'Target',[],'Probability',[])
    rows = [(source,target,probability) for (source,target),probability in matrix.items()]
    return t.with_rows(rows)

def table_to_vector(table):
    assert table.num_columns == 2, "You must have 2 columns for this task"
    label_column = table.column(0)
    prob_column = table.column(1)
    return pykov.Vector({label:prob for label,prob in zip(label_column,prob_column)})

def vector_to_table(vector,valueLabel='Probability'):
    t = Table().with_columns('State',[],valueLabel,[])
    rows = sorted(vector.items(),key=lambda x:x[0])
    return t.with_rows(rows)

def pykov_connection(function):
    def internal(*args,**kwargs):
        new_args = [(table_to_vector(argument) if isinstance(argument,Table) else argument) for argument in args]
        kwargs = {key:(table_to_vector(value) if isinstance(value,Table) else value) for key,value in kwargs}
        output = function(*new_args,**kwargs)
        if isinstance(output,pykov.Vector):
            return vector_to_table(output)
        if isinstance(output,pykov.Matrix):
            return matrix_to_pandas(output)
        return output
    return internal


class MarkovChain:
    def __init__(self,pykov_chain):
        self.chain = pykov_chain

    def __repr__(self):
        return matrix_to_pandas(self.chain).__repr__()

    def __str__(self):
        return matrix_to_pandas(self.chain).__str__()

    def _repr_html_(self):
        return matrix_to_pandas(self.chain)._repr_html_()
    
    @pykov_connection
    def move(self, state):
        """
        Transitions one step from the indicated state

        Parameters
        ----------
        state : String or float

        Returns
        -------
        String or float
            Next state


        Examples
        --------
        >>> mc = Table().states(make_array("A", "B")).transition_probability(make_array(0.5, 0.5, 0.3, 0.7)).toMarkovChain()
        >>> mc.move('A')
        'A'
        >>> mc.move('A')
        'B'
        >>> mc.move('B')
        'B'
        """
        return self.chain.move(state)
    
    @pykov_connection
    def distribution(self, starting_condition, n):
        """
        Finds the distribution of states after n steps given a starting condition

        Parameters
        ----------
        starting_condition : state or Distribution
            The initial distribution or the original state
        n : integer
            Number of transition steps

        Returns
        -------
        Table
            Shows the distribution after n steps

        Examples
        >>> mc = Table().states(make_array("A", "B")).transition_probability(make_array(0.5, 0.5, 0.3, 0.7)).toMarkovChain()
        >>> start = Table().state(make_array("A", "B")).probability(make_array(0.8, 0.2))
        >>> mc.distribution(start, 0)
        State | Probability
        A     | 0.8
        B     | 0.2
        >>> mc.distribution(start, 2)
        State | Probability
        State | Probability
        A     | 0.392
        B     | 0.608
        >>> mc.distribution(start, 10000)
        State | Probability
        A     | 0.375
        B     | 0.625
        """

        if not isinstance(starting_condition, pykov.Vector):
            starting_condition = pykov.Vector({starting_condition : 1})
        return self.chain.pow(starting_condition, n)
    
    @pykov_connection
    def steady_state(self):
        """
        The stationary distribution of the markov chain

        Returns
        -------
        Table
            steady state distribution

        >>> mc = Table().states(make_array("A", "B")).transition_probability(make_array(0.5, 0.5, 0.3, 0.7)).toMarkovChain()
        >>> mc.steady_state()
        State | Probability
        A     | 0.375
        B     | 0.625
        """
        return self.chain.steady()
    
    def mean_first_passage_times(self):
        """
        Finds the mean time it takes to reach state j from state i

        Returns
        -------
        DataFrame
            Mean first passage times from source to target

        Examples
        --------
        >>> mc = Table().states(make_array("A", "B")).transition_probability(make_array(0.5, 0.5, 0.3, 0.7)).toMarkovChain()
        >>> mc.mean_first_passage_times()
                  T: A  T: B
        S: A  2.666667   2.0
        S: B  3.333333   1.6
        """
        states = self.chain.states()
        my_dict = {}

        def find_steady(x):
            steady = self.steady_state()
            return steady.column(1)[np.where(steady.column(0) == x)[0]][0]

        for i in states:
            mfpt_to = self.chain.mfpt_to(i)
            for j in mfpt_to.keys():
                my_dict[(j,i)] = mfpt_to[j]
                my_dict[(i,i)] = 1 / find_steady(i)

        return matrix_to_pandas(pykov.Matrix(my_dict))



    def simulate_chain(self, starting_condition, n, end=None):
        """
        Simulates a path of length n following the markov chain with the initial condition of starting_condition

        Parameters
        ----------
        starting_condition : state or Distribution
            If a state, simulates n steps starting at that state. If a Distribution, samples from that distribution
            to find the starting state
        n : integer
            Number of steps to take
        end : state (optional)
            Chain stops as soon as it reaches end

        Returns
        -------
        Array
            Array of the path taken

        Examples
        --------
        >>> mc = Table().states(make_array("A", "B")).transition_probability(make_array(0.5, 0.5, 0.3, 0.7)).toMarkovChain()
        >>> mc.simulate_chain("A", 10)
        array(['A', 'A', 'A', 'A', 'B', 'B', 'A', 'A', 'B', 'B', 'B'])
        >>> mc.simulate_chain("B", 10)
        array(['B', 'B', 'B', 'A', 'B', 'A', 'B', 'A', 'A', 'A', 'B'])
        >>> start = Table().state(make_array("A", "B")).probability(make_array(.8, .2))
        >>> mc.simulate_chain(start, 10)
        array(['A', 'A', 'A', 'B', 'B', 'A', 'B', 'B', 'A', 'B', 'A'])
        >>> mc.simulate_chain(start, 10, end='A')
        array(['B', 'B', 'B', 'A'])

        """

        if isinstance(starting_condition, Table):
            start = starting_condition.sample()
            return np.array(self.chain.walk(n, start, end))
        else:
            return np.array(self.chain.walk(n, starting_condition, end))
    

    def prob_of_path(self, starting_condition, path):
        """
        Finds the probability of a path given a starting condition

        Parameters
        ----------
        starting_condition : state or Distribution
            If a state, finds the probability of the path starting at that state. If a Distribution,
            finds the probability of the path with the first element sampled from the Distribution
        path : array
            Array of states

        Returns
        -------
        float
            probability

        Examples
        --------
        >>> mc = Table().states(make_array("A", "B")).transition_probability(make_array(0.5, 0.5, 0.3, 0.7)).toMarkovChain()
        >>> mc.prob_of_path('A', make_array('A', 'B','B'))
        0.175
        >>> start = Table().state(make_array("A", "B")).probability(make_array(.8, .2))
        >>> mc.prob_of_path(start, make_array('A', 'A', 'B','B'))
        0.14
        >>> 0.175 * 0.8
        0.14
        """

        if isinstance(starting_condition, Table):
            first = path[0]

            # There has to be something better than this
            p_first = starting_condition.column(1)[np.where(starting_condition.column(0) == first)[0]][0]

            return p_first * np.e**(self.chain.walk_probability(path))

        return np.e ** (self.chain.walk_probability([starting_condition] + list(path)))
    
    def is_accessible(self,i,j):
        return self.chain.is_accessible(i,j)
    
    def communicates(self,i,j):
        return self.chain.communicates(i,j)
    
    @pykov_connection
    def accessibility_matrix(self):
        """
        Return matrix showing whether state j is accessible from state i. 1 if accessible, 0 if not

        Parameters
        ----------
        i : state
            Source state
        j : state
            Target state

        Returns
        -------
        DateFrame

        Examples
        --------
        >>> mc = Table().states(make_array("A", "B")).transition_probability(make_array(0.5, 0.5, 0.3, 0.7)).toMarkovChain()
        >>> mc.accessibility_matrix()
              T: A  T: B
        S: A     1     1
        S: B     1     1

        """
        return self.chain.accessibility_matrix()

    def mixing_time(self, cutoff=.25, jump=1, p=None):
        """
        Finds the mixing time

        Parameters
        ----------
        cutoff : float (optional)
            Probability at which distribution is mixed enough (default 0.25)
        jump : int (optional)
            Number of steps to make per iteration (default 1)

        Returns
        -------
        int
           Mixing time

        """
        return self.chain.mixing_time(cutoff, jump, p)

    def empirical_distribution(self, starting_condition, n, repetitions):
        """
        Finds the empirical distribution

        Parameters
        ----------
        starting_condition : state or distribution
            Starting state or distribution of starting state
        n : int
            number of steps
        repetitions : int
            number of repetitions

        Returns
        -------
        Table
            Distribution after n steps over a certain number of repetitions

        Examples
        -------
        >>> mc = Table().states(make_array("A", "B")).transition_probability(make_array(0.5, 0.5, 0.3, 0.7)).toMarkovChain()
        >>> start = Table().state(make_array("A", "B")).probability(make_array(.8, .2))
        >>> mc.empirical_distribution(start, 10, 100)
        Value | Proportion
        A     | 0.4
        B     | 0.6

        """
        end = []
        for i in range(repetitions):
            end.append(self.simulate_chain(starting_condition, n)[-1])
        ed = emp_dist(end)
        ed.relabel("Value", "State")
        return ed


    
def toMarkovChain(table):
    assert table.num_columns == 3, \
    'Must have columns: source,target,probability'
    assert all([round(probsum,6)==1 for probsum in table.group(0,collect=sum).column(2)]), \
        'Transition probabilities must sum to 1 for each source state'
    dict_of_values = {(row[0],row[1]):row[2] for row in table.rows}
    return MarkovChain(pykov.Chain(dict_of_values))