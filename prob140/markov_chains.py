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
        return self.chain.__repr__()

    def __str__(self):
        return self.chain.__str__()

    def _repr_html_(self):
        return matrix_to_pandas(self.chain)._repr_html_()
    
    @pykov_connection
    def move(self,state):
        return self.chain.move(state)
    
    @pykov_connection
    def distribution(self, starting_condition, n):
        return self.chain.pow(starting_condition, n)
    
    @pykov_connection
    def steady_state(self):
        return self.chain.steady()
    
    def mean_first_passage_time_to(self,target_state):
        return vector_to_table(self.chain.mfpt_to(target_state),'Mean Time')
    
    def simulate_chain(self, starting_condition, n, end=None):

        if isinstance(starting_condition, Table):
            start = starting_condition.sample()
            return np.array(self.chain.walk(n, start, end))
        else:
            return np.array(self.chain.walk(n, starting_condition, end))
    

    def prob_of_path(self, starting_condition, path):

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
        return self.chain.accessibility_matrix()

    def mixing_time(self, cutoff=.25, jump=1, p=None):
        return self.chain.mixing_time(cutoff, jump, p)

    def empirical_distribution(self, starting_condition, n, repetitions):

        end = []
        for i in range(repetitions):
            end.append(self.simulate_chain(starting_condition, n)[-1])
        return emp_dist(end)


    
def toMarkovChain(table):
    assert table.num_columns == 3, \
    'Must have columns: source,target,probability'
    assert all([round(probsum,6)==1 for probsum in table.group(0,collect=sum).column(2)]), \
        'Transition probabilities must sum to 1 for each source state'
    dict_of_values = {(row[0],row[1]):row[2] for row in table.rows}
    return MarkovChain(pykov.Chain(dict_of_values))