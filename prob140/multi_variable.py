from datascience import *
import numpy as np
import pandas as pd

def conditional(array):
    return array/sum(array[0:-1])

class JointDistribution(pd.DataFrame):

    def marginal_of_Y(self):
        copy = JointDistribution(self,copy=True)
        copy['Sum: Marginal of Y'] = copy.sum(axis=1)
        return copy

    def marginal_of_X(self):
        copy = JointDistribution(self,copy=True)
        copy.loc['Sum: Marginal of X'] = copy.sum(axis=0)
        return copy

    def both_marginals(self):
        copy = JointDistribution(self,copy=True)
        copy['Sum: Marginal of Y'] = copy.sum(axis=1)
        copy.loc['Sum: Marginal of X'] = copy.sum(axis=0)
        return copy

    def conditional_dist_X_given_Y(self):
        both = self.both_marginals()
        
        x = both.apply(conditional, axis=1).rename(columns = {'Sum: Marginal of Y':'Sum'})
        
        indices = both.index
        new = make_array()
        for i in np.arange(len(indices)-1):
            new_index = 'Dist. of X | '+indices[i]
            new = np.append(new, new_index)
        new = np.append(new, 'Marginal of X')
        
        return x.set_index(new)

    def conditional_dist_Y_given_X(self):
        both = self.both_marginals()
        
        indices = both.index
        new = np.append(both.index[0:-1], 'Sum')
        y = both.apply(conditional, axis=0).set_index(new)
        
        column_names = y.columns
        new = make_array()
        for i in np.arange(len(column_names)-1):
            new_name = 'Dist. of Y | '+column_names[i]
            new = np.append(new, new_name)
        new = np.append(new, 'Marginal of Y')
        
        y.columns = new
        return y

import itertools as it

def multi_domain(table,*args):
    var_names = [chr(ord('X')+i) for i in range(len(args))]
    var_values = list(zip(*it.product(*args)))
    new_table = table.copy()
    for column_name,column_value in reversed(list(zip(var_names,var_values))):
        new_table = new_table.with_column(column_name,column_value)
        new_table.move_to_start(column_name)

    return new_table

def toJoint(table,X_column_label=None,Y_column_label=None,probability_column_label=None):
    assert table.num_columns >= 3, "You must have columns for your X variable, for your Y variable, and for your probabilities"
    if X_column_label is None:
        X_column_label = table.labels[0]
    if Y_column_label is None:
        Y_column_label = table.labels[1]
    if probability_column_label is None:
        probability_column_label = table.labels[table.num_columns-1]

    assert np.allclose(sum(table[probability_column_label]),1), "Your probabilities don't sum to 1"
    

    x_possibilities = sorted(set(table[X_column_label]))
    y_possibilities = sorted(set(table[Y_column_label]),reverse=True)


    xInd = table.column_index(X_column_label)
    yInd = table.column_index(Y_column_label)
    pInd = table.column_index(probability_column_label)
    
    data = {poss: [0]*len(y_possibilities) for poss in x_possibilities}
    
    for row in table.rows:
        data[row[xInd]][y_possibilities.index(row[yInd])] += row[pInd]


    realData = {'%s=%s'%(X_column_label,str(poss)):value for poss,value in data.items()}
    index = ['%s=%s'%(Y_column_label,poss) for poss in y_possibilities]
    joint_dist = JointDistribution(realData,index=index)

    return joint_dist
    