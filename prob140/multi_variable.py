from datascience import *
import numpy as np
import pandas as pd
import warnings

def conditional(array):
    return array/sum(array[0:-1])

class JointDistribution(pd.DataFrame):

    def marginal_of_Y(self):
        """
        Find the marginal distribution of the second variable

        Examples
        --------
        >>> dist1 = Table().domain([0,1],[2,3]).probability([0.1, 0.2, 0.3, 0.4]).toJoint()
        >>> dist1.marginal_of_Y()
             X=0  X=1  Sum: Marginal of Y
        Y=3  0.2  0.4                 0.6
        Y=2  0.1  0.3                 0.4

        """
        copy = JointDistribution(self, copy=True)
        copy['Sum: Marginal of {0}'.format(self._Y_column_label)] = copy.sum(axis=1)
        return copy

    def marginal_of_X(self):
        """
        Finds the marginal distribution of the first variable

        Examples
        --------
        >>> dist1 = Table().domain([0,1],[2,3]).probability([0.1, 0.2, 0.3, 0.4]).toJoint()
        >>> dist1.marginal_of_X()
                            X=0  X=1
        Y=3                 0.2  0.4
        Y=2                 0.1  0.3
        Sum: Marginal of X  0.3  0.7
        """
        copy = JointDistribution(self, copy=True)
        copy.loc['Sum: Marginal of {0}'.format(self._X_column_label)] = copy.sum(axis=0)
        return copy

    def marginal(self, label):
        """
        Returns the marginal distribution of label

        Parameters
        ----------
        label : String
        The label of the variable of which we want to find the marginal distribution

        Examples
        --------
        >>> dist2 = Table().domain("Coin1",['H','T'],"Coin2", ['H','T']).probability(np.array([0.24, 0.36, 0.16, 0.24])).toJoint()
        >>> dist2.marginal("Coin1")
                                Coin1=H  Coin1=T
        Coin2=T                    0.36     0.24
        Coin2=H                    0.24     0.16
        Sum: Marginal of Coin1     0.60     0.40
        >>> dist2.marginal("Coin2")
                 Coin1=H  Coin1=T  Sum: Marginal of Coin2
        Coin2=T     0.36     0.24                     0.6
        Coin2=H     0.24     0.16                     0.4
        """
        copy = JointDistribution(self, copy=True)

        if label == self._X_column_label:
            copy.loc['Sum: Marginal of {0}'.format(self._X_column_label)] = copy.sum(axis=0)
        elif label == self._Y_column_label:
            copy['Sum: Marginal of {0}'.format(self._Y_column_label)] = copy.sum(axis=1)
        else:
            raise AssertionError("Label doesn't correspond with existing variable name")

        return copy

    def both_marginals(self):
        """
        Finds the marginal distribution of both variables

        Examples
        --------
        >>> dist1 = Table().domain([0,1],[2,3]).probability([0.1, 0.2, 0.3, 0.4]).toJoint()
        >>> dist1.both_marginals()
                            X=0  X=1  Sum: Marginal of Y
        Y=3                 0.2  0.4                 0.6
        Y=2                 0.1  0.3                 0.4
        Sum: Marginal of X  0.3  0.7                 1.0
        """
        copy = JointDistribution(self,copy=True)
        copy['Sum: Marginal of {0}'.format(self._Y_column_label)] = copy.sum(axis=1)
        copy.loc['Sum: Marginal of {0}'.format(self._X_column_label)] = copy.sum(axis=0)
        return copy

    def conditional_dist_X_given_Y(self):
        """
        Finds the conditional distribution of the first variable given the second variable

        Examples
        --------
        >>> dist1 = Table().domain([0,1],[2,3]).probability([0.1, 0.2, 0.3, 0.4]).toJoint()
        >>> dist1.conditional_dist_X_given_Y()
                               X=0       X=1  Sum
        Dist. of X | Y=3  0.333333  0.666667  1.0
        Dist. of X | Y=2  0.250000  0.750000  1.0
        Marginal of X     0.300000  0.700000  1.0

        """
        both = self.both_marginals()
        
        x = both.apply(conditional, axis=1)\
            .rename(columns={'Sum: Marginal of {0}'.format(self._Y_column_label) :'Sum'})
        
        indices = both.index
        new = make_array()
        for i in np.arange(len(indices)-1):
            new_index = 'Dist. of {0} | '.format(self._X_column_label)+indices[i]
            new = np.append(new, new_index)
        new = np.append(new, 'Marginal of {0}'.format(self._X_column_label))
        
        return x.set_index(new)

    def conditional_dist_Y_given_X(self):
        """
        Finds the conditional distribution of the second variable given the first variable

        Examples
        --------
        >>> dist1 = Table().domain([0,1],[2,3]).probability([0.1, 0.2, 0.3, 0.4]).toJoint()
        >>> dist1.conditional_dist_Y_given_X()
             Dist. of Y | X=0  Dist. of Y | X=1  Marginal of Y
        Y=3          0.666667          0.571429            0.6
        Y=2          0.333333          0.428571            0.4
        Sum          1.000000          1.000000            1.0
        """
        both = self.both_marginals()
        
        indices = both.index
        new = np.append(both.index[0:-1], 'Sum')
        y = both.apply(conditional, axis=0).set_index(new)
        
        column_names = y.columns
        new = make_array()
        for i in np.arange(len(column_names)-1):
            new_name = 'Dist. of {0} | '.format(self._Y_column_label)+column_names[i]
            new = np.append(new, new_name)
        new = np.append(new, 'Marginal of {0}'.format(self._Y_column_label))
        
        y.columns = new
        return y

    def conditional_dist_given(self, label):
        """
        Given the random variable label, finds the conditional distribution of the other variable

        Parameters
        ----------
        label : String
            given variable

        Examples
        --------
        >>> dist2 = Table().domain("Coin1",['H','T'],"Coin2", ['H','T']).probability(np.array([0.24, 0.36, 0.16, 0.24])).toJoint()
        >>> dist2.conditional_dist_given("Coin2")
                                  Coin1=H  Coin1=T  Sum
        Dist. of Coin1 | Coin2=T      0.6      0.4  1.0
        Dist. of Coin1 | Coin2=H      0.6      0.4  1.0
        Marginal of Coin1             0.6      0.4  1.0

        """

        if label == self._X_column_label:
            return self.conditional_dist_Y_given_X()
        elif label == self._Y_column_label:
            return self.conditional_dist_X_given_Y()
        else:
            raise AssertionError("Label doesn't correspond with existing variable name")

import itertools as it

def multi_domain(table,*args):

    if isinstance(args[0], str):
        assert len(args) % 2 == 0, "Must alternate between name and values"
        var_names = [args[2 * i] for i in range(len(args) // 2)]
        values = [args[2 * i + 1] for i in range(len(args) // 2)]
        var_values = list(zip(*it.product(*values)))
    else:
        var_names = [chr(ord('X')+i) for i in range(len(args))]
        var_values = list(zip(*it.product(*args)))

    new_table = table.copy()
    for column_name,column_value in reversed(list(zip(var_names,var_values))):
        new_table = new_table.with_column(column_name,column_value)
        new_table.move_to_start(column_name)

    return new_table

def multi_probability_function(table, pfunc):
    x = table.column(0)
    y = table.column(1)
    values = np.zeros(len(x))
    for i in range(len(x)):
        values[i] = pfunc(x[i], y[i])
    if any(values < 0):
        warnings.warn("Probability cannot be negative")
    return table.with_column('Probability', values)



def toJoint(table,X_column_label=None,Y_column_label=None,probability_column_label=None):
    """
    Converts a table of probabilities associated with two variables into a JointDistribution object

    Parameters
    ----------
    table : Table
        You can either call pass in a Table directly or call the toJoint() method of that Table. See examples
    X_column_label (optional) : String
        Label for the first variable. Defaults to the same label as that of first variable of Table
    Y_column_label (optional) : String
        Label for the second variable. Defaults to the same label as that of second variable of Table
    probability_column_label (optional) : String
        Label for probabilities

    Returns
    -------
    JointDistribution
        A JointDistribution object

    Examples
    --------
    >>> dist1 = Table().domain([0,1],[2,3])
    >>> dist1['Probability'] = make_array(0.1, 0.2, 0.3, 0.4)
    >>> dist1.toJoint()
         X=0  X=1
    Y=3  0.2  0.4
    Y=2  0.1  0.3
    >>> dist2 = Table().domain("Coin1",['H','T'], "Coin2", ['H','T'])
    >>> dist2['Probability'] = np.array([0.4*0.6, 0.6*0.6, 0.4*0.4, 0.6*0.4])
    >>> dist2.toJoint()
             Coin1=H  Coin1=T
    Coin2=T     0.36     0.24
    Coin2=H     0.24     0.16

    """
    assert table.num_columns >= 3, \
        "You must have columns for your X variable, for your Y variable, and for your probabilities"
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

    joint_dist._X_column_label = X_column_label
    joint_dist._Y_column_label = Y_column_label

    return joint_dist
    