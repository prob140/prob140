from datascience import *
from .rebinding import *
import numpy as np
import pandas as pd
import warnings
import ast

def conditional(array):
    value = array/sum(array[0:-1])
    return value


def evaluate(name):
    """
    Deletes name of RV and outputs the correct datatype, like int, float, or string

    Parameters
    ----------
    name : String
        in the form "rv=123123124"

    Returns
    -------
    String, int, float

    """
    try:
        index = name.index("=")
    except:
        index = name.index(" ")

    try:
        return ast.literal_eval(name[index + 1:])
    except:
        return name[index + 1:]

class JointDistribution(pd.DataFrame):

    def marginal(self, label):
        """
        Returns the marginal distribution of label

        Parameters
        ----------
        label : String
            The label of the variable of which we want to find the marginal distribution

        Returns
        -------
        JointDistribution Table

        Examples
        --------
        >>> dist2 = Table().values("Coin1",['H','T'],"Coin2", ['H','T']).probability(np.array([0.24, 0.36, 0.16, 0.24])).toJoint()
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

    def marginal_dist(self, label):
        """
        Finds the marginal marginal distribution of label, returns as a single variable distribution

        Parameters
        ----------
        label
            The label of the variable of which we want to find the marginal distribution

        Returns
        -------
        Table
            Single variable distribution of label

        """


        marginal = self.marginal(label).as_matrix()

        if label == self._X_column_label:
            x_labels = list(self)
            prob = marginal[-1,:]

            domain = [evaluate(lab) for lab in x_labels]

            return Table().values(domain).probability(prob)

        else:
            y_labels = list(self.index)
            prob = marginal[:,-1]

            domain = [evaluate(lab) for lab in y_labels]

            return Table().values(domain).probability(prob)


    def both_marginals(self):
        """
        Finds the marginal distribution of both variables

        Returns
        -------
        JointDistribution Table

        Examples
        --------
        >>> dist1 = Table().values([0,1],[2,3]).probability([0.1, 0.2, 0.3, 0.4]).toJoint()
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

    def conditional_dist(self, label, given="", show_ev=False):
        """
        Given the random variable label, finds the conditional distribution of the other variable

        Parameters
        ----------
        label : String
            given variable

        Returns
        -------
        JointDistribution Table

        Examples
        --------
        >>> coins = Table().values("Coin1",['H','T'],"Coin2", ['H','T']).probability(np.array([0.24, 0.36, 0.16,0.24])).toJoint()
        >>> coins.conditional_dist("Coin1","Coin2")
                                  Coin1=H  Coin1=T  Sum
        Dist. of Coin1 | Coin2=H      0.6      0.4  1.0
        Dist. of Coin1 | Coin2=T      0.6      0.4  1.0
        Marginal of Coin1             0.6      0.4  1.0
        >>> coins.conditional_dist("Coin2","Coin1")
                 Dist. of Coin2 | Coin1=H  Dist. of Coin2 | Coin1=T  Marginal of Coin2
        Coin2=H                       0.4                       0.4                0.4
        Coin2=T                       0.6                       0.6                0.6
        Sum                           1.0                       1.0                1.0
        """

        if label == self._Y_column_label:

            both = self.both_marginals()

            indices = both.index
            new = np.append(both.index[0:-1], 'Sum')
            y = both.apply(conditional, axis=0).set_index(new)

            matrix = y.as_matrix()[:-1,:]
            y_labels = list(self.index)
            domain = np.array([evaluate(lab) for lab in y_labels])

            exp_values = [sum(matrix[:,i] * domain) for i in range(len(matrix[0]))]


            column_names = y.columns
            new = make_array()
            for i in np.arange(len(column_names) - 1):
                new_name = 'Dist. of {0} | '.format(self._Y_column_label) + column_names[i]
                new = np.append(new, new_name)
            new = np.append(new, 'Marginal of {0}'.format(self._Y_column_label))

            y.columns = new

            if show_ev:
                y.loc["EV"] = exp_values

            return y

        elif label == self._X_column_label:
            both = self.both_marginals()

            x = both.apply(conditional, axis=1) \
                .rename(columns={'Sum: Marginal of {0}'.format(self._Y_column_label): 'Sum'})


            matrix = x.as_matrix()[:,:-1]
            x_labels = list(self)
            domain = np.array([evaluate(lab) for lab in x_labels])
            exp_values = [sum(matrix[i] * domain) for i in range(len(matrix))]

            indices = both.index


            new = make_array()
            for i in np.arange(len(indices) - 1):
                new_index = 'Dist. of {0} | '.format(self._X_column_label) + indices[i]
                new = np.append(new, new_index)
            new = np.append(new, 'Marginal of {0}'.format(self._X_column_label))


            new_df = x.set_index(new)

            if show_ev:
                new_df["EV"] = exp_values

            return new_df

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



def toJoint(table, X_column_label=None, Y_column_label=None, probability_column_label=None, reverse=True):
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
        Label for probabilities\
    reverse (optional) : Boolean
        If True, the vertical values will be reversed

    Returns
    -------
    JointDistribution
        A JointDistribution object

    Examples
    --------
    >>> dist1 = Table().values([0,1],[2,3])
    >>> dist1['Probability'] = make_array(0.1, 0.2, 0.3, 0.4)
    >>> dist1.toJoint()
         X=0  X=1
    Y=3  0.2  0.4
    Y=2  0.1  0.3
    >>> dist2 = Table().values("Coin1",['H','T'], "Coin2", ['H','T'])
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

    total = sum(table[probability_column_label])

    if round(total, 6) != 1:
        warnings.warn("Your probabilities sum to {0}".format(total))
    

    x_possibilities = sorted(set(table[X_column_label]))
    y_possibilities = sorted(set(table[Y_column_label]),reverse=reverse)


    xInd = table.column_index(X_column_label)
    yInd = table.column_index(Y_column_label)
    pInd = table.column_index(probability_column_label)
    
    data = {poss: [0]*len(y_possibilities) for poss in x_possibilities}
    
    for row in table.rows:
        data[row[xInd]][y_possibilities.index(row[yInd])] += row[pInd]

    x_order = ['%s=%s'%(X_column_label,poss) for poss in x_possibilities]

    realData = {'%s=%s'%(X_column_label,str(poss)):value for poss,value in data.items()}
    index = ['%s=%s'%(Y_column_label,poss) for poss in y_possibilities]

    # Reverting order back to original
    df = pd.DataFrame(realData,index=index)
    joint_dist = JointDistribution(df[x_order], index=index)

    joint_dist.reindex(index)

    joint_dist._X_column_label = X_column_label
    joint_dist._Y_column_label = Y_column_label

    return joint_dist
    