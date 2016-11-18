``prob140`` tutorial!
=====================

This is a brief introduction to the functionality in ``prob140``! For an
interactive guide, see the examples notebook in the GitLab directory.

.. contents:: Table of Contents
    :depth: 2
    :local:

Getting Started
---------------

Make sure to download the most recent ``prob140.py`` file from the `gitlab
repository <https://gitlab.com/probability/prob140/tree/master>`_

Keep the file in the same directory as your notebook, and run the following
import statements:

.. code-block:: python

    # HIDDEN

    import matplotlib
    from datascience import *
    from prob140 import *
    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as np
    plt.style.use('fivethirtyeight')



Creating a Finite Distribution
------------------------------

Discrete Distributions can be represented as either a `FiniteDistribution` or an
`InfiniteDistribution`. We will start with creating a `FiniteDistribution`.

See the `FiniteDistribution` documentation for more details

We can construct a `FiniteDistribution` by explicitly assigning values for
the `domain` and `probability`

.. ipython:: python

    from prob140 import *

    dist1 = FiniteDistribution().domain(make_array(2, 3, 4)).probability(make_array(0.25, 0.5, 0.25))

    print(dist1)

We can also construct a distribution by explicitly assigning values for the
`domain` but applying a probability function to the values of the domain

.. ipython:: python

    dist2 = FiniteDistribution().domain(np.arange(1, 8, 2))
    .probability_function(lambda x: 1/4)

    print(dist2)



Plotting
--------