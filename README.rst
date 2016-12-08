=======
prob140
=======

A Berkeley library for probability theory.

Written for `Prob140 (prob140.org) <prob140.org>`_  by Jason Zhang and Dibya Ghosh.

For documentation, visit `https://probability.gitlab.io/prob140/html/ <https://probability.gitlab.io/prob140/html/>`_

See the `example notebook <https://nbviewer.jupyter.org/urls/gitlab.com/probability/prob140/raw/master/Examples.ipynb>`_
for some samples on using the library


Installation
============

Use ``pip`` for installation::

    pip install prob140

You must also have an updated installation of Berkeley's
`Data Science library <https://github.com/data-8/datascience>`_::

    pip install datascience

If you are using ``prob140`` in a notebook, use the following header:

.. code-block:: python

    import matplotlib
    from datascience import *
    from prob140 import *
    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as np
    plt.style.use('fivethirtyeight')


Changelog
=========

v0.1.4.1
-------

* Single distribution plotting moved from the ``plot_dist`` method to the ``Plot`` function
* Multiple distribution plotting moved from the ``Plot`` function to the ``Plots`` function
* Events are now plotted by passing an argument to ``Plot``

v0.1.3
------

* Added joint distributions
* All ``FiniteDistribution`` objects changed to become ``datascience.tables.Table`` objects
* Began renaming

v0.1.2
------
Initial Release
