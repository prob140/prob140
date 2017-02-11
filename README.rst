=======
prob140
=======

A Berkeley library for probability theory.

Written for `Prob140 (prob140.org) <prob140.org>`_  by Jason Zhang and Dibya Ghosh.

Pykov Module written by `Riccardo Scalco <https://github.com/riccardoscalco/Pykov>`_. Included in library with permission.


For documentation, visit `https://probability.gitlab.io/prob140/html/ <https://probability.gitlab.io/prob140/html/>`_

For example notebooks on using the library, see `single variable notebook <https://nbviewer.jupyter.org/urls/gitlab.com/probability/prob140/raw/master/Examples.ipynb>`_
and `multi variable notebook <https://nbviewer.jupyter.org/urls/gitlab.com/probability/prob140/raw/master/joint_distribution.ipynb>`_


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
    
If you wish to use the Pykov module, also include: 

.. code-block:: python

    from prob140.pykov import *


Changelog highlights
====================

See `CHANGELOG <https://gitlab.com/probability/prob140/blob/master/CHANGELOG>`_ for full changelog

v0.1.8 (2017-01-30)
---------------------

* Added emp_dist to allow for empirical distributions

v0.1.7 (2017-01-16)
-------------------

* Marginal and Conditional Distributions accept any label names
* Can use .values instead of .domain
* Improvements to edges for Plot
* Sanity checks for probabilities

v0.1.6 (2017-01-14)
-------------------

* toJoint preserves order
* Various bug and usability fixes

v0.1.5 (2017-01-12)
-------------------

* Plotting width now works with events and masks
* JointDistribution can now be used with any variable

v0.1.4.1
--------

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
