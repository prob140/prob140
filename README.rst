=======
prob140
=======

A Berkeley library for probability theory.

Written for `Prob140 (prob140.org) <prob140.org>`_  by Jason Zhang and Dibya Ghosh.

For documentation and examples, visit `https://probability.gitlab.io/prob140/ <https://probability.gitlab.io/prob140/>`_

.. image:: https://gitlab.com/probability/prob140/badges/master/build.svg
    :target: https://gitlab.com/probability/prob140/pipelines
.. image:: https://gitlab.com/probability/prob140/badges/master/coverage.svg
    :target: https://probability.gitlab.io/prob140/coverage


Installation
============

Use ``pip`` for installation::

    pip install prob140

You must also have an updated installation of Berkeley's
`Data Science library <https://github.com/data-8/datascience>`_::

    pip install datascience
    

If you are using ``prob140`` in a notebook, use the following header:

.. code-block:: python

    from datascience import *
    from prob140 import *
    %matplotlib inline
    import matplotlib.pyplot as plt
    import numpy as np
    plt.style.use('fivethirtyeight')
    
If you wish to use the Pykov module, also include: 

.. code-block:: python

    from prob140.pykov import *


Changelog
=========

See `CHANGELOG <https://gitlab.com/probability/prob140/blob/master/CHANGELOG.rst>`_ for full changelog