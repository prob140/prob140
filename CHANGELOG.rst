v0.4.1.3 (2019-09-21)
---------------------

* Refactor `.probability(...)` to `.probabilities(...)`

v0.4.1.2 (2019-01-21)
---------------------

* `event` for Distribution tables now prints the total probability


v0.4.1.1 (2019-01-21)
---------------------

* The `Plot` function and `prob_event` and `event` methods now can take events defined by indicator functions


v0.4.1.0 (2019-01-20)
---------------------

* Updated syntax and behavior for `event` in JointDistribution objects

v0.4.0.0 (2019-01-18)
---------------------

* Distribution tables with two variables are now automatically changed to joint distributions
* Added `total_probability` to JointDistribution objects
* Added `event` to JointDistribution objects

v0.3.9.0 (2018-10-30)
---------------------

* Renamed `Plot_multivariate_normal_regression` to `Plot_multivariate_normal_cond_exp`.

v0.3.8.0 (2018-11-05)
---------------------

* Renamed `multivariate_normal_regression` to `Plot_multivariate_normal_regression`

v0.3.7.1 (2018-10-30)
---------------------

* Added kwargs to  `multivariate_normal_regression` and `Scatter_multivariate_normal`.

v0.3.7.0 (2018-10-30)
---------------------

* `multivariate_normal_regression` and `Scatter_multivariate_normal` now also take in elevation and azimuth.

v0.3.6.0 (2018-10-30)
---------------------

* `Plot_3d` now also takes in elevation and azimuth.

v0.3.5.2 (2018-10-20)
---------------------

* Updated metadata in `setup.py`.

v0.3.5.1 (2018-04-24)
---------------------

* Imported `multivariate_normal_regression`.

v0.3.5.0 (2018-04-24)
---------------------

* Added `multivariate_normal_regression`.

v0.3.4.1 (2018-04-15)
---------------------

* Increased viewing distance for `Scatter_multivariate_normal`.

v0.3.4.0 (2018-04-14)
---------------------

* Added Plot_bivariate_normal and Scatter_multivariate_normal.

v0.3.3.7 (2018-04-11)
---------------------

* Adding `visualize_cr`.

v0.3.3.7 (2018-03-21)
---------------------

* Can now pass figsize to Plot_3d.

v0.3.3.6 (2018-03-04)
---------------------

* New Matplotlib requires edgecolor to see edges.

v0.3.3.5 (2018-03-03)
---------------------

* Reformats y-ticks by scaling the labels rather than scaling the values to get percent per unit.

v0.3.3.4 (2018-02-15)
---------------------

* MarkovChain.distribution typecasts states that are iterables.

v0.3.3.3 (2018-02-11)
---------------------

* plot_path now checks if path is possible by iterating through it.
* simulate_path with plot_path=True no longer returns the path.

v0.3.3.2 (2018-02-11)
---------------------

* Zeroing out negative floating point errors.

v0.3.3.1 (2018-02-10)
---------------------

* Fixed bug in which MarkovChain.steady_state might use the wrong eigenvector.
* Added warnings if transition probabilities are negative or don't sum to 1.

v0.3.3.0 (2018-02-08)
---------------------

* plot_path now only plots the path if the probability is nonzero
* plot_path takes in a starting condition to be consistent with prob_of_path
* simulate_path takes an optional parameter to plot the path

v0.3.2.1 (2018-02-08)
---------------------

* Only show real component of steady state distribution when computing left eigenvector.

v0.3.2.0 (2018-01-18)
---------------------

* Restore functionality for calling `probability_function` for joint distributions.


v0.3.1.1 (2018-01-08)
---------------------

* MarkovChain.distribution() supports a state as a starting distribution.

v0.3.1.0 (2018-01-08)
---------------------

* Code refactor to follow PEP8
* All new Markov Chain module using numpy backend

  * Function definitions largely the same for common functions
  * MarkovChains can now be constructed using additional class functions

* Functions renamed:

  * Table.expected_value -> Table.ev
  * Table.variance -> Table.var

* Unit Tests! Bumped to around 66% code coverage.

v0.2.9.0 (2017-03-19)
---------------------

* Plot_3d

v0.2.8.1 (2017-03-18)
---------------------

* Plot_continuous now accepts python functions too


v0.2.8.0 (2017-03-13)
---------------------

* Updated unconstrain to rearrange_2 and nicefy to rearrange_1

v0.2.7.1 (2017-03-11)
---------------------

* SymPy integration being finalized - added `unconstrain` and updated `declare`

v0.2.7.0 (2017-03-10)
---------------------

* sample renamed to sample_from_dist to avoid conflicts with datascience

v0.2.6.3 (2017-03-09)
---------------------

* Fixed documentation for plots
* plots removed from global

v0.2.6.2 (2017-03-09)
---------------------

* Plot_continuous works with sympy

v0.2.6.1 (2017-03-09)
---------------------

* Plot_continuous now works with any function passed in as func

v0.2.6.0 (2017-03-06)
---------------------

* Wrapper for plotting continuous functions

v0.2.5.1 (2017-03-06)
---------------------

* Beginning to add SymPy integration in *symbolic_math.py*

v0.2.5.0 (2017-02-22)
---------------------

* Added log_probability_of_path

v0.2.4.4 (2017-02-20)
---------------------

* Fixing installation issues

v0.2.4.3a (2017-02-20)
----------------------

* fixed mfpt

v0.2.4.2 (2017-02-16)
---------------------

* Fixed typo in steady_state, not sure how it happened

v0.2.4.1 (2017-02-16)
---------------------

* Documentation fix

v0.2.4.0 (2017-02-13)
---------------------

* Removed T and S from markov chains
* added .column
* states now sorted

v0.2.3.8 (2017-02-13)
---------------------

* Added get target

v0.2.3.7 (2017-02-12)
---------------------

* Deprecation error fix

v0.2.3.6 (2017-02-12)
---------------------

* Distribution now shows states with probability 0

v0.2.3.5 (2017-02-11)
---------------------

* Added show_ev for conditional distributions

v0.2.3.4 (2017-02-11)
---------------------

* state --> states

v0.2.3.3
--------
* Documentation

v0.2.3.2 (2017-02-11)
---------------------
* Changed label for empirical distribution to state
* mc.distribution accepts states

v0.2.3.1 (2017-02-11)
---------------------

* Fixed mean_first_passage_times

v0.2.3.0 (2017-02-11)
---------------------

* Renamed a ton of functions
* Implemented starting conditions

v0.2.2.0 (2017-02-11)
---------------------

* Begin wrapping of pykov

v0.2.1.3 (2017-02-08)
---------------------

* Plots uses plt.bar instead of Table.hist
* Added optional parameter edges=


v0.2.1.2 (2017-02-04)
---------------------

* Added show_ave as optional parameter

v0.2.1.1 (2017-02-04)
---------------------

* Added show_ev and show_sd as optional parameters for plot

v0.2.1.0 (2017-02-04)
---------------------

* Added sample for single variable distributions
* Added CDF for single variable distributions

v0.2.0.0 (2017-02-03)
---------------------

* Pykov

v0.1.8.1 (2017-02-01)
---------------------

* Renamed emp_dist values to proportions rather than probabilities

v0.1.8.0 (2017-01-30)
---------------------

* Added emp_dist to allow for empirical distributions


v0.1.7.6 (2017-01-19)
---------------------

* __version__ instead of version

v0.1.7.5 (2017-01-18)
---------------------

* Joint Distributions no longer give a warning if probabilities rounded to 6 decimal places = 1

v0.1.7.4 (2017-01-17)
---------------------

* Single variable distributions now check that probabilities sum to 1

v0.1.7.3 (2017-01-17)
---------------------

* Plot now adds edge border if there are fewer than 75 bins
* Plot now has an optional parameter edge that accepts a boolean
* Added marginal_dist which returns a single variable distribution

v0.1.7.2 (2017-01-17)
---------------------

* .values is now an alias for .domain

v0.1.7.1 (2017-01-17)
---------------------

* Fixed vertical axis for Plot

v0.1.7.0 (2017-01-16)
---------------------

* Removed marginal_of_X, marginal_of_Y, etc
* conditional_dist_given(given) is now conditional_dist(label, given)

v0.1.6.4 (2017-01-15)
---------------------

* Joint Distribution functions can have arbitrary number of arguments again

v0.1.6.3 (2017-01-15)
---------------------

* fixed a bug in which toJoint just renamed the x-columns rather than changing the order

v0.1.6.2 (2017-01-14)
---------------------

* toJoint now preserve original order

v0.1.6.1 (2017-01-14)
---------------------

* JointDistribution probabilities don't have to sum to 1,

v0.1.6 (2017-01-14)
-------------------

* Added probability_function for JointDistribution
* probability_function now checks number of arguments in pfunc

v0.1.5.1 (2017-01-12)
---------------------

* Added JointDistribution to the init

v0.1.5 (2017-01-12)
-------------------

* Plotting width now works with events and masks
* JointDistribution can now be used with any variable

v0.1.4.3 (2016-12-20)
---------------------

* Changed the colors for plots

v0.1.4.2
--------

* Slight modifications to plot labels

v0.1.4a
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
