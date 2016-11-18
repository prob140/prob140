# Library for Probability 140

[nbviewer for Examples](https://nbviewer.jupyter.org/urls/gitlab.com/probability/prob140/raw/master/Examples.ipynb)

[Link for documentation](https://probability.gitlab.io/prob140/html/)

## DiscreteDistribution

Methods:
* `prob_event(arg)`: Finds the probability of a value or a list of values. Sums up the probabilities. Usage: `binDist.prob_event(0)` or `binDist.prob_event(np.arange(0,10))`
* `event(arg)`: Same as `prob_event(arg)` except returns a table with the probabilities rather than the sum
* `expected_value()`
* `variance()`
* `sd()`
* `plot(width=1, mask=None, **vargs)`: See Examples notebook
* `plot_event(event, **vargs)`: See Examples notebook

### FiniteDistribution

Subclass of Discrete DiscreteDistribution

Initialize with `FiniteDistribution().domain(list).probability(list)` or `FiniteDistribution().domain(list).probability_function(function)`

Examples:
```python
>>> FiniteDistribution().domain(make_array(2, 3, 4)).probability(make_array(0.25, 0.5, 0.25))
Domain | Probability
2      | 0.25
3      | 0.5
4      | 0.25

>>> FiniteDistribution().domain(make_array(1, 2, 3, 4)).probability_function(lambda x:x/10)
Domain | Probability
1      | 0.1
2      | 0.2
3      | 0.3
4      | 0.4
```

### InfiniteDistribution

Subclass of DiscreteDistribution

Initialize with `InfiniteDistribution().domain(start, end, step=1).probability_function(function)`

```python
>>> p = 0.2
>>> InfiniteDistribution().domain(1, inf).probability_function(lambda x: p*(1-p)**(x-1))
Domain           | Probability
1                | 0.2
2                | 0.16
3                | 0.128
4                | 0.1024
5                | 0.08192
6                | 0.065536
7                | 0.0524288
8                | 0.041943
9                | 0.0335544
10               | 0.0268435
... (Infinite rows omitted)
```

### JointDistribution

### Recent Changelog
* defining the domain out of order no longer messes up plots
* negative probabilities get a warnings
* renamed `p` to `prob_event`, `P` to `event`
* renamed `binWidth` to `width`
* `plot_event` and `mask` now work, but ignores width!
