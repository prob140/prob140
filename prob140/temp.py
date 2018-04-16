import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import ipywidgets as widgets
from ipywidgets import interact


def cr_process(N, theta):
    tables = np.array([])
    people = np.array([])

    counts = []
    for i in range(N):
        n = sum(people)
        new_table = len(tables) + 1

        tbl_choices = np.append(tables, new_table)
        tbl_probs = np.append(people, theta) / (n + theta)

        choice = int(np.random.choice(tbl_choices, p=tbl_probs))

        if choice == new_table:
            tables = tbl_choices
            people = np.append(people, 1)
        else:
            people[choice - 1] = people[choice - 1] + 1
        counts.append(people.copy())

    return counts


def visualize_cr(people=None):
    """
    Visualization for the Chinese Restaurant Process.

    Parameters
    ----------
    people : ndarray (optional)
        A list of table counts.
    """
    def plot(n):
        people = counts[n - 1]
        num_table = len(people)
        for i, p in enumerate(people):
            plt.scatter(np.arange(1, p + 1), [i + 1] * int(p))
        plt.yticks(np.arange(1, num_table + 1))
        plt.ylim(0, 15)
        plt.xlim(0.1, max(10, max(people) + 1))
        ax = plt.gca()
        ax.grid(False)
        plt.ylabel('Table Number')
        plt.xlabel('Number of People')

    if people is None:
        theta = stats.uniform.rvs(loc=0.5, scale=2, size=1)
        counts = cr_process(100, theta)
        n = widgets.IntSlider(
            value=1,
            min=1,
            max=100,
            description='n',
            continuous_update=False
        )
        interact(plot, n=n)
    else:
        counts = [people]
        plot(0)
