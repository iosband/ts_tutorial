"""Common scripts for plotting the results of experiments.
"""

from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


##############################################################################

class ResultsPlotter(object):

  def __init__(self, results):
    """Takes in a list of experiment results and performs common plots.

    Args:
      results - list of pandas dataframes for each experiment.
    """
    self.df = pd.concat(results)

  def alg_seed_plot_variable(self, variable='instant_regret'):
    """Simple plotting for single variable for each algorithm averaged by seed.
    Note that this is not a general function, just something useful.

    Args:
      variable - string saying which column to plot

    Returns:
      NULL - displays plot to screen.
    """
    self.df['alg'] = self.df.unique_id.apply(lambda x: x.split('|')[0])
    self.df['seed'] = self.df.unique_id.apply(lambda x: x.split('|')[1])

    alg_series = (self.df.groupby(['t', 'alg'])
                  .agg({variable: np.mean})
                  .reset_index()
                  .pivot(index='t', columns='alg')
                  .reset_index(drop=True))[variable]
    colors = ['b', 'r', 'g', 'k', 'y', 'c', 'm']
    for i, alg in enumerate(alg_series.columns):
      plt.plot(alg_series[alg], color=colors[i % len(colors)], alpha=0.75, label=alg)

    plt.legend(loc=0, fontsize=16)
    plt.xlabel("Time", fontsize=16)
    plt.ylabel(variable, fontsize=16)

    plt.show()

  def proportion_action_plot(self, alg):
    """Plot the proportion of actions selection for a single algorithm.

    Args:
      alg - which algorithm to plot

    Returns:
      NULL - displays plot to screen.
    """
    self.df['alg'] = self.df.unique_id.apply(lambda x: x.split('|')[0])
    self.df['seed'] = self.df.unique_id.apply(lambda x: x.split('|')[1])
    plt_df = copy.deepcopy(self.df[self.df.alg == alg])

    n_action = np.max(plt_df.action) + 1

    colors = ['b', 'r', 'g', 'k', 'y', 'c', 'm']
    for i in range(n_action):
      probs = plt_df.groupby('t').agg({'action': lambda x: np.mean(x == i)})
      plt.plot(probs, color=colors[i % len(colors)], alpha=0.75, label=i)

    plt.legend(loc=0, fontsize=16)
    plt.xlabel("Time", fontsize=16)
    plt.ylabel("Action proportion", fontsize=16)
    plt.title(alg + " algorithm", fontsize=16)

    plt.show()

