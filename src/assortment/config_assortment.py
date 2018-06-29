"""Specify the jobs to run via config file.

Dynamic pricing experiment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import numpy as np

from base.config_lib import Config
from base.experiment import ExperimentNoAction
from assortment.agent_assortment import TSAssortment, GreedyAssortment, EpsilonGreedyAssortment,AnnealingEpsilonGreedyAssortment
from assortment.env_assortment import ProductAssortment


def get_config():
  """Generates the config for the experiment."""
  name = 'product_assortment'
  num_products = 6
  prior_mean = 0
  prior_var_diagonal = 1
  prior_var_off_diagonal = 0.2
  noise_var = 0.04
  profits = np.array([1/6]*6)
  epsilon = 0.07
  k = 9
 

  agents = collections.OrderedDict(
      [('TS',
        functools.partial(TSAssortment,
                          num_products, prior_mean, prior_var_diagonal,prior_var_off_diagonal, noise_var, profits,epsilon,k)),
       ('greedy',
        functools.partial(GreedyAssortment,
                          num_products, prior_mean, prior_var_diagonal,prior_var_off_diagonal, noise_var, profits,epsilon,k)),
        (str(epsilon) + '-greedy',
        functools.partial(EpsilonGreedyAssortment,
                          num_products, prior_mean, prior_var_diagonal,prior_var_off_diagonal, noise_var, profits,epsilon,k)),
         (str(k)+'/('+str(k)+'+t)-greedy',
        functools.partial(AnnealingEpsilonGreedyAssortment,
                          num_products, prior_mean, prior_var_diagonal,prior_var_off_diagonal, noise_var, profits,epsilon,k))]
  )

  environments = collections.OrderedDict(
      [('env',
        functools.partial(ProductAssortment,
                          num_products, prior_mean, prior_var_diagonal,prior_var_off_diagonal, noise_var, profits))]
  )
  experiments = collections.OrderedDict(
      [(name, ExperimentNoAction)]
  )
  n_steps = 500
  n_seeds = 20000
  config = Config(name, agents, environments, experiments, n_steps, n_seeds)
  return config

