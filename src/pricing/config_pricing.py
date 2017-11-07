"""Specify the jobs to run via config file.

Dynamic pricing experiment.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

from base.config_lib import Config
from base.experiment import ExperimentNoAction
from pricing.agent_pricing import BootstrapDynamicPricing
from pricing.env_pricing import DynamicPricing


def get_config():
  """Generates the config for the experiment."""
  name = 'dynamic_pricing'
  num_products = 5
  scale = 1
  noise_var = 10
  p_max = 1

  agents = collections.OrderedDict(
      [('bsPricing',
        functools.partial(BootstrapDynamicPricing,
                          num_products, scale, noise_var, p_max))]
  )

  environments = collections.OrderedDict(
      [('env',
        functools.partial(DynamicPricing,
                          num_products, scale, noise_var, p_max))]
  )
  experiments = collections.OrderedDict(
      [(name, ExperimentNoAction)]
  )
  n_steps = 80
  n_seeds = 2000
  config = Config(name, agents, environments, experiments, n_steps, n_seeds)
  return config

