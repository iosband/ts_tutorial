"""Specify the jobs to run via config file.

A simple experiment comparing Thompson sampling to greedy algorithm. Finite
armed bandit with 3 arms. Greedy algorithm premature and suboptimal
exploitation.
See Figure 3 from https://arxiv.org/abs/1707.02038
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

from base.config_lib import Config
from base.experiment import BaseExperiment
from finite_arm.agent_finite import FiniteBernoulliBanditEpsilonGreedy
from finite_arm.agent_finite import FiniteBernoulliBanditTS
from finite_arm.env_finite import FiniteArmedBernoulliBandit

import numpy as np


def get_config():
  """Generates the config for the experiment."""
  name = 'finite_simple_rand'
  n_arm = 3
  agents = collections.OrderedDict(
      [('greedy',
        functools.partial(FiniteBernoulliBanditEpsilonGreedy, n_arm)),
       ('ts', functools.partial(FiniteBernoulliBanditTS, n_arm))]
  )

  environments = collections.OrderedDict()
  n_env = 100
  for env in range(n_env):
    probs = np.random.rand(n_arm)
    environments[env] = functools.partial(FiniteArmedBernoulliBandit, probs)

  experiments = collections.OrderedDict(
      [(name, BaseExperiment)]
  )
  n_steps = 1000
  n_seeds = 100
  config = Config(name, agents, environments, experiments, n_steps, n_seeds)
  return config
