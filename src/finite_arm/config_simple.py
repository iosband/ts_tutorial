"""Specify the jobs to run via config file.

A simple experiment comparing Thompson sampling to greedy algorithm. Finite
armed bandit with 3 arms. Greedy algorithm premature and suboptimal
exploitation.
See Figure 3 from https://arxiv.org/abs/1707.02038
"""

import collections
import functools

from base.config_lib import Config
from base.experiment import BaseExperiment
from finite_arm.agent_finite import FiniteBernoulliBanditEpsilonGreedy
from finite_arm.agent_finite import FiniteBernoulliBanditTS
from finite_arm.env_finite import FiniteArmedBernoulliBandit


def get_config():
  """Generates the config for the experiment."""
  name = 'finite_simple'
  n_arm = 3
  agents = collections.OrderedDict(
      [('greedy',
        functools.partial(FiniteBernoulliBanditEpsilonGreedy, n_arm)),
       ('ts', functools.partial(FiniteBernoulliBanditTS, n_arm))]
  )
  probs = [0.7, 0.8, 0.9]
  environments = collections.OrderedDict(
      [('env', functools.partial(FiniteArmedBernoulliBandit, probs))]
  )
  experiments = collections.OrderedDict(
      [(name, BaseExperiment)]
  )
  n_steps = 1000
  n_seeds = 10000
  config = Config(name, agents, environments, experiments, n_steps, n_seeds)
  return config
