"""Specify the jobs to run via config file.

Finite multi-armed bandit with drift.
Comparing the performance of Thompson sampling with forgetting factor to
Thompson sampling without forgetting factor in a nonstationary environment.
See Figure 12 https://arxiv.org/pdf/1707.02038.pdf
"""

import collections
import functools

from base.config_lib import Config
from base.experiment import BaseExperiment
from finite_arm.agent_finite import DriftingFiniteBernoulliBanditTS
from finite_arm.agent_finite import FiniteBernoulliBanditTS
from finite_arm.env_finite import DriftingFiniteArmedBernoulliBandit


def get_config():
  """Generates the config for the experiment."""
  name = 'finite_drift'
  n_arm = 3
  agents = collections.OrderedDict(
      [('stationary_ts',
        functools.partial(FiniteBernoulliBanditTS, n_arm)),
       ('nonstationary_ts',
        functools.partial(DriftingFiniteBernoulliBanditTS, n_arm))]
  )

  environments = collections.OrderedDict(
      [('env', functools.partial(DriftingFiniteArmedBernoulliBandit, n_arm))]
  )
  experiments = collections.OrderedDict(
      [(name, BaseExperiment)]
  )
  n_steps = 1000
  n_seeds = 10000
  config = Config(name, agents, environments, experiments, n_steps, n_seeds)
  return config

