"""Specify the jobs to run via config file.

A simple experiment comparing Thompson sampling to greedy algorithm. Finite
armed bandit with 3 arms. Greedy algorithm premature and suboptimal
exploitation.
See Figure 10(a) from https://arxiv.org/abs/1707.02038
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

from base.config_lib import Config
from base.experiment import BaseExperiment
from finite_arm.agent_finite import FiniteBernoulliBanditBootstrap
from finite_arm.agent_finite import FiniteBernoulliBanditLaplace
from finite_arm.agent_finite import FiniteBernoulliBanditTS
from finite_arm.agent_finite import FiniteBernoulliBanditLangevin
from finite_arm.env_finite import FiniteArmedBernoulliBandit

import numpy as np

def get_config():
  """Generates the config for the experiment."""
  name = 'finite_simple_sanity'
  n_arm = 3
  step_size = 0.01
  step_count = 100
  agents = collections.OrderedDict(
      [('Laplace TS', functools.partial(FiniteBernoulliBanditLaplace, n_arm)),
       ('Langevin TS', functools.partial(FiniteBernoulliBanditLangevin, n_arm,step_count,step_size),),
       ('bootstrap TS', functools.partial(FiniteBernoulliBanditBootstrap, n_arm)),
       ('TS', functools.partial(FiniteBernoulliBanditTS, n_arm))]
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
