"""Specify the jobs to run via config file.

Finite multi-armed bandit with mis-specified prior.
Comparing the performance of Thompson sampling with an informed prior, with
Thompson sampling with an uninformed (incorrect) prior.
See Figure 11 https://arxiv.org/pdf/1707.02038.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

from base.config_lib import Config
from base.experiment import ExperimentWithMean
from finite_arm.agent_finite import FiniteBernoulliBanditTS
from finite_arm.env_finite import DriftingFiniteArmedBernoulliBandit


def get_config():
  """Generates the config for the experiment."""
  name = 'finite_misspecified'
  n_arm = 3
  true_prior_success = [1, 1, 1]
  informative_prior_failure = [100, 100, 100]
  true_prior_failure = [50, 100, 200]

  def _correct_ts_init(n_arm):
    assert n_arm == 3  # adhoc method for this experiment
    agent = FiniteBernoulliBanditTS(n_arm)
    agent.set_prior(true_prior_success, informative_prior_failure)
    return agent

  agents = collections.OrderedDict(
      [('correct_ts',
        functools.partial(_correct_ts_init, n_arm)),
       ('misspecified_ts',
        functools.partial(FiniteBernoulliBanditTS, n_arm))]
  )

  def _env_init(n_arm):
    environment = DriftingFiniteArmedBernoulliBandit(n_arm, gamma=0.0)
    environment.set_prior(true_prior_success, true_prior_failure)
    return environment

  environments = collections.OrderedDict(
      [('env', functools.partial(_env_init, n_arm))]
  )
  experiments = collections.OrderedDict(
      [(name, ExperimentWithMean)]
  )
  n_steps = 1000
  n_seeds = 10000
  config = Config(name, agents, environments, experiments, n_steps, n_seeds)
  return config


