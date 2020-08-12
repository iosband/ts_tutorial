"""Specify the jobs to run via config file.

Binomial bridge bandit experiment.
Binomial bridge with only binary reward at the end --> no conjugate update.
See Figure 9 https://arxiv.org/pdf/1707.02038.pdf
"""

import collections
import functools

from base.config_lib import Config
from base.experiment import ExperimentNoAction
from graph.agent_indep_binary import BootstrapIndependentBBWithBinaryReward
from graph.agent_indep_binary import LaplaceIndependentBBWithBinaryReward
from graph.agent_indep_binary import StochasticLangevinMCMCIndependentBBWithBinaryReward
from graph.agent_indep_binary import EpsilonGreedyIndependentBBWithBinaryReward
from graph.env_graph_bandit import IndependentBinomialBridgeWithBinaryReward


def get_config():
  """Generates the config for the experiment."""
  name = 'graph_indep_binary_new'
  n_stages = 20
  shape = 2
  scale = 0.5
  tol = 0.001
  alpha = 0.2
  beta = 0.5
  langevin_batch_size = 100
  langevin_step_count = 200
  langevin_step_size = 0.0005
  epsilon = 0
  
  agents = collections.OrderedDict(
      [('Langevin TS',
        functools.partial(EpsilonGreedyIndependentBBWithBinaryReward,
                          n_stages, epsilon, shape, scale, tol, alpha, beta))])
  
  
#  agents = collections.OrderedDict(
#      [('Langevin TS',
#        functools.partial(StochasticLangevinMCMCIndependentBBWithBinaryReward,
#                          n_stages, shape, scale, tol, alpha, beta, langevin_batch_size,
#                          langevin_step_count, langevin_step_size)),
#       ('bootstrap TS',
#        functools.partial(BootstrapIndependentBBWithBinaryReward,
#                          n_stages, shape, scale, tol, alpha, beta)),
#       ('Laplace TS',
#        functools.partial(LaplaceIndependentBBWithBinaryReward,
#                          n_stages, shape, scale, tol, alpha, beta))]
#  )
       
  environments = collections.OrderedDict(
      [('env',
        functools.partial(IndependentBinomialBridgeWithBinaryReward,
                          n_stages, shape, scale))]
  )
  experiments = collections.OrderedDict(
      [(name, ExperimentNoAction)]
  )
  n_steps = 500
  n_seeds = 1000
  config = Config(name, agents, environments, experiments, n_steps, n_seeds)
  return config
