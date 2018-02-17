"""Specify the jobs to run via config file.

Binomial bridge bandit experiment.
Binomial bridge with only binary reward at the end --> no conjugate update.
See Figure 8 https://arxiv.org/pdf/1707.02038.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

from base.config_lib import Config
from base.experiment import ExperimentNoAction
from graph.agent_indep_binary import BootstrapIndependentBBWithBinaryReward
from graph.agent_indep_binary import LaplaceIndependentBBWithBinaryReward
from graph.agent_indep_binary import LangevinMCMCIndependentBBWithBinaryReward
from graph.env_graph_bandit import IndependentBinomialBridgeWithBinaryReward


def get_config():
  """Generates the config for the experiment."""
  name = 'graph_indep_binary'
  n_stages = 20
  shape = 2
  scale = 0.5
  tol = 0.1
  alpha = 0.2
  beta = 0.5

  agents = collections.OrderedDict(
      [('Bootstrap',
        functools.partial(BootstrapIndependentBBWithBinaryReward,
                          n_stages, shape, scale, tol, alpha, beta)),
       ('Laplace',
        functools.partial(LaplaceIndependentBBWithBinaryReward,
                          n_stages, shape, scale, tol, alpha, beta))]
  )

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
