"""Specify the jobs to run via config file.

Binomial bridge bandit experiment with correlated edges.
See Figure 7 https://arxiv.org/pdf/1707.02038.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

from base.config_lib import Config
from base.experiment import ExperimentNoAction
from graph.agent_correlated import BootstrapCorrelatedBB
from graph.agent_correlated import CorrelatedBBTS
from graph.agent_indep import IndependentBBTS
from graph.env_graph_bandit import CorrelatedBinomialBridge


def get_config():
  """Generates the config for the experiment."""
  name = 'graph_correlated'
  n_stages = 20
  mu0 = -0.5
  sigma0 = 1
  sigma_tilde = 1

  agents = collections.OrderedDict(
      [('thoughtful',
        functools.partial(CorrelatedBBTS,
                          n_stages, mu0, sigma0, sigma_tilde)),
       ('misspecified_ts',
        functools.partial(IndependentBBTS,
                          n_stages, mu0, sigma0, sigma_tilde)),
       ('bootstrap',
        functools.partial(BootstrapCorrelatedBB,
                          n_stages, mu0, sigma0, sigma_tilde))]
  )

  environments = collections.OrderedDict(
      [('env',
        functools.partial(CorrelatedBinomialBridge,
                          n_stages, mu0, sigma0, sigma_tilde))]
  )
  experiments = collections.OrderedDict(
      [(name, ExperimentNoAction)]
  )
  n_steps = 500
  n_seeds = 1000
  config = Config(name, agents, environments, experiments, n_steps, n_seeds)
  return config
