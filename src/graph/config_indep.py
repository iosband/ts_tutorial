"""Specify the jobs to run via config file.

Binomial bridge bandit experiment with independent segments.
Compare the performance of Thompson sampling with different egreedy.
See Figure 7 https://arxiv.org/pdf/1707.02038.pdf
"""

import collections
import functools

from base.config_lib import Config
from base.experiment import ExperimentNoAction
from graph.agent_indep import IndependentBBEpsilonGreedy
from graph.agent_indep import IndependentBBTS
from graph.env_graph_bandit import IndependentBinomialBridge


def get_config():
  """Generates the config for the experiment."""
  name = 'graph_indep'
  n_stages = 20
  mu0 = -0.5
  sigma0 = 1
  sigma_tilde = 1

  agents = collections.OrderedDict(
      [('TS',
        functools.partial(IndependentBBTS,
                          n_stages, mu0, sigma0, sigma_tilde)),
       ('greedy',
        functools.partial(IndependentBBEpsilonGreedy,
                          n_stages, mu0, sigma0, sigma_tilde, epsilon=0.0)),
       ('0.01-greedy',
        functools.partial(IndependentBBEpsilonGreedy,
                          n_stages, mu0, sigma0, sigma_tilde, epsilon=0.01)),
       ('0.05-greedy',
        functools.partial(IndependentBBEpsilonGreedy,
                          n_stages, mu0, sigma0, sigma_tilde, epsilon=0.05)),
       ('0.1-greedy',
        functools.partial(IndependentBBEpsilonGreedy,
                          n_stages, mu0, sigma0, sigma_tilde, epsilon=0.1))]
  )

  environments = collections.OrderedDict(
      [('env',
        functools.partial(IndependentBinomialBridge,
                          n_stages, mu0, sigma0, sigma_tilde))]
  )
  experiments = collections.OrderedDict(
      [(name, ExperimentNoAction)]
  )
  n_steps = 500
  n_seeds = 1000
  config = Config(name, agents, environments, experiments, n_steps, n_seeds)
  return config

