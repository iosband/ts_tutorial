"""Specify the jobs to run via config file.

Binomial bridge bandit experiment.
Comparison of the performance of approximate posterior sampling methods
in a simple domain where we *can* sample the true posterior.
See Figure 10(b) https://arxiv.org/pdf/1707.02038.pdf
"""

import collections
import functools

from base.config_lib import Config
from base.experiment import ExperimentNoAction
from graph.agent_correlated import BootstrapCorrelatedBB
from graph.agent_correlated import CorrelatedBBTS
from graph.agent_correlated import GibbsCorrelatedBB
from graph.env_graph_bandit import CorrelatedBinomialBridge
from graph.agent_correlated import CorrelatedBBLangevin

def get_config():
  """Generates the config for the experiment."""
  name = 'graph_correlated_sanity'
  n_stages = 20
  mu0 = -0.5
  sigma0 = 1
  sigma_tilde = 1
  step_count = 100
  step_size = 0.01
  
  agents = collections.OrderedDict(
      [('Langevin TS',
        functools.partial(CorrelatedBBLangevin,
                          n_stages, mu0, sigma0, sigma_tilde,step_count,step_size)),
       ('TS',
        functools.partial(CorrelatedBBTS,
                          n_stages, mu0, sigma0, sigma_tilde)),
       ('Gibbs TS',
        functools.partial(GibbsCorrelatedBB,
                          n_stages, mu0, sigma0, sigma_tilde)),
       ('bootstrap TS',
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

