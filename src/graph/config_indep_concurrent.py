"""Specify the jobs to run via config file.

Binomial bridge bandit experiment with independent segments.
Evaluates the performance of concurrent Thompson sampling with multiple agents.
See Figure 14 https://arxiv.org/pdf/1707.02038.pdf
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

from base.config_lib import Config
from base.experiment import ExperimentMultipleAgents
from graph.agent_indep import IndependentBBMultipleTS
from graph.env_graph_bandit import MultiAgentCorrelatedBinomialBridge


def get_config():
  """Generates the config for the experiment."""
  name = 'graph_indep_concurrent'
  n_stages = 20
  mu0 = -0.5
  sigma0 = 1
  sigma_tilde = 1
  num_agents = [1,10,20,50,100]
  
  agents_list = []
  for num_agent in num_agents:
    agents_list.append(('K = '+str(num_agent),
        functools.partial(IndependentBBMultipleTS,
                          n_stages, mu0, sigma0, sigma_tilde,num_agent)))
  
  agents = collections.OrderedDict(agents_list)
  
  
  environments = collections.OrderedDict(
      [('env',
        functools.partial(MultiAgentCorrelatedBinomialBridge,
                          n_stages, mu0, sigma0, sigma_tilde))]
  )
  experiments = collections.OrderedDict(
      [(name, ExperimentMultipleAgents)]
  )
  n_steps = 100
  n_seeds = 1000
  config = Config(name, agents, environments, experiments, n_steps, n_seeds)
  return config

