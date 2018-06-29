"""Agents for graph bandit problems.

These agents became too large to comfortably keep in one file, so we divided
it up into three separate sections for:
- Independent Binomial Bridge
- Correlated Binomial Bridge
- Independent Binomial Bridge with Binary Reward
"""

from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import numpy.linalg as npla
import random

from collections import defaultdict
from base.agent import Agent
from graph.env_graph_bandit import IndependentBinomialBridge

##############################################################################


class IndependentBBEpsilonGreedy(Agent):
  """Independent Binomial Bridge Epsilon Greedy"""

  def __init__(self, n_stages, mu0, sigma0, sigma_tilde, epsilon=0.0):
    """An agent for graph bandits.

    Args:
      n_stages - number of stages of the binomial bridge (must be even)
      mu0 - prior mean
      sigma0 - prior stddev
      sigma_tilde - noise on observation
      epsilon - probability of random path selection
    """
    assert (n_stages % 2 == 0)
    self.n_stages = n_stages
    self.mu0 = mu0
    self.sigma0 = sigma0
    self.sigma_tilde = sigma_tilde
    self.epsilon = epsilon

    # Set up the internal environment with arbitrary initial values
    self.internal_env = IndependentBinomialBridge(n_stages, mu0, sigma0)

    # Save the posterior for edges as tuple (mean, std) of posterior belief
    self.posterior = copy.deepcopy(self.internal_env.graph)
    for start_node in self.posterior:
      for end_node in self.posterior[start_node]:
        self.posterior[start_node][end_node] = (mu0, sigma0)

  def get_posterior_mean(self):
    """Gets the posterior mean for each edge.

    Returns:
      edge_length - dict of dicts edge_length[start_node][end_node] = distance
    """
    edge_length = copy.deepcopy(self.posterior)

    for start_node in self.posterior:
      for end_node in self.posterior[start_node]:
        mean, std = self.posterior[start_node][end_node]
        edge_length[start_node][end_node] = np.exp(mean + 0.5 * std**2)

    return edge_length

  def get_posterior_sample(self):
    """Gets a posterior sample for each edge.

    Return:
      edge_length - dict of dicts edge_length[start_node][end_node] = distance
    """
    edge_length = copy.deepcopy(self.posterior)

    for start_node in self.posterior:
      for end_node in self.posterior[start_node]:
        mean, std = self.posterior[start_node][end_node]
        edge_length[start_node][end_node] = np.exp(mean +
                                                   std * np.random.randn())

    return edge_length

  def update_observation(self, observation, action, reward):
    """Updates observations for binomial bridge.

    Args:
      observation - number of stages
      action - path chosen by the agent (not used)
      reward - dict of dict reward[start_node][end_node] = stochastic_time
    """
    assert observation == self.n_stages

    for start_node in reward:
      for end_node in reward[start_node]:
        y = reward[start_node][end_node]
        old_mean, old_std = self.posterior[start_node][end_node]

        # convert std into precision for easier algebra
        old_precision = 1. / (old_std**2)
        noise_precision = 1. / (self.sigma_tilde**2)
        new_precision = old_precision + noise_precision

        new_mean = (noise_precision * (np.log(y) + 0.5 / noise_precision) +
                    old_precision * old_mean) / new_precision
        new_std = np.sqrt(1. / new_precision)

        # update the posterior in place
        self.posterior[start_node][end_node] = (new_mean, new_std)

  def _pick_random_path(self):
    """Selects a path completely at random through the bridge."""
    path = []
    start_node = (0, 0)
    while True:
      path += [start_node]
      if start_node == (self.n_stages, 0):
        break
      start_node = random.choice(self.posterior[start_node].keys())
    return path

  def pick_action(self, observation):
    """Greedy path is shortest wrt posterior mean."""
    if np.random.rand() < self.epsilon:
      path = self._pick_random_path()

    else:
      posterior_means = self.get_posterior_mean()
      self.internal_env.overwrite_edge_length(posterior_means)
      path = self.internal_env.get_shortest_path()

    return path


##############################################################################


class IndependentBBTS(IndependentBBEpsilonGreedy):
  """Independent Binomial Bridge Thompson Sampling"""

  def pick_action(self, observation):
    """Greedy shortest path wrt posterior sample."""
    posterior_sample = self.get_posterior_sample()
    self.internal_env.overwrite_edge_length(posterior_sample)
    path = self.internal_env.get_shortest_path()

    return path
  
###############################################################################
class IndependentBBMultipleTS(IndependentBBTS):
  '''Concurrent TS agents.'''
  
  def __init__(self, n_stages, mu0, sigma0, sigma_tilde,num_agents=10):
    IndependentBBTS.__init__(self,n_stages, mu0, sigma0, sigma_tilde,0.0)
    self.num_agents = num_agents
    
  def update_observation(self, observation, action, rewards):
    assert(len(rewards) == self.num_agents)
    
    for reward in rewards:
      IndependentBBTS.update_observation(self, observation, action, reward)
    
  def pick_action(self, observation):
    """Picks TS action for all the agents independently."""
    paths = [IndependentBBTS.pick_action(self, observation) for x in 
             range(self.num_agents)]
    return paths
    
