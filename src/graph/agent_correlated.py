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
import scipy.linalg as spla

from collections import defaultdict
from base.agent import Agent
from graph.env_graph_bandit import CorrelatedBinomialBridge

_SMALL_NUMBER = 1e-10
###############################################################################
# Helper functions for correlated agents

def _prepare_posterior_update_elements(observation, action, reward, num_edges, \
                                       edge2index, sigma_tilde, internal_env):
  """Generates the concentration matrix used in posterior updating in Correlated
      BB problem.
      Inputs:
          observation - number of observations (= n_stages)
          action - the action taken, which is a path
          reward - observed reward for each edge, a dict of dict
          num_edges - number of total edges in the BB
          edge2index - a dict of dicts mapping each edge to a unique index
          sigma_tilde - observation noise power
          internal_env - the internal BB environment of the agent

      Return:
          a vector to be used in updating the mean vector, and a concentration
          matrix to be used when updating covariance matrix and the mean vector
  """
  # generating the local concentration matrix and log-rewards for each edge
  log_rewards = np.zeros(num_edges)
  local_concentration = np.zeros((observation, observation))
  first_edge_counter = 0
  for start_node in reward:
    for end_node in reward[start_node]:
      log_rewards[edge2index[start_node][end_node]] = \
        np.log(reward[start_node][end_node])
      secod_edge_counter = 0
      for another_start_node in reward:
        for another_end_node in reward[another_start_node]:
          if first_edge_counter == secod_edge_counter:
            local_concentration[first_edge_counter,secod_edge_counter] \
              = sigma_tilde ** 2
          elif internal_env.is_in_lower_half(start_node, end_node) \
            == internal_env.is_in_lower_half(another_start_node, another_end_node):
            local_concentration[first_edge_counter, secod_edge_counter] \
              = 2 * (sigma_tilde ** 2) / 3
          else:
            local_concentration[first_edge_counter, secod_edge_counter] \
                      = (sigma_tilde ** 2) / 3
          secod_edge_counter += 1
      first_edge_counter += 1

  # finding the inverse of local concentration matrix
  local_concentration_inv = npla.inv(local_concentration)

  # Generating the concentration matrix
  concentration = np.zeros((num_edges, num_edges))
  first_edge_counter = 0
  for start_node in reward:
    for end_node in reward[start_node]:
      secod_edge_counter = 0
      for another_start_node in reward:
        for another_end_node in reward[another_start_node]:
          concentration[edge2index[start_node][end_node] \
                        ,edge2index[another_start_node][another_end_node]] \
          = local_concentration_inv[first_edge_counter,secod_edge_counter]
          secod_edge_counter += 1
      first_edge_counter += 1

  return log_rewards, concentration


def _update_posterior(posterior, log_rewards, concentration):
  """Updates the posterior parameters of the correlated BB problem.

      Input:
          posterior - current posterior parameters in the form of (Mu, Sigma,
          Sigmainv)
          log_rewards - log of the delays observed for each traversed edge
          concentration - a concentration matrix computed based on the new
          observation

      Return:
          updated parameters: Mu, Sigma, Sigmainv
  """

  new_Sigma_inv = posterior[2] + concentration
  new_Sigma = npla.inv(new_Sigma_inv)
  new_Mu = new_Sigma.dot(posterior[2].dot(posterior[0]) +
                         concentration.dot(log_rewards))

  return new_Mu, new_Sigma, new_Sigma_inv


def _find_conditional_parameters(dim, S):
  """given a dim-dimensional covariance matrix S, returns a
      list containing the elements used for computing the condtional
      distribution of each of the components."""
  Sig12Sig22inv = []
  cond_var = []

  for e in range(dim):
    S11 = copy.copy(S[e][e])
    S12 = S[e][:]
    S12 = np.delete(S12, e)
    S21 = S[e][:]
    S21 = np.delete(S21, e)
    S22 = S[:][:]
    S22 = np.delete(S22, e, 0)
    S22 = np.delete(S22, e, 1)
    S22inv = npla.inv(S22)
    S12S22inv = S12.dot(S22inv)
    Sig12Sig22inv.append(S12S22inv)
    cond_var.append(S11 - S12S22inv.dot(S21))

  return cond_var, Sig12Sig22inv


##############################################################################


class CorrelatedBBTS(Agent):
  """Correlated Binomial Bridge Thompson Sampling"""

  def __init__(self, n_stages, mu0, sigma0, sigma_tilde, n_sweeps=10):
    """An agent for graph bandits.

    Args:
      n_stages - number of stages of the binomial bridge (must be even)
      mu0 - prior mean
      sigma0 - prior stddev
      sigma_tilde - noise on observation
      n_sweeps - number of sweeps, used only in Gibbs sampling
    """
    assert (n_stages % 2 == 0)
    self.n_stages = n_stages
    self.n_sweeps = n_sweeps

    # Set up the internal environment with arbitrary initial values
    self.internal_env = CorrelatedBinomialBridge(n_stages, mu0, sigma0)

    # Save a map (start_node,end_node)-->R to fascilitate calculations
    self.edge2index = defaultdict(dict)
    self.index2edge = defaultdict(dict)
    edge_counter = 0
    for start_node in self.internal_env.graph:
      for end_node in self.internal_env.graph[start_node]:
        self.edge2index[start_node][end_node] = edge_counter
        self.index2edge[edge_counter] = (start_node, end_node)
        edge_counter += 1

    # saving the number of total edges
    self.num_edges = edge_counter

    # prior parameters
    self.Mu0 = np.array([mu0] * self.num_edges)
    self.Sigma0 = np.diag([sigma0**2] * self.num_edges)
    self.Sigma0inv = np.diag([(1 / sigma0)**2] * self.num_edges)
    self.sigma_tilde = sigma_tilde

    # posterior distribution is saved as a triple containing the mean vector,
    # covariance matrix and its inverse
    self.posterior = (self.Mu0, self.Sigma0, self.Sigma0inv)

    # additional parameters used in the bootstrap version
    self.concentration_history = []
    self.log_reward_history = []
    self.history_size = 0

  def get_posterior_mean(self):
    """Gets the posterior mean for each edge.

    Return:
      edge_length - dict of dicts edge_length[start_node][end_node] = distance
    """
    edge_length = copy.deepcopy(self.internal_env.graph)

    for start_node in edge_length:
      for end_node in edge_length[start_node]:
        edge_index = self.edge2index[start_node][end_node]
        mean = self.posterior[0][edge_index]
        var = self.posterior[0][edge_index, edge_index]
        edge_length[start_node][end_node] = np.exp(mean + 0.5 * var)

    return edge_length

  def get_posterior_sample(self):
    """Gets a posterior sample for each edge

    Return:
      edge_length - dict of dicts edge_length[start_node][end_node] = distance
    """
    # flattened sample
    flattened_sample = np.random.multivariate_normal(self.posterior[0],
                                                     self.posterior[1])

    edge_length = copy.deepcopy(self.internal_env.graph)

    for start_node in edge_length:
      for end_node in edge_length[start_node]:
        edge_length[start_node][end_node] = \
            np.exp(flattened_sample[self.edge2index[start_node][end_node]])

    return edge_length

  def update_observation(self, observation, action, reward):
    """Updates observations for binomial bridge

    Args:
      observation - number of stages
      action - path chosen by the agent (not used)
      reward - dict of dict reward[start_node][end_node] = stochastic_time
    """
    assert (observation == self.n_stages)

    log_rewards, concentration = _prepare_posterior_update_elements(observation,\
            action, reward, self.num_edges, self.edge2index, self.sigma_tilde, \
            self.internal_env)

    # updating mean and ovariance matrix of the joint distribution
    new_Mu, new_Sigma, new_Sigma_inv = _update_posterior(self.posterior, \
                                                log_rewards, concentration)
    self.posterior = (new_Mu, new_Sigma, new_Sigma_inv)

  def pick_action(self, observation):
    """Greedy shortest path wrt posterior sample."""
    posterior_sample = self.get_posterior_sample()
    self.internal_env.overwrite_edge_length(posterior_sample)
    path = self.internal_env.get_shortest_path()

    return path


##############################################################################


class GibbsCorrelatedBB(CorrelatedBBTS):
  """Correlated Binomial Bridge Gibbs sampling method"""

  def get_sample(self):
    """Gets a Gibbs sample for each edge

    Return:
      edge_length - dict of dicts edge_length[start_node][end_node] = distance
    """
    # getting conditional variance of each component and a multiplier used in
    # finding the conditional mean
    cond_var, multiplier = _find_conditional_parameters(self.num_edges, \
                                                       self.posterior[1])

    # sweeping and updating the sample
    flattened_sample = copy.copy(self.posterior[0])
    for i in range(self.n_sweeps):
      for e in range(self.num_edges):
        others_values = np.delete(flattened_sample[:], e)
        others_mean = np.delete(self.posterior[0][:], e)
        our_cond_mean = self.posterior[0][e] + multiplier[e].dot(
            others_values - others_mean)
        flattened_sample[
            e] = our_cond_mean + np.sqrt(cond_var[e]) * np.random.randn()

    edge_length = copy.deepcopy(self.internal_env.graph)

    for start_node in edge_length:
      for end_node in edge_length[start_node]:
        edge_length[start_node][end_node] = \
            np.exp(flattened_sample[self.edge2index[start_node][end_node]])

    return edge_length

  def pick_action(self, observation):
    """Greedy shortest path wrt Gibbs sample."""
    bootstrap_sample = self.get_sample()
    self.internal_env.overwrite_edge_length(bootstrap_sample)
    path = self.internal_env.get_shortest_path()

    return path


##############################################################################


class BootstrapCorrelatedBB(CorrelatedBBTS):
  """Correlated Binomial Bridge Bootstrap method"""

  def get_sample(self):
    """Gets a bootstrap sample for each edge

    Return:
      edge_length - dict of dicts edge_length[start_node][end_node] = distance
    """
    Mu = copy.copy(np.random.multivariate_normal(self.Mu0, self.Sigma0))
    Sigma = copy.copy(self.Sigma0)
    Sigmainv = copy.copy(self.Sigma0inv)

    # resampling the history and updating the parameters
    if self.history_size > 0:
      random_indices = np.random.randint(0, self.history_size,
                                         self.history_size)
      for ind in random_indices:
        Mu, Sigma, Sigmainv = _update_posterior((Mu,Sigma,Sigmainv), \
           self.log_reward_history[ind], self.concentration_history[ind])

    # flattened sample
    flattened_sample = Mu

    edge_length = copy.deepcopy(self.internal_env.graph)

    for start_node in edge_length:
      for end_node in edge_length[start_node]:
        edge_length[start_node][end_node] = \
            np.exp(flattened_sample[self.edge2index[start_node][end_node]])

    return edge_length

  def update_observation(self, observation, action, reward):
    """Updates observations for binomial bridge

    Args:
      observation - number of stages
      action - path chosen by the agent (not used)
      reward - dict of dict reward[start_node][end_node] = stochastic_time
    """
    assert (observation == self.n_stages)

    log_rewards, concentration = _prepare_posterior_update_elements(observation,\
            action, reward, self.num_edges, self.edge2index, self.sigma_tilde, \
            self.internal_env)

    # updating the history
    self.log_reward_history.append(log_rewards)
    self.concentration_history.append(concentration)
    self.history_size += 1

  def pick_action(self, observation):
    """Greedy shortest path wrt bootstrap sample."""
    bootstrap_sample = self.get_sample()
    self.internal_env.overwrite_edge_length(bootstrap_sample)
    path = self.internal_env.get_shortest_path()

    return path

##############################################################################
class CorrelatedBBLangevin(CorrelatedBBTS):
  '''Correlated Binomial Bridge, Langevin Method.'''
  def __init__(self,
               n_stages,
               mu0,
               sigma0,
               sigma_tilde,
               step_count=200,
               step_size=.01):
    CorrelatedBBTS.__init__(self, n_stages, mu0, sigma0, sigma_tilde)
    self.step_count = step_count
    self.step_size = step_size
    
  def get_sample(self):
    '''generates a sample based on the Langevin method.'''
    mu = self.posterior[0]
    Sigma = self.posterior[1]
    SigmaInv = self.posterior[2]
    dim = len(mu)
    
    preconditioner = Sigma
    preconditioner_sqrt=spla.sqrtm(preconditioner)
    #Remove any complex component in preconditioner_sqrt arising from numerical error
    complex_part=np.imag(preconditioner)
    if (spla.norm(complex_part)> _SMALL_NUMBER):
        print("Warning. There may be numerical issues.  Preconditioner has complex values")
        print("Norm of the imaginary component is, ")+str(spla.norm(complex_part))
    preconditioner_sqrt=np.real(preconditioner_sqrt)
    
    x = mu
    for i in range(self.step_count):
      g = -SigmaInv.dot(x-mu) # posterior gradient at current point
      scaled_grad=preconditioner.dot(g)
      scaled_noise= preconditioner_sqrt.dot(np.random.randn(dim))
      x = x + self.step_size*scaled_grad + np.sqrt(2*self.step_size)*scaled_noise
      
    # mapping and reformatting the final point to a desired sample
    edge_length = copy.deepcopy(self.internal_env.graph)

    for start_node in edge_length:
      for end_node in edge_length[start_node]:
        edge_length[start_node][end_node] = \
            np.exp(x[self.edge2index[start_node][end_node]])

    return edge_length
      
  def pick_action(self, observation):
    """Greedy shortest path wrt Langevin sample."""
    langevin_sample = self.get_sample()
    self.internal_env.overwrite_edge_length(langevin_sample)
    path = self.internal_env.get_shortest_path()

    return path
    