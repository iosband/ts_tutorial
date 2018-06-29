"""Dynamic pricing agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import numpy.linalg as npla

from base.agent import Agent
from base.agent import random_argmax

##############################################################################


class TSAssortment(Agent):
  """TS agent for product assortment."""

  def __init__(self, num_products, prior_mean, prior_var_diagonal,prior_var_off_diagonal, noise_var, profits, epsilon=0.06, k = 9):
    '''Inputs:
      num_products - number of products.
      prior_mean - prior mean of the entries
      prior_var_diagonal - prior variance of diagonal entries
      prior_var_off_diagonal - prior variance of off-diagonal entries
      nois_var - uncertainty in the demand 
      profits - the profit of selling each of the products
      epsilon - used in epsilon-greedy agent
      k - constant used in annealing epsilon greedy.'''
    self.num_products = num_products
    self.prior_mean = prior_mean
    self.prior_var_diagonal = prior_var_diagonal
    self.prior_var_off_diagonal = prior_var_off_diagonal
    self.noise_var = noise_var
    self.profits = profits
    self.epsilon = epsilon
    self.period = 0
    self.k = k
    
    self.posterior_mean = np.array([prior_mean]*(num_products**2))
    vars_off_diagonal = prior_var_off_diagonal*np.ones((num_products,num_products))
    vars_off_diagonal = vars_off_diagonal - np.diag(np.diag(vars_off_diagonal))
    vars_diagonal = np.diag([prior_var_diagonal]*num_products)
    vars_all = vars_off_diagonal + vars_diagonal
    self.posterior_cov = np.diag(np.reshape(vars_all.T,num_products**2))
    self.posterior_cov_inv = npla.inv(self.posterior_cov)
  
  def _get_selection_matrix(self,assortment):
    '''given an assortment, returns a selection matrix which is useful when
    updating the posterior parameters.'''
    support_size = np.sum(assortment)
    support = np.nonzero(assortment)[0]
    selection_matrix = np.zeros((support_size,assortment.size))
    for i in range(support_size):
      selection_matrix[i][support[i]] = 1
      
    return selection_matrix
  
  def update_observation(self, observation, assortment, random_demand):
    """Updates observations for binomial bridge

    Args:
      assortment - selected assortment
      random_demand - random demand observed for the products
    """
    assert observation == self.num_products
    self.period += 1
    
    if np.sum(assortment)>0:
      log_demand = np.log(random_demand)    
      S = self._get_selection_matrix(assortment)
      X = np.kron(assortment.T,S)
      new_cov_inv = self.posterior_cov_inv + X.T.dot(X)/self.noise_var
      self.posterior_cov = npla.inv(new_cov_inv)
      self.posterior_mean = self.posterior_cov.dot(self.posterior_cov_inv.dot(self.posterior_mean)+\
                                                   X.T.dot(log_demand)/self.noise_var)
      self.posterior_cov_inv = new_cov_inv
 
  def get_posterior_mean(self):
    return self.posterior_mean
  
  def get_posterior_sample(self):
    cov_square_root = npla.cholesky(self.posterior_cov)
    raw_sample = np.random.randn(self.num_products**2)
    sample = self.posterior_mean + cov_square_root.dot(raw_sample)
    return sample
  
  def find_optimal_assortment(self, theta_hat):
    '''finds the optimal assortment, given a sampled parameter.'''
    # generating all possible assortments
    assortment_tuples = list(itertools.product([0, 1], repeat=self.num_products))
    total_profit = []
    for assortment in assortment_tuples:
      expected_demand = np.array(assortment)*np.exp(self.noise_var/2 + 
                                theta_hat.dot(np.array(assortment)))
      total_profit.append(expected_demand.dot(self.profits))
    optimal_ind = random_argmax(np.array(total_profit))
    return np.array(assortment_tuples[optimal_ind])
  
  def pick_action(self, observation):
    """Take an action based on a posterior sample."""
    theta_flattened = self.get_posterior_sample()
    theta_hat = np.reshape(theta_flattened,
                           (self.num_products,self.num_products),order='F')
    assortment = self.find_optimal_assortment(theta_hat)
    return assortment

  
class GreedyAssortment(TSAssortment):
  """Greedy agent for product assortment."""
  def pick_action(self, observation):
    """Take an action based on posterior mean."""
    theta_flattened = self.get_posterior_mean()
    theta_hat = np.reshape(theta_flattened,
                           (self.num_products,self.num_products),order='F')
    assortment = self.find_optimal_assortment(theta_hat)
    return assortment
  
class EpsilonGreedyAssortment(TSAssortment):
  """Epsilon Greedy agent for product assortment."""
  def pick_action(self, observation):
    play_type = np.random.uniform(0,1,1)
    if play_type<=self.epsilon:
      """Take a random action."""
      all_actions = list(itertools.product([0, 1], repeat=self.num_products))
      rand_action = np.random.randint(1,len(all_actions))
      assortment = np.array(all_actions[rand_action])
    else:
      """Take an action based on posterior mean."""
      theta_flattened = self.get_posterior_mean()
      theta_hat = np.reshape(theta_flattened,
                             (self.num_products,self.num_products),order='F')
      assortment = self.find_optimal_assortment(theta_hat)
    return assortment

class AnnealingEpsilonGreedyAssortment(TSAssortment):
  """Epsilon Greedy agent for product assortment with annealing epsilon."""
  def pick_action(self, observation):
    epsilon = self.k/(self.period+self.k)
    play_type = np.random.uniform(0,1,1)
    if play_type <= epsilon:
      """Take a random action."""
      all_actions = list(itertools.product([0, 1], repeat=self.num_products))
      rand_action = np.random.randint(1,len(all_actions))
      assortment = np.array(all_actions[rand_action])
    else:
      """Take an action based on posterior mean."""
      theta_flattened = self.get_posterior_mean()
      theta_hat = np.reshape(theta_flattened,
                             (self.num_products,self.num_products),order='F')
      assortment = self.find_optimal_assortment(theta_hat)
    return assortment