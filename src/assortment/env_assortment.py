"""dynamic pricing bandit environments."""

from __future__ import division
from __future__ import print_function

import itertools
import numpy as np

from base.environment import Environment
from base.agent import random_argmax
##############################################################################


class ProductAssortment(Environment):
  """An environment for product assortment."""

  def __init__(self, num_products, prior_mean, prior_var_diagonal, prior_var_off_diagonal, noise_var, profits):
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
    
    theta_off_diagonal = np.diag([prior_mean]*num_products) + \
                          np.random.normal(0,prior_var_off_diagonal,
                                  size=(num_products,num_products))
    theta_off_diagonal = theta_off_diagonal - np.diag(np.diag(theta_off_diagonal))
    theta_diagonal = np.diag([prior_mean]*num_products) +\
        np.random.normal(0,prior_var_diagonal, size=(num_products,num_products))
    self.theta = theta_diagonal + theta_off_diagonal
    
    self.optimal_assortment = np.zeros(num_products)
    self.optimal_profit = 0
    
    self._find_optimal_assortment()
    
  def _find_optimal_assortment(self):
    '''finds the optimal assortment of the products.'''
    
    # generating all possible assortments
    assortment_tuples = list(itertools.product([0, 1], repeat=self.num_products))
    total_profit = []
    for assortment in assortment_tuples:
      expected_demand = np.array(assortment)*np.exp(self.noise_var/2 
                                + self.theta.dot(np.array(assortment)))
      total_profit.append(expected_demand.dot(self.profits))
    optimal_ind = random_argmax(np.array(total_profit))
    self.optimal_assortment = np.array(assortment_tuples[optimal_ind])
    self.optimal_profit = total_profit[optimal_ind]
        
  def get_observation(self):
    return self.num_products

  def get_optimal_reward(self):
    return self.optimal_profit

  def get_expected_reward(self, assortment):
    expected_demand = assortment*np.exp(self.noise_var/2 + self.theta.dot(assortment))
    return expected_demand.dot(self.profits)
  
  def get_stochastic_reward(self, assortment):
    random_demand = assortment*np.exp(self.theta.dot(assortment) + np.sqrt(self.noise_var)*\
                  np.random.normal(0,1,self.num_products))
    support = np.nonzero(random_demand)[0]
    return random_demand[support]
