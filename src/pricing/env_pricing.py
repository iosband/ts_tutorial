"""Dynamic pricing bandit environments."""

from __future__ import division
from __future__ import print_function

import numpy as np

from cvxpy import *

from base.environment import Environment

##############################################################################


class DynamicPricing(Environment):
  """An environment with linear demand."""

  def __init__(self, num_products, scale, noise_var, p_max):
    '''Inputs:
      num_products - number of products to be priced.
      scale - diagonal entry of the scale matrix of Wishart prior
      nois_var - uncertainty in the demand (sigma^2 in the paper)
      p_max - maximum price.'''
    self.num_products = num_products
    self.scale = scale
    self.noise_var = noise_var
    self.p_max = p_max
    self.scale_matrix = np.diag([scale]*num_products)
    
    # setting the intercept term so that the expected demand is 
    # positive with high probability
    self.intercept = p_max*((num_products-1)+np.sqrt(2)-1)*scale\
                                                *np.ones(num_products)
    
    # generating theta from the Wishart prior
    temp = np.random.normal(0,np.sqrt(scale),\
                            size = (num_products+2,num_products))
    self.theta = np.matmul(temp.T,temp)
    
    self.optimal_price = np.zeros(num_products)
    self.optimal_demand = np.zeros(num_products)
    self.optimal_revenue = 0
    self._find_optimal_price()

  def _find_optimal_price(self):
    '''finds the optimal price for the product.'''
    x = Variable(self.num_products)
    objective = Maximize(self.intercept.T*x-quad_form(x,self.theta))
    constraints = [0 <= x, x <= self.p_max]
    prob = Problem(objective, constraints)
    result = prob.solve()
    
    self.optimal_price = np.array(x.value).flatten()
    self.optimal_demand = self.intercept - self.theta.dot(self.optimal_price)
    self.optimal_revenue = self.intercept.T.dot(self.optimal_price) - \
                  self.optimal_price.T.dot(self.theta.dot(self.optimal_price))
  def get_observation(self):
    return self.num_products

  def get_optimal_reward(self):
    return self.optimal_revenue

  def get_expected_reward(self, price):
    return self.intercept.T.dot(price) - price.T.dot(self.theta.dot(price))
  
  def get_expected_demand(self, price):
    return self.intercept - self.theta.dot(price)
  
  def get_stochastic_reward(self, price):
    mean = self.get_expected_demand(price) - self.noise_var/2
    cov = np.diag([self.noise_var]*self.num_products)
    return np.exp(np.random.multivariate_normal(mean,cov))
  
