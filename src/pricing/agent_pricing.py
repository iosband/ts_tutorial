"""Dynamic pricing agent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.linalg as npla

from cvxpy import *

from base.agent import Agent
from base.agent import random_argmax

##############################################################################


class BootstrapDynamicPricing(Agent):
  """TS agent for dynamic pricing."""

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
                                                
    self.price_history = []
    self.observation_history = []
    self.history_size = 0
    
    self.resampled_price_history = []
    self.resampled_observation_history = []
    
  def update_observation(self, observation, price, random_demand):
    """Updates observations for binomial bridge

    Args:
      price - price of the items
      path - random demand observed for the items
    """
    assert observation == self.num_products
    
    self.price_history.append(price)
    
    # instead of the demands, we save the following transformation which makes
    # computations easier
    self.observation_history.append(np.log(random_demand) - self.intercept + \
                                    self.noise_var/2)
    self.history_size += 1

  def _resample_history(self):
    """Generates a resampled version of the history."""

    if self.history_size > 0:
      random_indices = np.random.randint(0, self.history_size,
                                         self.history_size)
      self.resampled_price_history = [self.price_history[i] \
                                                for i in random_indices]
      self.resampled_observation_history = [self.observation_history[i] \
                                                   for i in random_indices]
    else:
      self.resampled_price_history = self.price_history
      self.resampled_observation_history = self.observation_history
        
  def _generate_prior_sample(self):
    '''generates a sample from the Wishart prior.'''
    temp = np.random.normal(0,np.sqrt(self.scale),\
                            size = (self.num_products+2,self.num_products))
    return np.matmul(temp.T,temp)
  
  def project_to_SD_cone(self,X):
    '''Although we are using SDP packages to find the bootstrap sample for 
    theta, numerical errors may cause the sample not to be semidefinite. This 
    function projects the samples onto the positive semidefinite cone to make 
    sure subsequent optimization problem is convex. '''
    X = 0.5*(X+X.T)
    eig_values,eig_vectors = npla.eigh(X)
    ok = np.all([eig_value >=0 for eig_value in eig_values])
    if not ok:
      new_eig_values = np.diag([max(0,eig_value) for eig_value in eig_values])
      return eig_vectors.dot(new_eig_values.dot(eig_vectors.T))
    else:
      return X
      
  def generate_bootstrap_sample(self):
    '''generates bootstrap sample.'''
    
    V = self._generate_prior_sample()
    
    if self.history_size==0:
      return V
    else:
      Vinv = npla.inv(V)
      self._resample_history()
      
      X = Semidef(self.num_products)
      problems = []
      cost = -log_det(X)/2 + trace(Vinv*X)/2
      problems = [Problem(Minimize(cost),[])]
      for i in range(self.history_size):
        cost_new = sum_squares(self.resampled_observation_history[i]+\
                        X*self.resampled_price_history[i])/(2*self.noise_var)
        problems.append(Problem(Minimize(cost_new),[]))
      problem = sum(problems)
      problem.solve(solver=SCS)
      X_opt = X.value
      return self.project_to_SD_cone(X_opt)
    
  def find_optimal_price(self, theta_hat):
    '''finds the optimal price, given a sampled parameter.'''
    x = Variable(self.num_products)
    objective = Maximize(self.intercept.T*x-quad_form(x,theta_hat))
    constraints = [0 <= x, x <= self.p_max]
    prob = Problem(objective, constraints)
    result = prob.solve()
    return np.array(x.value).flatten()
  
  def pick_action(self, observation):
    """Take an action based on the bootstrap sample."""
    theta_hat = self.generate_bootstrap_sample()
    price = self.find_optimal_price(theta_hat)
    return price