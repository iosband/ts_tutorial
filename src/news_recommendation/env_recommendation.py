''' News Article Recommendation'''

from __future__ import division
from __future__ import generators
from __future__ import print_function

import numpy as np


from base.environment import Environment

class NewsRecommendation(Environment):
  """News Recommendation Environment. The environment provides the features 
  vectors at any period and determines the rewards of a played action."""

  def __init__(self,num_articles,dim,theta_mean=0,theta_std=1):
    """Args:
      num_articles - number of news articles
      dim - dimension of the problem
      theta_mean - mean of each component of theta
      theta_std - std of each component of theta
      """
    self.num_articles = num_articles
    self.dim = dim
    self.theta_mean = theta_mean
    self.theta_std = theta_std
    
    # generating the true parameters
    self.thetas = [self.theta_mean + self.theta_std*np.random.randn(self.dim) 
                                            for _ in range(self.num_articles)]
    
    # keeping current rewards
    self.current_rewards = [0]*self.num_articles
    
  def get_observation(self):
    '''generates context vector and computes the true
    reward of each article.'''
    
    context = []
    context_vector = np.random.binomial(1,max(0,1/(self.dim-1)),self.dim)
    context_vector[0] = 1        
    for i in range(self.num_articles):
      context.append(context_vector)
      self.current_rewards[i] = 1/(1+np.exp(-self.thetas[i].dot(context_vector)))
        
    return context
    
  def get_optimal_reward(self):
    return np.max(self.current_rewards)
  
  def get_expected_reward(self,article):
    return self.current_rewards[article]
  
  def get_stochastic_reward(self,article):
    expected_reward = self.get_expected_reward(article)
    stochastic_reward = np.random.binomial(1,expected_reward)
    return stochastic_reward

