"""Agents for news recommendation problem."""

from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
import random as rnd

from base.agent import Agent

_SMALL_NUMBER = 1e-10
_MEDIUM_NUMBER=.01
_LARGE_NUMBER = 1e+2
##############################################################################

class GreedyNewsRecommendation(Agent):
  """Greedy News Recommender."""
  
  def __init__(self,num_articles,dim,theta_mean=0,theta_std=1,epsilon=0.0,
               alpha=0.2,beta=0.5,tol=0.0001):
    """Args:
      num_articles - number of news articles
      dim - dimension of the problem
      theta_mean - mean of each component of theta
      theta_std - std of each component of theta
      epsilon - used in epsilon-greedy.
      alpha - used in backtracking line search
      beta - used in backtracking line search
      tol - stopping criterion of Newton's method.
      """
      
    self.num_articles = num_articles
    self.dim = dim
    self.theta_mean = theta_mean
    self.theta_std = theta_std
    self.back_track_alpha = alpha
    self.back_track_beta = beta
    self.tol = tol
    self.epsilon = epsilon
   
    # keeping current map estimates and Hessians for each news article
    self.current_map_estimates = [self.theta_mean*np.ones(self.dim) 
                                            for _ in range(self.num_articles)]
    self.current_Hessians = [np.diag([1/self.theta_std**2]*self.dim) 
                                            for _ in range(self.num_articles)]
  
    # keeping the observations for each article
    self.num_plays = [0 for _ in range(self.num_articles)]
    self.contexts = [[] for _ in range(self.num_articles)]
    self.rewards = [[] for _ in range(self.num_articles)]
    
  def _compute_gradient_hessian_prior(self,x):
    '''computes the gradient and Hessian of the prior part of 
        negative log-likelihood at x.'''
    Sinv = np.diag([1/self.theta_std**2]*self.dim) 
    mu = self.theta_mean*np.ones(self.dim)
    
    g = Sinv.dot(x - mu)
    H = Sinv
    
    return g,H
  
  def _compute_gradient_hessian(self,x,article):
    """computes gradient and Hessian of negative log-likelihood  
    at point x, based on the observed data for the given article."""
    
    g,H = self._compute_gradient_hessian_prior(x)
    
    for i in range(self.num_plays[article]):
      z = self.contexts[article][i]
      y = self.rewards[article][i]
      pred = 1/(1+np.exp(-x.dot(z)))
      
      g = g + (pred-y)*z
      H = H + pred*(1-pred)*np.outer(z,z)
    
    return g,H

  def _evaluate_log1pexp(self, x):
    """given the input x, returns log(1+exp(x))."""
    if x > _LARGE_NUMBER:
      return x
    else:
      return np.log(1+np.exp(x))

  def _evaluate_negative_log_prior(self, x):
    """returning negative log-prior evaluated at x."""
    Sinv = np.diag([1/self.theta_std**2]*self.dim) 
    mu = self.theta_mean*np.ones(self.dim)
    
    return 0.5*(x-mu).T.dot(Sinv.dot(x-mu))

  def _evaluate_negative_log_posterior(self, x, article):
    """evaluate negative log-posterior for article at point x."""

    value = self._evaluate_negative_log_prior(x)
    for i in range(self.num_plays[article]):
      z = self.contexts[article][i]
      y = self.rewards[article][i]
      value = value + self._evaluate_log1pexp(x.dot(z)) - y*x.dot(z)
      
    return value
  
  def _back_track_search(self, x, g, dx, article):
    """Finding the right step size to be used in Newton's method.
    Inputs:
      x - current point
      g - gradient of the function at x
      dx - the descent direction

    Retruns:
      t - the step size"""

    step = 1
    current_function_value = self._evaluate_negative_log_posterior(x, article)
    difference = self._evaluate_negative_log_posterior(x + step*dx, article) - \
    (current_function_value + self.back_track_alpha*step*g.dot(dx))
    while difference > 0:
      step = self.back_track_beta * step
      difference = self._evaluate_negative_log_posterior(x + step*dx, article) - \
          (current_function_value + self.back_track_alpha*step*g.dot(dx))

    return step

  def _optimize_Newton_method(self, article):
    """Optimize negative log_posterior function via Newton's method for the
    given article."""
    
    x = self.current_map_estimates[article]
    error = self.tol + 1
    while error > self.tol:
      g, H = self._compute_gradient_hessian(x,article)
      delta_x = -npla.solve(H, g)
      step = self._back_track_search(x, g, delta_x, article)
      x = x + step * delta_x
      error = -g.dot(delta_x)
      
    # computing the gradient and hessian at final point
    g, H = self._compute_gradient_hessian(x,article)

    # updating current map and Hessian for this article
    self.current_map_estimates[article] = x
    self.current_Hessians[article] = H
    return x, H
  
  def update_observation(self, context, article, feedback):
    '''updates the observations for displayed article, given the context and 
    user's feedback. The new observations are saved in the history of the 
    displayed article and the current map estimate and Hessian of this 
    article are updated right away.
    
    Args:
      context - a list containing observed context vector for each article
      article - article which was recently shown
      feedback - user's response.
      '''
    self.num_plays[article] += 1
    self.contexts[article].append(context[article])
    self.rewards[article].append(feedback)
    
    # updating the map estimate and Hessian for displayed article
    _,__ = self._optimize_Newton_method(article)
  
  def _map_rewards(self,context):
    map_rewards = []
    for i in range(self.num_articles):
      x = context[i]
      theta = self.current_map_estimates[i]
      map_rewards.append(1/(1+np.exp(-theta.dot(x))))
    return map_rewards
  
  def pick_action(self,context):
    '''Greedy action based on map estimates.'''
    map_rewards = self._map_rewards(context)
    article = np.argmax(map_rewards)
    return article
##############################################################################
class EpsilonGreedyNewsRecommendation(GreedyNewsRecommendation):
  '''Epsilon greedy agent for the news recommendation problem.'''
  
  def pick_action(self,context):
    '''Greedy action based on map estimates.'''
    map_rewards = self._map_rewards(context)
    if np.random.uniform()<self.epsilon:
      article = np.random.randint(0,self.num_articles)
    else:
      article = np.argmax(map_rewards)
    return article
##############################################################################
class LaplaceTSNewsRecommendation(GreedyNewsRecommendation):   
  '''Laplace approximation to TS for news recommendation problem.'''
  def _sampled_rewards(self,context):
    sampled_rewards = []
    for i in range(self.num_articles):
      x = context[i]
      mean = self.current_map_estimates[i]
      cov = npla.inv(self.current_Hessians[i])
      theta = np.random.multivariate_normal(mean, cov)
      sampled_rewards.append(1/(1+np.exp(-theta.dot(x))))
    return sampled_rewards
    
  def pick_action(self,context):
    '''Greedy action based on map estimates.'''
    sampled_rewards = self._sampled_rewards(context)
    article = np.argmax(sampled_rewards)
    return article

##############################################################################
class LangevinTSNewsRecommendation(GreedyNewsRecommendation):
  def __init__(self,num_articles,dim,theta_mean=0,theta_std=1,epsilon=0.0,
               alpha=0.2,beta=0.5,tol=0.0001,batch_size = 100, step_count=200,
               step_size=.01):
    GreedyNewsRecommendation.__init__(self,num_articles,dim,theta_mean,theta_std,
              epsilon,alpha,beta,tol)
    self.batch_size = batch_size
    self.step_count = step_count
    self.step_size = step_size
    
  def _compute_stochastic_gradient(self, x, article):
    '''computes a stochastic gradient of the negative log-posterior for the given
     article.'''
    
    if self.num_plays[article]<=self.batch_size:
      sample_indices = range(self.num_plays[article])
      gradient_scale = 1
    else:
      gradient_scale = self.num_plays[article]/self.batch_size
      sample_indices = rnd.sample(range(self.num_plays[article]),self.batch_size)
    
    g = np.zeros(self.dim)
    for i in sample_indices:
      z = self.contexts[article][i]
      y = self.rewards[article][i]
      pred = 1/(1+np.exp(-x.dot(z)))
      g = g + (pred-y)*z
    
    g_prior,_ = self._compute_gradient_hessian_prior(x)
    g = gradient_scale*g + g_prior
    return g
  
  def _Langevin_samples(self):
    '''gives the Langevin samples for each of the articles'''
    sampled_thetas = []
    for a in range(self.num_articles):
      # determining starting point and conditioner
      x = self.current_map_estimates[a]
      preconditioner = npla.inv(self.current_Hessians[a])
      preconditioner_sqrt=spla.sqrtm(preconditioner)
      
      #Remove any complex component in preconditioner_sqrt arising from numerical error
      complex_part=np.imag(preconditioner)
      if (spla.norm(complex_part)> _SMALL_NUMBER):
          print("Warning. There may be numerical issues.  Preconditioner has complex values")
          print("Norm of the imaginary component is, ")+str(spla.norm(complex_part))
      preconditioner_sqrt=np.real(preconditioner_sqrt)
      
      for i in range(self.step_count):
        g = -self._compute_stochastic_gradient(x,a)
        scaled_grad=preconditioner.dot(g)
        scaled_noise = preconditioner_sqrt.dot(np.random.randn(self.dim)) 
        x = x + self.step_size * scaled_grad+np.sqrt(2*self.step_size)*scaled_noise
      sampled_thetas.append(x)
    return sampled_thetas
  
  def _sampled_rewards(self,context):
    sampled_rewards = []
    sampled_theta = self._Langevin_samples()
    for i in range(self.num_articles):
      x = context[i]
      theta = sampled_theta[i]
      sampled_rewards.append(1/(1+np.exp(-theta.dot(x))))
    return sampled_rewards
    
  def pick_action(self,context):
    '''Greedy action based on map estimates.'''
    sampled_rewards = self._sampled_rewards(context)
    article = np.argmax(sampled_rewards)
    return article
 