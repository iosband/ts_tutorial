"""Finite bandit agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from base.agent import Agent
from base.agent import random_argmax

##############################################################################


class FiniteBernoulliBanditEpsilonGreedy(Agent):
  """Simple agent made for finite armed bandit problems."""

  def __init__(self, n_arm, a0=1, b0=1, epsilon=0.0):
    self.n_arm = n_arm
    self.epsilon = epsilon
    self.prior_success = np.array([a0 for arm in range(n_arm)])
    self.prior_failure = np.array([b0 for arm in range(n_arm)])

  def set_prior(self, prior_success, prior_failure):
    # Overwrite the default prior
    self.prior_success = np.array(prior_success)
    self.prior_failure = np.array(prior_failure)

  def get_posterior_mean(self):
    return self.prior_success / (self.prior_success + self.prior_failure)

  def get_posterior_sample(self):
    return np.random.beta(self.prior_success, self.prior_failure)

  def update_observation(self, observation, action, reward):
    # Naive error checking for compatibility with environment
    assert observation == self.n_arm

    if np.isclose(reward, 1):
      self.prior_success[action] += 1
    elif np.isclose(reward, 0):
      self.prior_failure[action] += 1
    else:
      raise ValueError('Rewards should be 0 or 1 in Bernoulli Bandit')

  def pick_action(self, observation):
    """Take random action prob epsilon, else be greedy."""
    if np.random.rand() < self.epsilon:
      action = np.random.randint(self.n_arm)
    else:
      posterior_means = self.get_posterior_mean()
      action = random_argmax(posterior_means)

    return action


##############################################################################


class FiniteBernoulliBanditTS(FiniteBernoulliBanditEpsilonGreedy):
  """Thompson sampling on finite armed bandit."""

  def pick_action(self, observation):
    """Thompson sampling with Beta posterior for action selection."""
    sampled_means = self.get_posterior_sample()
    action = random_argmax(sampled_means)
    return action


##############################################################################


class FiniteBernoulliBanditBootstrap(FiniteBernoulliBanditTS):
  """Bootstrapped Thompson sampling on finite armed bandit."""

  def get_posterior_sample(self):
    """Use bootstrap resampling instead of posterior sample."""
    total_tries = self.prior_success + self.prior_failure
    prob_success = self.prior_success / total_tries
    boot_sample = np.random.binomial(total_tries, prob_success) / total_tries
    return boot_sample

##############################################################################


class FiniteBernoulliBanditLaplace(FiniteBernoulliBanditTS):
  """Laplace Thompson sampling on finite armed bandit."""

  def get_posterior_sample(self):
    """Gaussian approximation to posterior density (match moments)."""
    (a, b) = (self.prior_success + 1e-6 - 1, self.prior_failure + 1e-6 - 1)
    # The modes are not well defined unless alpha, beta > 1
    assert np.all(a > 0)
    assert np.all(b > 0)
    mode = a / (a + b)
    hessian = a / mode + b / (1 - mode)
    laplace_sample = mode + np.sqrt(1 / hessian) * np.random.randn(self.n_arm)
    return laplace_sample

##############################################################################


class DriftingFiniteBernoulliBanditTS(FiniteBernoulliBanditTS):
  """Thompson sampling on finite armed bandit."""

  def __init__(self, n_arm, a0=1, b0=1, gamma=0.01):
    self.n_arm = n_arm
    self.a0 = a0
    self.b0 = b0
    self.prior_success = np.array([a0 for arm in range(n_arm)])
    self.prior_failure = np.array([b0 for arm in range(n_arm)])
    self.gamma = gamma

  def update_observation(self, observation, action, reward):
    # Naive error checking for compatibility with environment
    assert observation == self.n_arm

    # All values decay slightly, observation updated
    self.prior_success = self.prior_success * (
        1 - self.gamma) + self.a0 * self.gamma
    self.prior_failure = self.prior_failure * (
        1 - self.gamma) + self.b0 * self.gamma
    self.prior_success[action] += reward
    self.prior_failure[action] += 1 - reward
