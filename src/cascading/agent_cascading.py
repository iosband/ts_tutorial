"""Agents for cascading bandit problems.
"""

from __future__ import division
from __future__ import print_function

import numpy as np

from base.agent import Agent

##############################################################################


class CascadingBanditEpsilonGreedy(Agent):

  def __init__(self, num_items, num_positions, a0=1, b0=1, epsilon=0.0):
    """An agent for cascading bandits.

    Args:
      num_items - "L" in math notation
      num_positions - "K" in math notation
      a0 - prior success
      b0
    """
    self.num_items = num_items
    self.num_positions = num_positions
    self.a0 = a0
    self.b0 = b0
    self.prior_success = np.array([a0 for item in range(num_items)])
    self.prior_failure = np.array([b0 for item in range(num_items)])
    self.epsilon = epsilon
    self.timestep = 1

  def set_prior(self, prior_success, prior_failure):
    # Overwrite the default prior
    self.prior_success = np.array(prior_success)
    self.prior_failure = np.array(prior_failure)

  def get_posterior_mean(self):
    return self.prior_success / (self.prior_success + self.prior_failure)

  def get_posterior_sample(self):
    return np.random.beta(self.prior_success, self.prior_failure)

  def update_observation(self, observation, action, reward):
    """Updates observations for cascading bandits.

    Args:
      observation - tuple of (round_failure, round_success) each lists of items
      action - action_list of all the actions tried prior round
      reward - success or not
    """
    for action in observation['round_failure']:
      self.prior_failure[action] += 1

    for action in observation['round_success']:
      self.prior_success[action] += 1

    # Update timestep for UCB agents
    self.timestep += 1

  def pick_action(self, observation):
    if np.random.rand() < self.epsilon:
      action_list = np.random.randint(
          low=0, high=self.num_items, size=self.num_positions)
    else:
      posterior_means = self.get_posterior_mean()
      action_list = posterior_means.argsort()[::-1][:self.num_positions]
    return action_list


##############################################################################


def _ucb_1(empirical_mean, timestep, count):
  """Computes UCB1 upper confidence bound.

  Args:
    empirical_mean - empirical mean
    timestep - time elapsed
    count - number of visits to that object
  """
  confidence = np.sqrt((1.5 * np.log(timestep)) / count)
  return empirical_mean + confidence


class CascadingBanditUCB1(CascadingBanditEpsilonGreedy):

  def pick_action(self, observation):
    posterior_means = self.get_posterior_mean()
    ucb_values = np.zeros(self.num_items)
    for item in range(self.num_items):
      count = self.prior_success[item] + self.prior_failure[item]
      ucb_values[item] = _ucb_1(posterior_means[item], self.timestep, count)

    action_list = ucb_values.argsort()[::-1][:self.num_positions]
    return action_list


##############################################################################


def _kl_ucb(empirical_mean, timestep, count, tolerance=1e-3, maxiter=25):
  """Computes KL-UCB via binary search

  Args:
    empirical_mean - empirical mean
    timestep - time elapsed
    count - number of visits to that object
    tolerance - accuracy for numerical bisection
    maxiter - maximum number of iterations
  """
  kl_bound = (np.log(timestep) + 3 * np.log(np.log(timestep + 1e-6))) / count
  upper_bound = 1
  lower_bound = empirical_mean

  # Most of the experiment is spent for small values of KL --> biased search
  n_iter = 0
  while (upper_bound - lower_bound) > tolerance:
    n_iter += 1
    midpoint = (upper_bound + 3 * lower_bound) / 4
    dist = _d_kl(empirical_mean, midpoint)

    if dist < kl_bound:
      lower_bound = midpoint
    else:
      upper_bound = midpoint

    if n_iter > maxiter:
      print(
          'WARNING: maximum number of iterations exceeded, accuracy only %0.2f'
          % (upper_bound - lower_bound,))
      break

  return lower_bound


def _d_kl(p, q, epsilon=1e-6):
  """Compute the KL divergence for single numbers."""
  if p <= epsilon:
    A = 0
  else:
    A = np.inf if q <= epsilon else p * np.log(p / q)

  if p >= 1 - epsilon:
    B = 0
  else:
    B = np.inf if q >= 1 - epsilon else (1 - p) * np.log((1 - p) / (1 - q))

  return A + B


class CascadingBanditKLUCB(CascadingBanditEpsilonGreedy):

  def pick_action(self, observation):
    posterior_means = self.get_posterior_mean()
    ucb_values = np.zeros(self.num_items)
    for item in range(self.num_items):
      count = self.prior_success[item] + self.prior_failure[item]
      ucb_values[item] = _kl_ucb(posterior_means[item], self.timestep, count)

    action_list = ucb_values.argsort()[::-1][:self.num_positions]
    return action_list


##############################################################################


class CascadingBanditTS(CascadingBanditEpsilonGreedy):

  def pick_action(self, observation):
    posterior_sample = self.get_posterior_sample()
    action_list = posterior_sample.argsort()[::-1][:self.num_positions]
    return action_list
