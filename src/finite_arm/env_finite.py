"""Finite armed bandit environments."""
import numpy as np

from base.environment import Environment

##############################################################################


class FiniteArmedBernoulliBandit(Environment):
  """Simple N-armed bandit."""

  def __init__(self, probs):
    self.probs = np.array(probs)
    assert np.all(self.probs >= 0)
    assert np.all(self.probs <= 1)

    self.optimal_reward = np.max(self.probs)
    self.n_arm = len(self.probs)

  def get_observation(self):
    return self.n_arm

  def get_optimal_reward(self):
    return self.optimal_reward

  def get_expected_reward(self, action):
    return self.probs[action]

  def get_stochastic_reward(self, action):
    return np.random.binomial(1, self.probs[action])


##############################################################################


class DriftingFiniteArmedBernoulliBandit(FiniteArmedBernoulliBandit):
  """N-armed bandit with drift.

  Note that this code does not generate a specific fixed bandit environemnt.
  Instead we simulate a draw from the Bayesian prior online, this is done to
  show a simple example where the prior/posterior computations are conjugate.

  gamma = 0 will allow us to do resampling without drift.
  """

  def __init__(self, n_arm, a0=1., b0=1., gamma=0.01):
    self.n_arm = n_arm
    self.a0 = a0
    self.b0 = b0
    self.prior_success = np.array([a0 for a in range(n_arm)])
    self.prior_failure = np.array([b0 for a in range(n_arm)])
    self.gamma = gamma
    self.probs = np.array([np.random.beta(a0, b0) for a in range(n_arm)])

  def set_prior(self, prior_success, prior_failure):
    # Overwrite the default prior
    self.prior_success = np.array(prior_success)
    self.prior_failure = np.array(prior_failure)

  def get_optimal_reward(self):
    return np.max(self.probs)

  def advance(self, action, reward):
    # All arms drift back to mixing distribution at rate gamma
    self.prior_success = self.prior_success * (
        1 - self.gamma) + self.a0 * self.gamma
    self.prior_failure = self.prior_failure * (
        1 - self.gamma) + self.b0 * self.gamma

    # Sampled arm has some learning
    self.prior_success[action] += reward
    self.prior_failure[action] += 1 - reward

    # Resample posterior probabilities
    self.probs = np.array([
        np.random.beta(self.prior_success[a], self.prior_failure[a])
        for a in range(self.n_arm)
    ])
