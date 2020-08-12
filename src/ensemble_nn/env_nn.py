"""Neural net bandit environment."""
import numpy as np
import numpy.random as rd

from base.environment import Environment


class TwoLayerNNBandit(Environment):
  """A bandit environment where the mapping from actions to expected rewards
  is represented by a two-layer neural network with the following structure:
    input -> affine -> relu -> linear -> output
  """

  def __init__(self,
               input_dim,
               hidden_dim,
               num_actions,
               prior_var,
               noise_var,
               seed=4321):
    """Initialize a random 2-layer neural network instance.

    Args:
      input_dim: int size of input vector.
      hidden_dim: int number of hidden units.
      num_actions: int number of random actions to generate.
      prior_var: prior variance of weights.
      noise_var: variance of noise.
    """
    np.random.seed(seed)

    # sample true weights from N(0, prior_var * I)
    self.W1 = np.sqrt(prior_var) * rd.randn(hidden_dim, input_dim)
    self.W2 = np.sqrt(prior_var) * rd.randn(hidden_dim)

    # generate actions: each dimension is uniformly sampled from [-1, 1],
    # except for the last dimension, which is set to 1
    self.actions = np.hstack((rd.uniform(-1, 1, (num_actions, input_dim - 1)),
                              np.ones((num_actions, 1))))

    # compute expected rewards
    affine_out = np.sum(self.actions[:, np.newaxis, :] * self.W1, axis=2)
    relu_out = np.maximum(0, affine_out)
    self.expected_rewards = np.sum(relu_out * self.W2, axis=1)

    self.optimal_reward = np.max(self.expected_rewards)

    self.noise_var = noise_var
    self.t = 0  # time

  def get_actions(self):
    """Numpy array of num_actions random potential action vectors."""
    return self.actions

  def get_observation(self):
    return self.t

  def get_optimal_reward(self):
    return self.optimal_reward

  def get_expected_reward(self, action):
    return self.expected_rewards[action]

  def get_stochastic_reward(self, action):
    self.t += 1
    noise = np.sqrt(self.noise_var) * rd.randn()
    return self.expected_rewards[action] + noise
