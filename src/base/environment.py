""" Environment determines the underlying law of the system.

All bandit problems should inherit from environment.
"""

from __future__ import division
from __future__ import print_function

import numpy as np


##############################################################################

class Environment(object):
  """Base class for all bandit environments."""

  def __init__(self):
    """Initialize the environment."""
    pass

  def get_observation(self):
    """Returns an observation from the environment."""
    pass

  def get_optimal_reward(self):
    """Returns the optimal possible reward for the environment at that point."""
    pass

  def get_expected_reward(self, action):
    """Gets the expected reward of an action."""
    pass

  def get_stochastic_reward(self, action):
    """Gets a stochastic reward for the action."""
    pass

  def advance(self, action, reward):
    """Updating the environment (useful for nonstationary bandit)."""
    pass

