'''Environment for cascading bandit
'''

from __future__ import division
from __future__ import print_function

import numpy as np

from base.environment import Environment


############################################################################

class CascadingBandit(Environment):

  def __init__(self, num_items, num_positions, a0, b0):
    """A cascading bandit instance

    Args:
      num_items - "L" in math notation
      num_positions - "K" in math notation
      a0 - prior success
      b0 - prior failute
    """
    assert(num_items >= num_positions)
    self.num_items = num_items
    self.num_positions = num_positions
    self.a0 = a0
    self.b0 = b0
    self.probs = np.array([np.random.beta(a0, b0) for a in range(num_items)])

    # Compute the optimal reward by sorting the best num_positions elements
    probs_ordered = np.sort(self.probs)[::-1]
    self.optimal_reward = 1 - np.prod(1 - probs_ordered[:num_positions])

    # Maintain a list of all actions that failed and succeeded
    self.round_failure = []
    self.round_success = []


  def get_observation(self):
    observation = {'round_failure': self.round_failure,
                   'round_success': self.round_success}
    return observation

  def get_optimal_reward(self):
    return self.optimal_reward

  def get_expected_reward(self, action_list):
    """Gets the expected reward of an action list

    Args:
      action_list - list of actions (of length self.num_positions)
    """
    assert(len(action_list) == self.num_positions)
    action_probs = self.probs[action_list]
    expected_reward = 1 - np.prod(1 - action_probs)
    return expected_reward

  def get_stochastic_reward(self, action_list):
    """Generates a stochastic reward from action list.
    Also updates self.recent_success for subsequent observations.

    Args:
      action_list - list of actions (of length self.num_positions)
    """
    assert(len(action_list) == self.num_positions);

    self.round_failure = []
    self.round_success = []

    for action in action_list:
      click = np.random.binomial(1, self.probs[action])
      if click == 1:
        self.round_success += [action]
        return click
      else:
        self.round_failure += [action]

    return 0


