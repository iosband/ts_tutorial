"""Graph shortest path problems."""

from __future__ import division
from __future__ import generators
from __future__ import print_function

import numpy as np

from collections import defaultdict
from base.environment import Environment
from graph.dijkstra import Dijkstra

##############################################################################


class IndependentBinomialBridge(Environment):
  """Graph shortest path on a binomial bridge.

  The agent proceeds up/down for n_stages, but must end with equal ups/downs.
      e.g. (0, 0) - (1, 0) - (2, 0) for n_stages = 2
                  \        /
                    (1, 1)
  We label nodes (x, y) for x=0, 1, .., n_stages and y=0, .., y_lim
  y_lim = x + 1 if x < n_stages / 2 and then decreases again appropriately.
  """

  def __init__(self, n_stages, mu0, sigma0, sigma_tilde=1.):
    """We only implement binomial bridge for an even number of stages.

    The graph is stored as a dict of dicts where graph[node1][node2] is the
    edge distance between node1 and node2.

    Args:
      n_stages - number of stages must be divisible by two
      mu0 - prior mean for independent edges
      sigma0 - prior stddev for independent edges
      sigma_tilde - observation noise stddev
    """
    assert (n_stages % 2 == 0)
    self.n_stages = n_stages
    self.mu0 = mu0
    self.sigma0 = sigma0
    self.sigma_tilde = sigma_tilde

    self.nodes = set()
    self.graph = defaultdict(dict)

    self.optimal_reward = None  # populated when we calculate shortest path

    self._create_graph()
    _ = self.get_shortest_path()

  def get_observation(self):
    """Returns an observation from the environment."""
    return self.n_stages

  def _get_width_bridge(self, x):
    """Computes the width of the binomial bridge at stage x.

    This function = x for x < n_stages, but then goes back down.

    Args:
      x - integer location of depth component
    """
    depth = x - 2 * np.maximum(x - self.n_stages / 2, 0) + 1
    return int(depth)

  def _create_graph(self):
    """Randomly initializes the graph from the prior."""

    # Initially populate the nodes (but not edges)
    for x in range(self.n_stages + 1):
      for y in range(self._get_width_bridge(x)):
        node = (x, y)
        self.nodes.add(node)

    # Add edges only to the correct parts of the graph
    for x in range(self.n_stages + 1):
      for y in range(self._get_width_bridge(x)):
        node = (x, y)

        right_up = (x + 1, y - 1)
        right_equal = (x + 1, y)
        right_down = (x + 1, y + 1)

        if right_down in self.nodes:
          distance = np.exp(self.mu0 + self.sigma0 * np.random.randn())
          self.graph[node][right_down] = distance

        if right_equal in self.nodes:
          distance = np.exp(self.mu0 + self.sigma0 * np.random.randn())
          self.graph[node][right_equal] = distance

        if right_up in self.nodes and right_equal not in self.nodes:
          # Only add edges on way back up if 'equal' node is not present
          distance = np.exp(self.mu0 + self.sigma0 * np.random.randn())
          self.graph[node][right_up] = distance

  def overwrite_edge_length(self, edge_length):
    """Overwrites the existing edge lengths with specified values.

    Args:
      edge_length - dict of dicts edge_length[start_node][end_node] = distance
    """
    for start_node in edge_length:
      for end_node in edge_length[start_node]:
        self.graph[start_node][end_node] = edge_length[start_node][end_node]

  def get_shortest_path(self):
    """Finds the shortest path through the binomial tree.

    Returns:
      path - list of nodes traversed in order.
    """
    start = (0, 0)
    end = (self.n_stages, 0)
    final_distance, predecessor = Dijkstra(self.graph, start, end)

    path = []
    iter_node = end
    while True:
      path.append(iter_node)
      if iter_node == start:
        break
      iter_node = predecessor[iter_node]

    path.reverse()

    # Updating the optimal reward
    self.optimal_reward = -final_distance[end]

    return path

  def get_optimal_reward(self):
    """Returns the optimal possible reward for the environment at that point."""
    return self.optimal_reward

  def get_expected_reward(self, path):
    """Gets the expected reward of an action (in this case a path)

    Args:
      path - list or list-like path of nodes from (0,0) to (n_stage, 0)

    Returns:
      expected_reward - negative value of total expected distance of path
    """
    expected_distance = 0
    for start_node, end_node in zip(path, path[1:]):
      expected_distance += self.graph[start_node][end_node]

    return -expected_distance

  def get_stochastic_reward(self, path):
    """Gets a stochastic reward for the action (in this case a path).

    Args:
      path - list of list-like path of nodes from (0,0) to (n_stage, 0)

    Returns:
      time_elapsed - dict of dicts for elapsed time in each observed edge.
    """
    time_elapsed = defaultdict(dict)
    for start_node, end_node in zip(path, path[1:]):
      mean_time = self.graph[start_node][end_node]
      lognormal_mean = np.log(mean_time) - 0.5 * self.sigma_tilde**2
      stoch_time = np.exp(lognormal_mean + self.sigma_tilde * np.random.randn())
      time_elapsed[start_node][end_node] = stoch_time

    return time_elapsed


##############################################################################


class CorrelatedBinomialBridge(IndependentBinomialBridge):
  """ A Binomial Bridge with corrrelated elapsed time of each edge."""

  def is_in_lower_half(self, start_node, end_node):
    """checks whether the edge start_node-->end_node is located in the lower
    half of the bridge."""

    start_depth = self._get_width_bridge(start_node[0])
    end_depth = self._get_width_bridge(end_node[0])
    if start_node[1] > start_depth / 2:
      return True
    elif start_node[1] < start_depth / 2:
      return False
    else:
      return (start_depth<end_depth and end_node[1]==(start_node[1]+1)) \
      or (start_depth>end_depth and end_node[1]==start_node[1])

  def get_stochastic_reward(self, path):
    """Gets a stochastic reward for the action (in this case a path).

    Args:
      path - list of list-like path of nodes from (0,0) to (n_stage, 0)

    Returns:
      time_elapsed - dict of dicts for elapsed time in each observed edge.
    """

    #shared factors:
    all_edges_factor = np.exp(-(self.sigma_tilde**2) / 6 +
                              self.sigma_tilde * np.random.randn() / np.sqrt(3))
    upper_half_factor = np.exp(-(self.sigma_tilde**2) / 6 + self.sigma_tilde *
                               np.random.randn() / np.sqrt(3))
    lower_half_factor = np.exp(-(self.sigma_tilde**2) / 6 + self.sigma_tilde *
                               np.random.randn() / np.sqrt(3))

    time_elapsed = defaultdict(dict)
    for start_node, end_node in zip(path, path[1:]):
      mean_time = self.graph[start_node][end_node]
      idiosyncratic_factor = np.exp(
          -(self.sigma_tilde**2) / 6 +
          self.sigma_tilde * np.random.randn() / np.sqrt(3))
      if self.is_in_lower_half(start_node, end_node):
        stoch_time = lower_half_factor * all_edges_factor * idiosyncratic_factor * mean_time
      else:
        stoch_time = upper_half_factor * all_edges_factor * idiosyncratic_factor * mean_time
      time_elapsed[start_node][end_node] = stoch_time

    return time_elapsed


##############################################################################


class IndependentBinomialBridgeWithBinaryReward(IndependentBinomialBridge):
  """The reward would be a binary variable indicating whether the recommended
  path was a good one."""

  def __init__(self, n_stages, shape=2, scale=0.5):
    """We only implement binomial bridge for an even number of stages.

    The graph is stored as a dict of dicts where graph[node1][node2] is the
    edge distance between node1 and node2.

    Args:
      n_stages - number of stages must be divisible by two
      shape - shape parameter of the gamma prior
      scale - scale parameter of the gamma prior
    """
    assert (n_stages % 2 == 0)
    self.n_stages = n_stages
    self.shape = shape
    self.scale = scale

    self.nodes = set()
    self.graph = defaultdict(dict)

    self.optimal_reward = None  # populated when we calculate shortest path

    self._create_graph()
    _ = self.get_shortest_path()

  def _create_graph(self):
    """Randomly initializes the graph from the prior."""

    # Initially populate the nodes (but not edges)
    for x in range(self.n_stages + 1):
      for y in range(self._get_width_bridge(x)):
        node = (x, y)
        self.nodes.add(node)

    # Add edges only to the correct parts of the graph
    for x in range(self.n_stages + 1):
      for y in range(self._get_width_bridge(x)):
        node = (x, y)

        right_up = (x + 1, y - 1)
        right_equal = (x + 1, y)
        right_down = (x + 1, y + 1)

        if right_down in self.nodes:
          distance = np.random.gamma(self.shape, self.scale)
          self.graph[node][right_down] = distance

        if right_equal in self.nodes:
          distance = np.random.gamma(self.shape, self.scale)
          self.graph[node][right_equal] = distance

        if right_up in self.nodes and right_equal not in self.nodes:
          # Only add edges on way back up if 'equal' node is not present
          distance = np.random.gamma(self.shape, self.scale)
          self.graph[node][right_up] = distance

  def get_shortest_path(self):
    """Finds the shortest path through the binomial tree.

    Returns:
      path - list of nodes traversed in order.
    """
    start = (0, 0)
    end = (self.n_stages, 0)
    final_distance, predecessor = Dijkstra(self.graph, start, end)

    path = []
    iter_node = end
    while True:
      path.append(iter_node)
      if iter_node == start:
        break
      iter_node = predecessor[iter_node]

    path.reverse()

    # Updating the optimal reward
    # writing exp() in this format to prevent possible overflow
    exp_argument = final_distance[end] - self.n_stages
    if exp_argument > 500:
      self.optimal_reward = 0
    else:
      self.optimal_reward = 1 / (1 + np.exp(exp_argument))

    return path

  def get_expected_reward(self, path):
    """Gets the expected reward of an action (in this case a path)

    Args:
      path - list or list-like path of nodes from (0,0) to (n_stage, 0)

    Returns:
      expected_reward - probability of getting a stochastic reward = 1 for the
      path
    """
    total_time = 0
    for start_node, end_node in zip(path, path[1:]):
      total_time += self.graph[start_node][end_node]

    return 1 / (1 + np.exp(total_time - self.n_stages))

  def get_stochastic_reward(self, path):
    """Gets a stochastic reward for the action (in this case a path).

    Args:
      path - list of list-like path of nodes from (0,0) to (n_stage, 0)

    Returns:
      binary_feedback - binary feedback received for the path
    """
    expected_reward = self.get_expected_reward(path)
    return np.random.binomial(1, expected_reward)

##############################################################################
class MultiAgentCorrelatedBinomialBridge(IndependentBinomialBridge):
  '''Binomail Brdige with independent travel times for multiple agents.'''
  
  def get_expected_reward(self, paths):
    '''determines the expected reward for each of the agents.'''
    
    expected_rewards = [IndependentBinomialBridge.get_expected_reward(self,path)
    for path in paths]
    expected_rewards = np.array(expected_rewards)
    return expected_rewards
  
  def get_stochastic_reward(self, paths):
    '''gives the stochastic reward observed by each agent.'''
    
    times_elapsed = [IndependentBinomialBridge.get_stochastic_reward(self,path)
    for path in paths]
    return times_elapsed
      
      