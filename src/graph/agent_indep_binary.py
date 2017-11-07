"""Agents for graph bandit problems.

These agents became too large to comfortably keep in one file, so we divided
it up into three separate sections for:
- Independent Binomial Bridge
- Correlated Binomial Bridge
- Independent Binomial Bridge with Binary Reward
"""

from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla

from collections import defaultdict
from base.agent import Agent
from graph.env_graph_bandit import IndependentBinomialBridgeWithBinaryReward


_SMALL_NUMBER = 1e-10
_LARGE_NUMBER = 1e+2
##############################################################################


class BootstrapIndependentBBWithBinaryReward(Agent):
  """Bootstrap method on Binomial Bridges with binary feedback for each path."""

  def __init__(self, n_stages, shape=2, scale=0.5, tol=0.1, alpha=0.2,
               beta=0.5):
    """An agent for graph bandits.

    Args:
      n_stages - number of stages of the binomial bridge (must be even)
      shape - shape parameter of the Gamma prior
      scale - scale parameter of the Gamma prior
      tol - tolerance used in Newton's method
      alpha, beta - parameters of backtracking line search
    """
    assert (n_stages % 2 == 0)
    self.n_stages = n_stages
    self.shape = shape
    self.scale = scale
    self.tol = tol
    self.back_track_alpha = alpha
    self.back_track_beta = beta

    # Set up the internal environment with arbitrary initial values
    self.internal_env = IndependentBinomialBridgeWithBinaryReward(
        n_stages, shape, scale)

    # Save a map (start_node,end_node)-->R to fascilitate calculations
    self.edge2index = defaultdict(dict)
    self.index2edge = defaultdict(dict)
    edge_counter = 0
    for start_node in self.internal_env.graph:
      for end_node in self.internal_env.graph[start_node]:
        self.edge2index[start_node][end_node] = edge_counter
        self.index2edge[edge_counter] = (start_node, end_node)
        edge_counter += 1

    # saving the number of total edges
    self.num_edges = edge_counter

    # prior covariance matrix and initial point for optimization
    self.prior_covariance = np.diag([self.shape *
                                     (self.scale)**2] * self.num_edges)
    self.initial_point = np.ones(self.num_edges)

    # a prior sample
    self.prior_sample = None  # populated as needed

    # initializing a history of traveresed paths and binary feedbacks
    self.path_history = []
    self.feedback_history = []

    # keeping track of history size
    self.history_size = 0

    # resampled history
    self.resampled_path_history = []
    self.resampled_feedback_history = []

  def update_observation(self, observation, path, feedback):
    """Updates observations for binomial bridge

    Args:
      observation - number of stages
      path - path chosen by the agent (not used)
      feedback - binary reward received for the path
    """
    assert (observation == self.n_stages)
    self.feedback_history.append(feedback)
    traversed_edges_index = []
    for start_node, end_node in zip(path, path[1:]):
      traversed_edges_index.append(self.edge2index[start_node][end_node])
    self.path_history.append(traversed_edges_index)
    self.history_size += 1
    assert (self.history_size == len(self.feedback_history))

  def _resample_history(self, random=True):
    """Generates a resampled version of the history.
    If random = False, then no resampling is done."""

    if self.history_size > 0:
      if random:
        random_indices = np.random.randint(0, self.history_size,
                                           self.history_size)
        resampled_feedback_history = []
        resampled_path_hitory = []
        for ind in random_indices:
          resampled_feedback_history.append(self.feedback_history[ind])
          resampled_path_hitory.append(self.path_history[ind])

        self.resampled_path_history = resampled_path_hitory
        self.resampled_feedback_history = resampled_feedback_history
      else:
        self.resampled_path_history = self.path_history
        self.resampled_feedback_history = self.feedback_history

  def _find_containing_paths(self):
    """
       Return:
         A list of lists, mapping an edge index to the index of the resampled
         paths in
         self.resampled_path_history whichs contain that edge.
    """

    contained_paths = [[] for i in range(self.num_edges)]
    for i in range(self.history_size):
      path = self.resampled_path_history[i]
      for edge in path:
        contained_paths[edge].append(i)

    return contained_paths

  def _generate_prior_sample(self):
    self.prior_sample = np.random.gamma(self.shape, self.scale, self.num_edges)

  def _compute_gradient_and_hessian_prior_part(self, x):
    """Return the gradient and hessian of the randomized prior part in
    log-posterior pdf. """
    grad = -2 * self.prior_covariance.dot(x - self.prior_sample)
    hessian = -2 * self.prior_covariance
    return grad, hessian

  def _get_diag_grad(self, path_weights, feedbacks):
    K = np.exp(-self.n_stages)
    diag_grad = 0
    for i in range(len(path_weights)):
      # TODO(iosband) THIS PART OF THE CODE BLOWS UP WITH LANGEVIN
      ratio = (K * np.exp(path_weights[i])) / (1 + K * np.exp(path_weights[i]))
      diag_grad += (1 - feedbacks[i]) - ratio
    return diag_grad

  def _get_diag_hessian(self, path_weights):
    K = np.exp(-self.n_stages)
    diag_hess = 0
    for i in range(len(path_weights)):
      diag_hess += -(K * np.exp(path_weights[i])) / ((
          1 + K * np.exp(path_weights[i]))**2)
    return diag_hess

  def _get_off_diag_hessian(self, path_weights):
    K = np.exp(-self.n_stages)
    off_hess = 0
    for i in range(len(path_weights)):
      off_hess += -(K * np.exp(path_weights[i])) / ((
          1 + K * np.exp(path_weights[i]))**2)
    return off_hess

  def _compute_gradient_and_hessian(self, x):
    """Return: gradient and hessian of the (modified) log-posterior pdf at point x."""

    edge_container = self._find_containing_paths()

    grad = np.zeros(self.num_edges)
    hessian = np.zeros((self.num_edges, self.num_edges))
    for edge in range(self.num_edges):
      involved_cases = edge_container[edge]
      path_weights = [sum([x[i] for i in self.resampled_path_history[case]]) \
                      for case in involved_cases]
      feedbacks = [
          self.resampled_feedback_history[case] for case in involved_cases
      ]
      grad[edge] = self._get_diag_grad(path_weights, feedbacks)
      hessian[edge][edge] = self._get_diag_hessian(path_weights)

      for new_edge in range(edge):
        involved_cases_2 = edge_container[new_edge]
        involved_both = [i for i in involved_cases if i in involved_cases_2]
        path_weights = [
            sum([x[i] for i in self.resampled_path_history[case]])
            for case in involved_both
        ]
        hessian[edge][new_edge] += self._get_off_diag_hessian(path_weights)
        hessian[new_edge][edge] += self._get_off_diag_hessian(path_weights)

    # adding with prior part
    grad_prior, hessian_prior = self._compute_gradient_and_hessian_prior_part(x)
    grad = grad + grad_prior
    hessian = hessian + hessian_prior
    return grad, hessian

  def _evaluate_log1pexp(self, x):
    """given the input x, returns log(1+exp(x))."""
    if x > _LARGE_NUMBER:
      return x
    else:
      return np.log(1+np.exp(x))

  def _evaluate_log_prior(self, x):
    """returning the randomized prior part in the log-posterior pdf."""
    return -(x - self.prior_sample).dot(
        self.prior_covariance.dot(x - self.prior_sample))

  def _evaluate_log_posterior(self, x):
    """evaluate the (randomized) objective function we want to maximize over at x."""

    prior_part = self._evaluate_log_prior(x)

    constant = np.exp(-self.n_stages)
    likelihood_part = 0
    for count, path in enumerate(self.resampled_path_history):
      path_weight = sum([x[i] for i in path])
      likelihood_part += ((1 - self.resampled_feedback_history[count]) *
                          (path_weight - self.n_stages) -
                          self._evaluate_log1pexp(np.log(constant)+path_weight))
    return likelihood_part + prior_part

  def _back_track_search(self, x, g, dx):
    """Finding the right step size to be used in Newton's method.
    Inputs:
      x - current point
      g - gradient of the function at x
      dx - the descent direction

    Retruns:
      t - the step size"""

    step = 1
    current_function_value = self._evaluate_log_posterior(x)
    difference = self._evaluate_log_posterior(x + step*dx) - \
    (current_function_value + self.back_track_alpha*step*g.dot(dx))
    while difference < 0:
      step = self.back_track_beta * step
      difference = self._evaluate_log_posterior(x + step*dx) - \
          (current_function_value + self.back_track_alpha*step*g.dot(dx))

    return step

  def _optimize_Newton_method(self, projection=False):
    """Optimize the log_posterior function via Newton's method. If
    projection = True, the values of xs are projected to be positive
    (i.e., > _SMALL_NUMBER). This is to prevent numerical flow, specially used
    in Laplace method where we are dealing with logs."""

    self._generate_prior_sample()
    x = self.initial_point
    error = self.tol + 1
    while error > self.tol:
      g, H = self._compute_gradient_and_hessian(x)
      delta_x = -npla.solve(H, g)
      step = self._back_track_search(x, g, delta_x)
      x = x + step * delta_x
      if projection:
        x = np.maximum(x, _SMALL_NUMBER)
      error = g.dot(delta_x)
    # computing the gradient and hessian at final point
    g, H = self._compute_gradient_and_hessian(x)

    # saving the current solution as the starting point for next episode
    self.initial_point = x
    return x, H

  def get_sample(self):
    """Sets the bootstrap sample for each edge length

    Return:
      edge_length - dict of dicts edge_length[start_node][end_node] = distance
    """
    # resampling the history
    self._resample_history()

    # flattened sample
    flattened_sample, _ = self._optimize_Newton_method()

    # making sure all the samples are positive
    flattened_sample = np.maximum(flattened_sample, _SMALL_NUMBER)

    edge_length = copy.deepcopy(self.internal_env.graph)

    for start_node in edge_length:
      for end_node in edge_length[start_node]:
        edge_index = self.edge2index[start_node][end_node]
        edge_length[start_node][end_node] = flattened_sample[edge_index]

    return edge_length

  def pick_action(self, observation):
    """Greedy shortest path wrt bootstrap sample."""
    bootstrap_sample = self.get_sample()
    self.internal_env.overwrite_edge_length(bootstrap_sample)
    path = self.internal_env.get_shortest_path()

    return path


##############################################################################


class LaplaceIndependentBBWithBinaryReward(
    BootstrapIndependentBBWithBinaryReward):
  """Laplace method on Binomial Bridges with binary feedback for each path."""

  def _evaluate_log_prior(self, x):
    """returning the randomized prior part in the log-posterior pdf."""
    return np.sum((self.shape - 1) * np.log(x) - x / self.scale)

  def _compute_gradient_and_hessian_prior_part(self, x):
    """Return the gradient and hessian of the log-prior pdf."""

    grad = (self.shape - 1) / x - 1 / self.scale
    hessian = np.diag(-(self.shape - 1) / (x**2))
    return grad, hessian

  def _back_track_search(self, x, g, dx):
    """Finding the right step size to be used in Newton's method.
    Inputs:
      x - current point
      g - gradient of the function at x
      dx - the descent direction

    Retruns:
      step - the step size"""

   # reducing the step size so that the new point is in the domain
    cont = True
    step = 1
    while cont:
      x_new = x + step*dx
      if np.min(x_new)>0:
        cont = False
      else:
        step = self.back_track_beta*step

    # reducing the step size via line search
    current_function_value = self._evaluate_log_posterior(x)
    difference = self._evaluate_log_posterior(x + step*dx) - \
    (current_function_value + self.back_track_alpha*step*g.dot(dx))
    while difference < 0:
      step = self.back_track_beta * step
      difference = self._evaluate_log_posterior(x + step*dx) - \
          (current_function_value + self.back_track_alpha*step*g.dot(dx))

    return step

  def get_sample(self):
    """Sets the bootstrap sample for each edge length

    Return:
      edge_length - dict of dicts edge_length[start_node][end_node] = distance
    """
    # resampling the history
    self._resample_history(False)

    # flattened sample
    x_map, hessian = self._optimize_Newton_method(True)
    cov = npla.inv(-hessian)
    flattened_sample = np.random.multivariate_normal(x_map, cov)

    # making sure all the samples are positive
    flattened_sample = np.maximum(flattened_sample, _SMALL_NUMBER)

    edge_length = copy.deepcopy(self.internal_env.graph)

    for start_node in edge_length:
      for end_node in edge_length[start_node]:
        edge_index = self.edge2index[start_node][end_node]
        edge_length[start_node][end_node] = flattened_sample[edge_index]

    return edge_length


class LangevinMCMCIndependentBBWithBinaryReward(
    LaplaceIndependentBBWithBinaryReward):
  """Langevin MCMC method on Binomial Bridges with binary feedback for each path."""
  """adds step_count and step_size"""

  def __init__(self,
               n_stages,
               shape=2,
               scale=0.5,
               tol=0.1,
               alpha=0.2,
               beta=0.5,
               step_count=100,
               step_size=.01):
    BootstrapIndependentBBWithBinaryReward.__init__(self, n_stages, shape,
                                                    scale, tol, alpha, beta)
    self.step_size = step_size
    self.step_count = step_count

  def _compute_just_gradient(self, x):
    """Return: gradient and hessian of the log-posterior pdf at point x."""

    edge_container = self._find_containing_paths()
    grad = np.zeros(self.num_edges)
    for edge in range(self.num_edges):
      involved_cases = edge_container[edge]
      path_weights = [sum([x[i] for i in self.resampled_path_history[case]]) \
                      for case in involved_cases]
      feedbacks = [
          self.resampled_feedback_history[case] for case in involved_cases
      ]

      grad[edge] = self._get_diag_grad(path_weights, feedbacks)

    # adding with prior part
    grad_prior = (self.shape - 1) / x - 1 / self.scale
    grad = grad + grad_prior
    return grad

  def _optimize_gradient_descent_method(self, projection=False):
    """Optimize the log_posterior function via gradient descent method. If
    projection = True, the values of xs are projected to be positive
    (i.e., > _SMALL_NUMBER). This is to prevent numerical flow."""

    x = self.initial_point
    error = self.tol + 1
    while error > self.tol:
      g = self._compute_just_gradient(x)
      delta_x = g
      step = self._back_track_search(x, g, delta_x)
      x = x + step * delta_x
      if projection:
        x = np.maximum(x, _SMALL_NUMBER)
      error = g.dot(g)

    # saving the current solution as the starting point for next episode
    self.initial_point = x
    return x


  def test_gradient_ascent(self, test_point, tolerance):
    """Computes the MAP estimate via Netwon's method. Computes it again by running
    gradient ascent from the test_point. If the two differ by more than the tolarance,
    it prints an error message."""

    newton_map, hessian=self._optimize_Newton_method(True)

    old_init_point =self.initial_point

    self.initial_point=test_point
    gradient_map_estimate=self._optimize_gradient_descent_method(True)
    abs_error = np.max(np.abs(gradient_map_estimate-newton_map))
    if (abs_error > tolerance):
        print ('Warning: gradient ascent maximizer had an erorr of size '+str(abs_error))

    self.initial_point= old_init_point

  def get_sample(self):
    """Sets the bootstrap sample for each edge length

    Return:
      edge_length - dict of dicts edge_length[start_node][end_node] = distance
    """
    # resampling the history
    self._resample_history(False)

    # flattened sample
    x_map, hessian = self._optimize_Newton_method(True)
    cov = npla.inv(-hessian)
    flattened_sample = np.random.multivariate_normal(x_map, cov)

    # making sure all the samples are positive
    flattened_sample = np.maximum(flattened_sample, _SMALL_NUMBER)

    ##Testing Gradient Decent Method.
    self.test_gradient_ascent(flattened_sample, .05)

    #print(np.max(np.abs(test_map_estimate-x_map)))

    edge_length = copy.deepcopy(self.internal_env.graph)

    for start_node in edge_length:
      for end_node in edge_length[start_node]:
        edge_index = self.edge2index[start_node][end_node]
        edge_length[start_node][end_node] = flattened_sample[edge_index]

    return edge_length


"""This code produces errors, and is temporarily commented out. DO NOT DELTE"""
#  def get_sample(self):
#    """Sets the sample for each edge length
#
#    Return:
#      edge_length - dict of dicts edge_length[start_node][end_node] = distance
#    """
#    # no resampling the history
#    self._resample_history(False)
#
#    # Compute MAP estimate, and precondition by the Hessian at the MAP estimate
#    flat_sample, hessian = self._optimize_Newton_method(projection=True)
#    preconditioner = spla.pinvh(-hessian)
#    preconditioner_sqrt = spla.sqrtm(preconditioner)
#    if np.max(np.abs(np.imag(preconditioner_sqrt))) > _SMALL_NUMBER:
#      raise ValueError('Imaginary components introduced to preconditioner')
#    preconditioner_sqrt = np.real(preconditioner_sqrt)
#
#    # take step_count steps of Langevin MCMC
#    for _ in range(self.step_count):
#      gradient = self._compute_just_gradient(flat_sample)
#      scaled_gradient = preconditioner.dot(gradient)
#      scaled_noise = preconditioner_sqrt.dot(np.random.randn(flat_sample.size))
#      flat_sample = (flat_sample - self.step_size * scaled_gradient
#                     + np.sqrt(2 * self.step_size) * scaled_noise)
#      flat_sample = np.maximum(flat_sample, _SMALL_NUMBER)
#
#    edge_length = copy.deepcopy(self.internal_env.graph)
#    for start_node in edge_length:
#      for end_node in edge_length[start_node]:
#        edge_index = self.edge2index[start_node][end_node]
#        edge_length[start_node][end_node] = flat_sample[edge_index]
#
#    return edge_length
