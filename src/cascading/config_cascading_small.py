"""Specify the jobs to run via config file.

Binomial bridge bandit experiment with independent segments.
Compare the performance of Thompson sampling with different egreedy.
See Figure 6 https://arxiv.org/pdf/1707.02038.pdf
"""
import collections
import functools


from base.config_lib import Config
from base.experiment import ExperimentNoAction
from cascading.agent_cascading import CascadingBanditKLUCB
from cascading.agent_cascading import CascadingBanditTS
from cascading.agent_cascading import CascadingBanditUCB1
from cascading.env_cascading import CascadingBandit


def get_config():
  """Generates the config for the experiment."""
  name = 'cascade'
  num_items = 50
  num_positions = 10
  true_a0 = 1
  true_b0 = 40
  best_optimism = 0.1

  def _ts_init(num_items, num_positions):
    agent = CascadingBanditTS(
        num_items, num_positions, a0=true_a0, b0=true_b0)
    return agent

  def _ucb1_init(num_items, num_positions):
    agent = CascadingBanditUCB1(
        num_items, num_positions, a0=true_a0, b0=true_b0, optimism=1)
    return agent

  def _ucb_best_init(num_items, num_positions):
    agent = CascadingBanditUCB1(
        num_items, num_positions, a0=true_a0, b0=true_b0, optimism=best_optimism)
    return agent

  agents = collections.OrderedDict(
      [('ts',
        functools.partial(_ts_init, num_items, num_positions)),
       ('ucb1',
        functools.partial(_ucb1_init, num_items, num_positions)),
       ('ucb-best',
        functools.partial(_ucb_best_init, num_items, num_positions))]
  )

  environments = collections.OrderedDict(
      [('env',
        functools.partial(CascadingBandit,
                          num_items, num_positions, true_a0, true_b0))]
  )

  # Very large experiment so don't log as frequently to keep file sensible.
  experiments = collections.OrderedDict(
      [(name,
        functools.partial(ExperimentNoAction, rec_freq=10))]
  )
  n_steps = 5000
  n_seeds = 1000
  config = Config(name, agents, environments, experiments, n_steps, n_seeds)
  return config
