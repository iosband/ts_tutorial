"""Specify the jobs to run via config file.

Neural net bandit experiment.
The mapping from actions to observations is represented by a two-layer
neural network. We compare the following algorithms: epsilon-greedy (fixed
epsilon and annealing epsilon), dropout (arXiv:1506.02142), and ensemble
sampling.
"""

import collections
import functools

from base.config_lib import Config
from base.experiment import ExperimentNoAction
from ensemble_nn.env_nn import TwoLayerNNBandit
from ensemble_nn.agent_nn import TwoLayerNNEpsilonGreedy as EpsilonGreedy
from ensemble_nn.agent_nn import TwoLayerNNEpsilonGreedyAnnealing as EpsilonAnnealing
from ensemble_nn.agent_nn import TwoLayerNNDropout
from ensemble_nn.agent_nn import TwoLayerNNEnsembleSampling as EnsembleSampling


def get_config():
  """Generates the config for the experiment."""
  name = 'ensemble_nn'

  input_dim = 100
  hidden_dim = 50
  num_actions = 100
  prior_var = 1.
  noise_var = 100.

  # We have to do something weird since ensemble_nn requires construction of
  # the environment before the agent in order to specify the actions.
  agents = collections.OrderedDict(
      [('epsilon=0.01',
        lambda: functools.partial(EpsilonGreedy, epsilon_param=0.01)),
       ('epsilon=0.05',
        lambda: functools.partial(EpsilonGreedy, epsilon_param=0.05)),
       ('epsilon=0.1',
        lambda: functools.partial(EpsilonGreedy, epsilon_param=0.1)),
       ('epsilon=0.2',
        lambda: functools.partial(EpsilonGreedy, epsilon_param=0.2)),
       ('epsilon=0.3',
        lambda: functools.partial(EpsilonGreedy, epsilon_param=0.3)),
       ('epsilon=10/(10+t)',
        lambda: functools.partial(EpsilonAnnealing, epsilon_param=10.)),
       ('epsilon=20/(20+t)',
        lambda: functools.partial(EpsilonAnnealing, epsilon_param=20.)),
       ('epsilon=30/(30+t)',
        lambda: functools.partial(EpsilonAnnealing, epsilon_param=30.)),
       ('epsilon=40/(40+t)',
        lambda: functools.partial(EpsilonAnnealing, epsilon_param=40.)),
       ('epsilon=50/(50+t)',
        lambda: functools.partial(EpsilonAnnealing, epsilon_param=50.)),
       ('ensemble=1',
        lambda: functools.partial(EnsembleSampling, num_models=3)),
       ('ensemble=10',
        lambda: functools.partial(EnsembleSampling, num_models=10)),
       ('ensemble=30',
        lambda: functools.partial(EnsembleSampling, num_models=30)),
       ('ensemble=100',
        lambda: functools.partial(EnsembleSampling, num_models=100)),
       ('ensemble=300',
        lambda: functools.partial(EnsembleSampling, num_models=300)),
       ('dropout=0.1',
        lambda: functools.partial(TwoLayerNNDropout, drop_prob=0.1)),
       ('dropout=0.25',
        lambda: functools.partial(TwoLayerNNDropout, drop_prob=0.25)),
       ('dropout=0.5',
        lambda: functools.partial(TwoLayerNNDropout, drop_prob=0.5)),
       ('dropout=0.75',
        lambda: functools.partial(TwoLayerNNDropout, drop_prob=0.75)),
       ('dropout=0.9',
        lambda: functools.partial(TwoLayerNNDropout, drop_prob=0.9))]
  )

  # Similarly we do not actually evaluate the environment since we need a custom
  # experiment function.
  def _custom_partial_nn():
    f = functools.partial(TwoLayerNNBandit, input_dim, hidden_dim,
                          num_actions, prior_var, noise_var)
    return f
  environments = collections.OrderedDict([('env', _custom_partial_nn)])

  n_steps = 1000
  n_seeds = 1000

  def _env_constructor(agent_lambda, env_lambda, n_steps, seed, unique_id):
    """Constructor for neural network experiments.

    This is more involved than other configs since the construction of the
    actors requires specification of the environment. We could/should improve
    on this with a code refactor, but leaving it now since it's working.
    """
    environment = env_lambda(seed=seed)
    actions = environment.get_actions()
    agent = agent_lambda(input_dim, hidden_dim, actions,
                         n_steps, prior_var, noise_var)
    experiment = ExperimentNoAction(
        agent, environment, n_steps, seed, unique_id=unique_id)
    return experiment

  experiments = collections.OrderedDict([(name, _env_constructor)])

  config = Config(name, agents, environments, experiments, n_steps, n_seeds)
  return config
