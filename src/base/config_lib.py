"""Functions to parse experiment 'config' files for distributed experiments.

A config file is a namedtuple that includes all the:
  name, agents, environments, experiments, n_steps, n_seeds
that we will need to run to get the full results for our investigation.

`iterate_through_config` shows how we use the config file to define the cross
product of all possible experiment variants that we will want to run. We use
a lazy evaluation of the underlying agent/environment/experiment so that we can
run very large experiment sweeps without slowdown/memory bottlenecks.

For usage please see `batch_runner.py`.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import pandas as pd


Config = collections.namedtuple(
    'Config',
    ['name', 'agents', 'environments', 'experiments', 'n_steps', 'n_seeds'])


#############################################################################
# Code for loading/passing configs.


def iterate_through_config(config_in):
  """Iterator to pass through config file without evaluating.

  Args:
    config_in: Config namedtuple describing all experiments.

  Yields:
    info: a dictionary with information on each job of the experiment.
  """
  unique_id = 0
  for seed in range(config_in.n_seeds):
    for env_name, env_constructor in config_in.environments.iteritems():
      for agent_name, agent_constructor in config_in.agents.iteritems():
        for _, exp_constructor in config_in.experiments.iteritems():
          info = {'experiment_name': config_in.name,
                  'unique_id': unique_id,
                  'seed': seed,
                  'agent_name': agent_name,
                  'agent_constructor': agent_constructor,
                  'env_name': env_name,
                  'environment_constructor': env_constructor,
                  'experiment_constructor': exp_constructor}
          yield info
          unique_id += 1


def get_job_config(config_in, job_id):
  """Retrieve the config for a specific job.

  Args:
    config_in: Config namedtuple describing all experiments.
    job_id: integer identifier for which specific experiment to retrieve.

  Returns:
    exp_config: dictionary including experiment, experiment name and unique_id.

  Raises:
    ValueError: if the job_id is not found in the config, typically this means
      you have asked for a job_id larger than the total cross product of valid
      experiments.
  """
  for job_info in iterate_through_config(config_in):
    if job_id == job_info['unique_id']:
      agent = job_info['agent_constructor']()
      env = job_info['environment_constructor']()
      seed = job_info['seed']
      unique_id = job_info['unique_id']
      exp = job_info['experiment_constructor'](
          agent, env, config_in.n_steps, seed, unique_id=unique_id)
      exp_config = {
          'experiment_name': job_info['experiment_name'],
          'unique_id': unique_id,
          'experiment': exp,
      }
      return exp_config
  raise ValueError('No job_id %d found', job_id)


def get_params_df(config_in):
  """Returns a dataframe with all the possible config parameters.

  Args:
    config_in: Config namedtuple describing all experiments.

  Returns:
    params_df: Pandas dataframe mapping unique_id back to hyperparams.
  """
  config_list = []
  for job_info in iterate_through_config(config_in):
    info_row = {
        'agent': job_info['agent_name'],
        'environment': job_info['env_name'],
        'seed': job_info['seed'],
        'unique_id': job_info['unique_id']
    }
    config_list.append(info_row)
  return pd.DataFrame(config_list)
