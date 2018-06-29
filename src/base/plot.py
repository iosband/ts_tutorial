"""Common scripts for plotting/analysing the results of experiments.

This code is designed to work with the .csv files that are output by
`batch_runner.py`.

Some of this code is generic, but a lot of it is designed specifically to
generate the plots that are used in the TS tutorial paper. For usage in
generating these plots see `batch_analysis.py`.
"""

from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import pandas as pd
import plotnine as gg

sys.path.append(os.getcwd())
gg.theme_set(gg.theme_bw(base_size=16, base_family='serif'))
gg.theme_update(figure_size=(10, 8))

_DEFAULT_DATA_PATH = '/path/to/your/data'
_DATA_CACHE = {}

#############################################################################
# Loading data

def set_data_path(file_path):
  """Overwrite globale default data path."""
  _DEFAULT_DATA_PATH = file_path


def _name_cleaner(agent_name):
  """Renames agent_name to prettier string for plots."""
  rename_dict = {'correct_ts': 'Correct TS',
                 'kl_ucb': 'KL UCB',
                 'misspecified_ts': 'Misspecified TS',
                 'ucb1': 'UCB1',
                 'nonstationary_ts': 'Nonstationary TS',
                 'stationary_ts': 'Stationary TS',
                 'greedy': 'Greedy',
                 'ts': 'TS',
                 'action_0': 'Action 0',
                 'action_1': 'Action 1',
                 'action_2': 'Action 2',
                 'bootstrap': 'bootstrap TS',
                 'laplace': 'Laplace TS',
                 'thoughtful': 'Thoughtful TS',
                 'gibbs': 'Gibbs TS'}
  if agent_name in rename_dict:
    return rename_dict[agent_name]
  else:
    return agent_name


def load_data(experiment_name, data_path=_DEFAULT_DATA_PATH):
  """Function to load in the data relevant to a specific experiment.

  Args:
    experiment_name: string = name of experiment config.
    data_path: string = where to look for the files.

  Returns:
    df: dataframe of experiment data (uses cache for faster reloading).
  """
  if experiment_name in _DATA_CACHE:
    return _DATA_CACHE[experiment_name]
  else:
    all_files = os.listdir(data_path)
    good_files = []
    for file_name in all_files:
      if '.csv' not in file_name:
        continue
      else:
        file_experiment = file_name.split('exp=')[1].split('|')[0]
        if file_experiment == experiment_name:
          good_files.append(file_name)

    data = []
    for file_name in good_files:
      file_path = os.path.join(data_path, file_name)
      if 'id=' in file_name:
        if os.path.getsize(file_path) < 1024:
          continue
        else:
          data.append(pd.read_csv(file_path))
      elif 'params' in file_name:
        params_df = pd.read_csv(file_path)
        params_df['agent'] = params_df['agent'].apply(_name_cleaner)
      else:
        raise ValueError('Something is wrong with file names.')

    df = pd.concat(data)
    df = pd.merge(df, params_df, on='unique_id')
    _DATA_CACHE[experiment_name] = df
    return _DATA_CACHE[experiment_name]


#############################################################################
# Basic instant regret plots


def simple_algorithm_plot(experiment_name, data_path=_DEFAULT_DATA_PATH):
  """Simple plot of average instantaneous regret by agent, per timestep.

  Args:
    experiment_name: string = name of experiment config.
    data_path: string = where to look for the files.

  Returns:
    https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
  """
  df = load_data(experiment_name, data_path)
  plt_df = (df.groupby(['t', 'agent'])
            .agg({'instant_regret': np.mean})
            .reset_index())
  p = (gg.ggplot(plt_df)
       + gg.aes('t', 'instant_regret', colour='agent')
       + gg.geom_line(size=1.25, alpha=0.75)
       + gg.xlab('time period (t)')
       + gg.ylab('per-period regret')
       + gg.scale_colour_brewer(name='agent', type='qual', palette='Set1'))
  
  plot_dict = {experiment_name + '_simple': p}
  return plot_dict


def cumulative_travel_time_plot(experiment_name, data_path=_DEFAULT_DATA_PATH):
  """Plot cumulative ratio total travel time relative to optimal shortest path.

  Args:
    experiment_name: string = name of experiment config.
    data_path: string = where to look for the files.

  Returns:
    https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf
  """
  df = load_data(experiment_name, data_path)
  df['cum_ratio'] = (df.cum_optimal - df.cum_regret) / df.cum_optimal
  plt_df = (df.groupby(['t', 'agent'])
            .agg({'cum_ratio': np.mean})
            .reset_index())
  p = (gg.ggplot(plt_df)
       + gg.aes('t', 'cum_ratio', colour='agent')
       + gg.geom_line(size=1.25, alpha=0.75)
       + gg.xlab('time period (t)')
       + gg.ylab('Total distance / optimal')
       + gg.scale_colour_brewer(name='agent', type='qual', palette='Set1')
       + gg.aes(ymin=1)
       + gg.geom_hline(yintercept=1, linetype='dashed', size=2, alpha=0.5))
  
  plot_dict = {experiment_name + '_cum': p}
  return plot_dict


#############################################################################
# Action proportion plots

def plot_action_proportion(df_agent):
  """Plot the action proportion for the sub-dataframe for a single agent."""
  n_action = np.max(df_agent.action) + 1
  plt_data = []
  for i in range(n_action):
    probs = (df_agent.groupby('t')
             .agg({'action': lambda x: np.mean(x == i)})
             .rename(columns={'action': 'action_' + str(i)}))
    plt_data.append(probs)
  plt_df = pd.concat(plt_data, axis=1).reset_index()
  p = (gg.ggplot(pd.melt(plt_df, id_vars='t'))
       + gg.aes('t', 'value', colour='variable', group='variable')
       + gg.geom_line(size=1.25, alpha=0.75)
       + gg.xlab('time period (t)')
       + gg.ylab('Action probability')
       + gg.ylim(0, 1)
       + gg.scale_colour_brewer(name='Variable', type='qual', palette='Set1'))
  return p

def compare_action_selection_plot(experiment_name='finite_simple',
                                  data_path=_DEFAULT_DATA_PATH):
  """Specialized plotting script for TS tutorial paper action proportion."""
  df = load_data(experiment_name, data_path)
  plot_dict = {}
  for agent, df_agent in df.groupby(['agent']):
    key_name = experiment_name + '_' + agent + '_action'
    plot_dict[key_name] = plot_action_proportion(df_agent)
  return plot_dict


#############################################################################
# Misspecified prior plots

def misspecified_plot(experiment_name='finite_misspecified',
                      data_path=_DEFAULT_DATA_PATH):
  """Specialized plotting script for TS tutorial paper misspecified TS."""
  df = load_data(experiment_name, data_path)

  def _parse_np_array(np_string):
    return np.array(np_string.replace('[', '')
                    .replace(']', '')
                    .strip()
                    .split())
  df['posterior_mean'] = df.posterior_mean.apply(_parse_np_array)

  # Action means
  new_col_list = ['mean_0', 'mean_1', 'mean_2']
  for n, col in enumerate(new_col_list):
    df[col] = df['posterior_mean'].apply(lambda x: float(x[n]))

  plt_df = (df.groupby(['agent', 't'])
            .agg({'instant_regret': np.mean,
                  'mean_0': np.mean,
                  'mean_1': np.mean,
                  'mean_2': np.mean})
            .reset_index())

  regret_plot = (gg.ggplot(plt_df)
                 + gg.aes('t', 'instant_regret', colour='agent')
                 + gg.geom_line(size=1.25, alpha=0.75)
                 + gg.xlab('time period (t)')
                 + gg.ylab('per-period regret')
                 + gg.scale_colour_brewer(name='agent', type='qual', palette='Set1')
                 + gg.coord_cartesian(ylim=(0, 0.02)))

  melt_df = pd.melt(plt_df, id_vars=['agent', 't'], value_vars=new_col_list)
  melt_df['group_id'] = melt_df.agent + melt_df.variable
  action_plot = (gg.ggplot(melt_df)
                 + gg.aes('t', 'value', colour='agent', group='group_id')
                 + gg.geom_line(size=1.25, alpha=0.75)
                 + gg.coord_cartesian(ylim=(0, 0.05))
                 + gg.xlab('time period (t)')
                 + gg.ylab('Expected mean reward')
                 + gg.scale_colour_brewer(name='agent', type='qual', palette='Set1'))

  plot_dict = {}
  plot_dict['misspecified_regret'] = regret_plot
  plot_dict['misspecified_action'] = action_plot
  return plot_dict


#############################################################################
# Ensemble neural network plotting


def ensemble_plot(experiment_name='ensemble_nn', data_path=_DEFAULT_DATA_PATH):
  """Specialized plotting script for TS tutorial paper ensemble NN."""
  df = load_data(experiment_name, data_path)
  plt_df = (df.groupby(['agent', 't'])
            .agg({'instant_regret': np.mean})
            .reset_index())

  def _get_agent_family(agent_name):
    if 'dropout' in agent_name.lower():
      return 'Dropout'
    elif 'ensemble' in agent_name.lower():
      return 'Ensemble'
    elif '/' in agent_name.lower():
      return 'Annealing epsilon'
    else:
      return 'Fixed epsilon'

  def _rename_ensemble(agent_name):
    if 'ensemble' in agent_name:
      n_ensemble = agent_name.split('-')[0]
      new_name = 'ensemble=' + n_ensemble.zfill(3)

      return new_name
    else:
      return agent_name

  plt_df['agent_name'] = plt_df.agent.apply(_rename_ensemble)
  plt_df['agent_family'] = plt_df.agent.apply(_get_agent_family)

  custom_colors = ['#d53e4f', '#fdae61', '#a6d96a', '#66c2a5', '#5e4fa2']

  plot_dict = {}
  for agent_family, df_family in plt_df.groupby(['agent_family']):
    if agent_family == 'Ensemble':
      custom_labels = ['Ensemble 3', 'Ensemble 10', 'Ensemble 30',
                       'Ensemble 100', 'Ensemble 300']
      gg_legend = gg.scale_colour_manual(values=custom_colors,
                                         labels=custom_labels,
                                         name='agent')
    else:
      gg_legend = gg.scale_colour_manual(custom_colors, name='agent')

    p = (gg.ggplot(df_family)
         + gg.aes('t', 'instant_regret', colour='agent_name')
         + gg.geom_line(size=1.25, alpha=0.75)
         + gg.facet_wrap('~ agent_family')
         + gg_legend
         + gg.coord_cartesian(ylim=(0, 60))
         + gg.xlab('time period (t)')
         + gg.ylab('per-period regret')
         + gg.theme(figure_size=(6, 6)))
    plot_dict[experiment_name + '_' + agent_family] = p

  return plot_dict

###############################################################################
# plotting concurrent agents regret
 
def get_agent_id(agent_name):
  letters = {1:'a',10:'b',20:'c',50:'d',100:'e'}
  agent_num = int(agent_name.split(' = ')[1])
  return letters[agent_num]

def concurrent_agents_plot(experiment_name='graph_indep_concurrent', 
                           data_path=_DEFAULT_DATA_PATH, paper_version=True):
  '''Passing paper_version=True should be used to reproduce Fig. 14 of the paper 
  for K = 1,10,20,50,100. In this case, the labels in the legend are manually 
  ordered by the values of K. Otherwise, the labels are ordered alphabetically.'''
  
  df = load_data(data_path, experiment_name)
  
  plt_df_per_action = (df.groupby(['agent','t','agent_id','action_id'])
            .agg({'instant_regret': np.mean})
            .reset_index())
  
  plt_df_per_period = (df.groupby(['agent','t'])
            .agg({'instant_regret': np.mean})
            .reset_index())
  
  if not paper_version:
    p_per_action = (gg.ggplot(plt_df_per_action)
       + gg.aes('action_id', 'instant_regret', colour='agent')
       + gg.geom_line()+ gg.geom_line(size=1.25, alpha=0.75)
       + gg.xlim(0, 2.5*len(plt_df_per_period.groupby('t')))
       + gg.scale_colour_brewer(name='agent', type='qual', palette='Set1')
       + gg.labels.xlab('number of actions')
       + gg.labels.ylab('per-period regret'))
    
    p_per_period = (gg.ggplot(plt_df_per_period)
       + gg.aes('t', 'instant_regret', colour='agent')
       + gg.geom_line()+ gg.geom_line(size=1.25, alpha=0.75)
       + gg.scale_colour_brewer(name='agent', type='qual', palette='Set1')
       + gg.labels.xlab('time period (t)')
       + gg.labels.ylab('per-period regret'))
  else:
    plt_df_per_action['agent_id'] = plt_df_per_action.agent.apply(get_agent_id)
    plt_df_per_period['agent_id'] = plt_df_per_period.agent.apply(get_agent_id)
    
    custom_labels = ['K = 1','K = 10','K = 20','K = 50','K = 100']
    custom_colors = ["#E41A1C","#377EB8","#4DAF4A","#984EA3","#FF7F00"]
    
    p_per_action = (gg.ggplot(plt_df_per_action)
      + gg.aes('action_id', 'instant_regret', colour='agent_id')
      + gg.geom_line()+ gg.geom_line(size=1.25, alpha=0.75)
      + gg.xlim(0, 2.5*len(plt_df_per_period.groupby('t')))
      + gg.scale_color_manual(name='agent', labels = custom_labels,values=custom_colors)
      + gg.labels.xlab('number of actions')
      + gg.labels.ylab('per-action regret'))
  
    p_per_period = (gg.ggplot(plt_df_per_period)
      + gg.aes('t', 'instant_regret', colour='agent_id')
      + gg.geom_line()+ gg.geom_line(size=1.25, alpha=0.75)
      + gg.scale_color_manual(name='agent', labels = custom_labels,values=custom_colors)
      + gg.labels.xlab('time period (t)')
      + gg.labels.ylab('per-period regret'))
  
  plot_dict = {}
  plot_dict['per_action_plot'] = p_per_action
  plot_dict['per_period_plot'] = p_per_period
  return plot_dict 

##############################################################################
# generating Fig. 10(a) of the paper, making it consistent with Fig. 10(b)
def rename_agent(name):
  if name=='Laplace TS':
    return 'a'
  elif name=='Langevin TS':
    return 'b'
  elif name=='TS':
    return 'c'
  elif name=='bootstrap TS':
    return 'd'
  else:
    raise ValueError('There is an unknown agent. Perhaps you need to use other plotting functions.') 
  
  
def customized_algorithm_plot(experiment_name='finite_simple_sanity', data_path=_DEFAULT_DATA_PATH):
  """Simple plot of average instantaneous regret by agent, per timestep.

  Args:
    experiment_name: string = name of experiment config.
    data_path: string = where to look for the files.

  Returns:
    p: ggplot plot
  """
  df = load_data(experiment_name, data_path)
  plt_df = (df.groupby(['t', 'agent'])
            .agg({'instant_regret': np.mean})
            .reset_index())
  plt_df['agent_new_name'] = plt_df.agent.apply(rename_agent)
    
  custom_labels = ['Laplace TS','Langevin TS','TS','bootstrap TS']
  custom_colors = ["#E41A1C","#377EB8","#4DAF4A","#984EA3"]
  
  p = (gg.ggplot(plt_df)
       + gg.aes('t', 'instant_regret', colour='agent_new_name')
       + gg.geom_line(size=1.25, alpha=0.75)
       + gg.xlab('time period (t)')
       + gg.ylab('per-period regret')
       + gg.scale_color_manual(name='agent', labels = custom_labels,values=custom_colors))
  return p
