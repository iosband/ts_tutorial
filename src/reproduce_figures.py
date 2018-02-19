"""Script interface to reproduce figures from https://arxiv.org/abs/1707.02038


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import importlib
import collections
import os
import sys

import numpy as np
import pandas as pd
import plotnine as gg
import base.plot as bp

from base import config_lib

sys.path.append(os.getcwd())
gg.theme_set(gg.theme_bw(base_size=16, base_family='serif'))


# FIGURE_OPTIONS will hold all of the details for specific details to reproduce
# each figure. These include the config, number of jobs and the plot function.

FigureOptions = collections.namedtuple(
    'FigureOptions', ['fig_name', 'config','paper_n_jobs', 'plot_fun'])

FIGURE_OPTIONS = collections.OrderedDict([
    ['3', FigureOptions(fig_name='3',
                        config='finite_arm.config_simple',
                        paper_n_jobs=20000,
                        plot_fun=bp.compare_action_selection_plot)],
    ['4a', FigureOptions(fig_name='4a',
                         config='finite_arm.config_simple',
                         paper_n_jobs=20000,
                         plot_fun=bp.simple_algorithm_plot)],
    ['4b', FigureOptions(fig_name='4b',
                         config='finite_arm.config_simple_rand',
                         paper_n_jobs=20000,
                         plot_fun=bp.simple_algorithm_plot)],
    ['6', FigureOptions(fig_name='6',
                        config='graph.config_indep',
                        paper_n_jobs=5000,
                        plot_fun=bp.cumulative_travel_time_plot)],
    ['7', FigureOptions(fig_name='7',
                        config='graph.config_correlated',
                        paper_n_jobs=3000,
                        plot_fun=bp.cumulative_travel_time_plot)],
    ['8', FigureOptions(fig_name='8',
                        config='graph.config_indep_binary',
                        paper_n_jobs=2000,
                        plot_fun=bp.simple_algorithm_plot)],
    ['9a', FigureOptions(fig_name='9a',
                         config='finite_arm.config_simple_sanity',
                         paper_n_jobs=30000,
                         plot_fun=bp.simple_algorithm_plot)],
    ['9b', FigureOptions(fig_name='9b',
                         config='graph.config_correlated_sanity',
                         paper_n_jobs=3000,
                         plot_fun=bp.simple_algorithm_plot)],
    ['11', FigureOptions(fig_name='11',
                         config='finite_arm.config_misspecified',
                         paper_n_jobs=20000,
                         plot_fun=bp.misspecified_plot)],
    ['12', FigureOptions(fig_name='12',
                         config='finite_arm.config_drift',
                         paper_n_jobs=20000,
                         plot_fun=bp.simple_algorithm_plot)],
    ['13', FigureOptions(fig_name='13',
                         config='pricing.config_pricing',
                         paper_n_jobs=2000,
                         plot_fun=bp.simple_algorithm_plot)],
    ['17', FigureOptions(fig_name='17',
                         config='ensemble_nn.config_nn',
                         paper_n_jobs=20000,
                         plot_fun=bp.ensemble_plot)],
])


###############################################################################
# Functions to reproduce each figure in the tutorial paper

def _save_plot_to_file(plot_dict, plot_path, run_frac=None):
  """Plots a dictionary of plotnine plots to file."""
  for plot_name, p in plot_dict.iteritems():
    file_path = os.path.join(plot_path, plot_name.lower() + '.png')
    file_path = file_path.replace(' ', '_')
    print('*' * 80)
    print('Saving final plot to ' + file_path)
    if run_frac is not None:
      print('This may not *precisely* match the paper due to run_frac {}'
            .format(run_frac))
    print('*' * 80)
    print('\n\n\n')
    if 'ensemble' in file_path:
      p.save(file_path, height=8, width=6)
    else:
      p.save(file_path, height=8, width=8)


def _load_experiment_name_from_config(config_path):
  experiment_config = importlib.import_module(config_path)
  config = experiment_config.get_config()
  return config.name


def reproduce_figure(figure_options, run_frac, data_path, plot_path):
  """Function to reproduce figures for TS tutorial.

  Args:
    figure_options: a FigureOptions namedtuple.
    run_frac: float in [0,1] of how many jobs to run vs paper.
    data_path: where to save intermediate experiment .csv.
    plot_path: where to save output plot.

  Returns:
    None, experiment results are written to data_path and plots to plot_path.
  """
  experiment_name = _load_experiment_name_from_config(figure_options.config)
  n_jobs = int(run_frac * figure_options.paper_n_jobs)

  # Logging to screen
  print('Reproducing Figure {}, from TS Tutorial https://arxiv.org/abs/1707.02038'
         .format(figure_options.fig_name))
  print('With run_frac {} this will launch {} jobs, compared to {} in the paper.'
        .format(run_frac, n_jobs, figure_options.paper_n_jobs))
  print('The config file with all necessary details of the underlying'
        ' experiment is \n   {}.'.format(figure_options.config))
  print('The experiment results are saved to {} with experiment_id {}.'
        .format(data_path, experiment_name))
  print('The output plots are saved to {} with experiment_id {}.'
        .format(plot_path, experiment_name))
  print('\n')
  print('*' * 80)

  # Running the jobs via command line (this can/should be parallelized)
  for i in range(n_jobs):
    print('Starting job {} out of {}'.format(i, n_jobs))
    os.system('ipython batch_runner.py --config {} --job_id {} --save_path {}'
              .format(figure_options.config, i, data_path))

  # Plotting output
  plot_dict = figure_options.plot_fun(experiment_name, data_path)
  _save_plot_to_file(plot_dict, plot_path, run_frac)



###############################################################################
# Main function == script wrapper.

if __name__ == '__main__':
  # Parsing command line options
  parser = argparse.ArgumentParser(description='Reproduce figures.')
  fig_help = ('Figures to reproduce, must be one of the following options:\n'
              '{}'.format(FIGURE_OPTIONS.keys() + ['all']))
  parser.add_argument('--figure', help=fig_help, type=str, default='3')
  run_help = 'Proportion of paper experiments to run in [0, 1]'
  parser.add_argument('--run_frac', help=run_help, type=float, default=0.01)
  data_help = 'Path to store intermediate .csv files of experiment results.'
  parser.add_argument('--data_path', help=data_help, type=str, default='/tmp/')
  plot_help = 'Path to store output paper plots.'
  parser.add_argument('--plot_path', help=plot_help, type=str, default='/tmp/')
  args = parser.parse_args()

  # Checking valid command line options
  assert args.run_frac >= 0.
  assert args.run_frac <= 1.
  assert args.figure in set(FIGURE_OPTIONS.keys() + ['all'])
  assert os.path.isdir(args.data_path)
  assert os.path.isdir(args.plot_path)

  # Logging to screen
  print('*' * 80)
  print('Parsing command line fig={}, run_frac={}, data_path={}, save_path={}'
        .format(args.figure, args.run_frac, args.data_path, args.plot_path))
  print('WARNING - this can take a long time on a single machine... you may want to parallelize the jobs.\n')

  # Running jobs
  if args.figure == 'all':
    for fig in FIGURE_OPTIONS:
      reproduce_figure(FIGURE_OPTIONS[fig], args.run_frac,
                       args.data_path, args.plot_path)

  else:
    reproduce_figure(FIGURE_OPTIONS[args.figure], args.run_frac,
                     args.data_path, args.plot_path)





