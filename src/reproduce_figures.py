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
# Figures are named with reference to "A Tutorial on Thompson Sampling":
#     https://arxiv.org/abs/1707.02038.

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

def _load_experiment_name_from_config(config_path):
  """Extract the identifying experiment name from config."""
  experiment_config = importlib.import_module(config_path)
  config = experiment_config.get_config()
  return config.name


def _logging(figure_options, run_frac, data_path, plot_path):
  """Logging to screen.

  Args:
    figure_options: a FigureOptions namedtuple.
    run_frac: float in [0,1] of how many jobs to run vs paper.
    data_path: where to save intermediate experiment .csv.
    plot_path: where to save output plot.

  Returns:
    experiment_name: identifying string from config file.
    n_jobs: number of jobs to run.
  """
  experiment_name = _load_experiment_name_from_config(figure_options.config)
  n_jobs = int(run_frac * figure_options.paper_n_jobs)

  # Logging to screen
  print('*' * 80)
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

  return experiment_name, n_jobs


def _save_plot_to_file(plot_dict, plot_path, run_frac=None):
  """Plots a dictionary of plotnine plots to file.

  Args:
    plot_dict: {plot_name: p} for p = plotnine plot.
    plot_path: path to save the file to.
    run_frac: optional float indicating run fraction (just for logging.)

  Returns:
    NULL, plot is written to file_path as a png file.
  """
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
  experiment_name, n_jobs = _logging(
      figure_options, run_frac, data_path, plot_path)

  # Running the jobs via command line (this can/should be parallelized)
  for i in range(n_jobs):
    print('Starting job {} out of {}'.format(i, n_jobs))
    os.system('ipython batch_runner.py --config {} --job_id {} --save_path {}'
              .format(figure_options.config, i, data_path))

  # Plotting output
  plot_dict = figure_options.plot_fun(experiment_name, data_path)
  _save_plot_to_file(plot_dict, plot_path, run_frac)


def main(fig_str, run_frac, data_path, plot_path):
  """Either runs all of the experiments, or just a single figure."""
  if fig_str == 'all':
    for fig in FIGURE_OPTIONS:
      reproduce_figure(FIGURE_OPTIONS[fig], run_frac, data_path, plot_path)
  else:
    reproduce_figure(FIGURE_OPTIONS[fig_str], run_frac, data_path, plot_path)


###############################################################################
# Main function == script wrapper.

if __name__ == '__main__':
  # Parsing command line options
  parser = argparse.ArgumentParser(description='Reproduce figures.')
  fig_help = ('Figures to reproduce. Must be one of the following options:\n'
              '{}'.format(FIGURE_OPTIONS.keys() + ['all']))
  parser.add_argument('--figure', help=fig_help, type=str, default='3')
  run_help = 'Proportion of paper experiments to run. Must be in [0, 1].'
  parser.add_argument('--run_frac', help=run_help, type=float, default=0.01)
  data_help = 'Path to store intermediate .csv files of experiment results. Must exist in OS.'
  parser.add_argument('--data_path', help=data_help, type=str, default='/tmp/')
  plot_help = 'Path to store output paper plots. Must exist in OS.'
  parser.add_argument('--plot_path', help=plot_help, type=str, default='/tmp/')
  args = parser.parse_args()

  # Checking valid command line options
  run_frac_err = (run_help
                  + '\n\tYour input run_frac={}, please try again.'
                  .format(args.run_frac))
  assert args.run_frac >= 0., run_frac_err
  assert args.run_frac <= 1., run_frac_err

  figure_err = (fig_help
                + '\n\tYour input figure={}, please try again.'
                .format(args.figure))
  assert args.figure in set(FIGURE_OPTIONS.keys() + ['all']), figure_err

  dat_err =  (data_help
              + '\n\tYour input data_path={}, please try again.'
              .format(args.data_path))
  assert os.path.isdir(args.data_path), data_err

  plot_err = (plot_help
              + '\n\tYour input data_path={}, please try again.'
              .format(args.plot_path))
  assert os.path.isdir(args.plot_path), plot_err

  # Logging to screen
  print('*' * 80)
  print('Parsing command line fig={}, run_frac={}, data_path={}, save_path={}'
        .format(args.figure, args.run_frac, args.data_path, args.plot_path))
  print('WARNING - this can take a long time on a single machine... you may want to parallelize the jobs.\n')

  # Running jobs
  main(args.figure, args.run_frac, args.data_path, args.plot_path)




