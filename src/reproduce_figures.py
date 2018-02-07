"""Script interface to reproduce figures from https://arxiv.org/abs/1707.02038


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import pandas as pd
import plotnine as gg
import base.plot as bp

sys.path.append(os.getcwd())
gg.theme_set(gg.theme_bw(base_size=16, base_family='serif'))


# Fraction of total jobs to run relative to TS Tutorial paper (must be <= 1)
# RUN_FRACTION < 1 will allow for faster runtime but at the cost of noise.
RUN_FRACTION = 1.0
SAVE_PATH = '/tmp/'
PLOT_PATH = '/tmp/'


def print_to_file(plot_dict, plot_path=PLOT_PATH):
  for plot_name, p in plot_dict.iteritems():
    file_path = os.path.join(plot_path, plot_name.lower() + '.png')
    file_path = file_path.replace(' ', '_')
    if 'ensemble' in file_path:
      p.save(file_path, height=8, width=6)
    else:
      p.save(file_path, height=8, width=8)


def reproduce_figure_3(run_fraction=RUN_FRACTION, save_path=SAVE_PATH):
  n_jobs = int(20000 * run_fraction)
  config_path = 'finite_arm.config_simple'
  for i in range(n_jobs):
    os.system('ipython batch_runner.py --config {} --job_id {} --save_path {}'
              .format(config_path, i, save_path))

  plot_dict = {}
  plot_dict.update(bp.compare_action_selection_plot())
  print_to_file(plot_dict)


def reproduce_figure_4a(run_fraction=RUN_FRACTION, save_path=SAVE_PATH):
  n_jobs = int(20000 * run_fraction)
  config_path = 'finite_arm.config_simple'
  plot_name = 'finite_simple'
  for i in range(n_jobs):
    os.system('ipython batch_runner.py --config {} --job_id {} --save_path {}'
              .format(config_path, i, save_path))

  plot_dict = {plot_name: bp.simple_algorithm_plot(plot_name)}
  print_to_file(plot_dict)


def reproduce_figure_4b(run_fraction=RUN_FRACTION, save_path=SAVE_PATH):
  n_jobs = int(20000 * run_fraction)
  config_path = 'finite_arm.config_simple_rand'
  plot_name = 'finite_simple_rand'
  for i in range(n_jobs):
    os.system('ipython batch_runner.py --config {} --job_id {} --save_path {}'
              .format(config_path, i, save_path))

  plot_dict = {plot_name: bp.simple_algorithm_plot(plot_name)}
  print_to_file(plot_dict)


def reproduce_figure_6(run_fraction=RUN_FRACTION, save_path=SAVE_PATH):
  n_jobs = int(5000 * run_fraction)
  config_path = 'graph.config_indep'
  plot_name = 'graph_indep'
  for i in range(n_jobs):
    os.system('ipython batch_runner.py --config {} --job_id {} --save_path {}'
              .format(config_path, i, save_path))

  plot_dict = {plot_name: bp.simple_algorithm_plot(plot_name)}
  plot_dict[plot_name + '_cum'] = bp.cumulative_travel_time_plot(plot_name)
  print_to_file(plot_dict)


def reproduce_figure_7(run_fraction=RUN_FRACTION, save_path=SAVE_PATH):
  n_jobs = int(3000 * run_fraction)
  config_path = 'graph.config_correlated'
  plot_name = 'graph_correlated'
  for i in range(n_jobs):
    os.system('ipython batch_runner.py --config {} --job_id {} --save_path {}'
              .format(config_path, i, save_path))

  plot_dict = {plot_name: bp.simple_algorithm_plot(plot_name)}
  plot_dict[plot_name + '_cum'] = bp.cumulative_travel_time_plot(plot_name)
  print_to_file(plot_dict)


def reproduce_figure_8(run_fraction=RUN_FRACTION, save_path=SAVE_PATH):
  n_jobs = int(2000 * run_fraction)
  config_path = 'graph.config_indep_binary'
  plot_name = 'graph_indep_binary'
  for i in range(n_jobs):
    os.system('ipython batch_runner.py --config {} --job_id {} --save_path {}'
              .format(config_path, i, save_path))

  plot_dict = {plot_name: bp.simple_algorithm_plot(plot_name)}
  print_to_file(plot_dict)


def reproduce_figure_9a(run_fraction=RUN_FRACTION, save_path=SAVE_PATH):
  n_jobs = int(30000 * run_fraction)
  config_path = 'finite_arm.config_simple_sanity'
  plot_name = 'finite_simple_sanity'
  for i in range(n_jobs):
    os.system('ipython batch_runner.py --config {} --job_id {} --save_path {}'
              .format(config_path, i, save_path))

  plot_dict = {plot_name: bp.simple_algorithm_plot(plot_name)}
  print_to_file(plot_dict)


def reproduce_figure_9b(run_fraction=RUN_FRACTION, save_path=SAVE_PATH):
  n_jobs = int(3000 * run_fraction)
  config_path = 'graph.config_correlated_sanity'
  plot_name = 'graph_correlated_sanity'
  for i in range(n_jobs):
    os.system('ipython batch_runner.py --config {} --job_id {} --save_path {}'
              .format(config_path, i, save_path))

  plot_dict = {plot_name: bp.simple_algorithm_plot(plot_name)}
  print_to_file(plot_dict)


def reproduce_figure_11(run_fraction=RUN_FRACTION, save_path=SAVE_PATH):
  n_jobs = int(20000 * run_fraction)
  config_path = 'finite_arm.config_misspecified'
  plot_name = 'finite_misspecified'
  for i in range(n_jobs):
    os.system('ipython batch_runner.py --config {} --job_id {} --save_path {}'
              .format(config_path, i, save_path))

  plot_dict = {}
  plot_dict.update(bp.misspecified_plot())
  print_to_file(plot_dict)


def reproduce_figure_12(run_fraction=RUN_FRACTION, save_path=SAVE_PATH):
  n_jobs = int(20000 * run_fraction)
  config_path = 'finite_arm.config_drift'
  plot_name = 'finite_drift'
  for i in range(n_jobs):
    os.system('ipython batch_runner.py --config {} --job_id {} --save_path {}'
              .format(config_path, i, save_path))

  plot_dict = {}
  plot_dict[plot_name] = bp.simple_algorithm_plot(plot_name)
  print_to_file(plot_dict)


def reproduce_figure_13(run_fraction=RUN_FRACTION, save_path=SAVE_PATH):
  n_jobs = int(2000 * run_fraction)
  config_path = 'pricing.config_pricing'
  plot_name = 'dynamic_pricing'
  for i in range(n_jobs):
    os.system('ipython batch_runner.py --config {} --job_id {} --save_path {}'
              .format(config_path, i, save_path))

  plot_dict = {}
  plot_dict[plot_name] = bp.simple_algorithm_plot(plot_name)
  print_to_file(plot_dict)


### Cascading figures are not included at this point ###


def reproduce_figure_17(run_fraction=RUN_FRACTION, save_path=SAVE_PATH):
  n_jobs = int(20000 * run_fraction)
  config_path = 'ensemble_nn.config_nn'
  plot_name = 'ensemble_nn'
  for i in range(n_jobs):
    os.system('ipython batch_runner.py --config {} --job_id {} --save_path {}'
              .format(config_path, i, save_path))

  plot_dict = {}
  plot_dict.update(bp.ensemble_plot())
  print_to_file(plot_dict)



### Figure 19 is generated using TabulaRL https://github.com/iosband/TabulaRL







