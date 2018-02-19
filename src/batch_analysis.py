"""Analysis script to generate the plots for the TS tutorial paper.

This script assumes that you have run all of the `config` files in each of the
subfolders using `batch_runner.py`, with default experiment names and the .csv
files of the results saved in `_DATA_FILEPATH`.

All plots are saved to `_PLOT_FILEPATH`.
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

_DATA_FILEPATH = 'path/to/your/data'  # .csv files of experiments
_PLOT_FILEPATH = 'path/to/your/plot'  # where you want to save plots

bp.set_data_path(_DATA_FILEPATH)

##############################################################################

plot_dict = {}

# Most plots are simple instantaneous regret
simple_plots = ['finite_simple', 'finite_simple_rand', 'finite_simple_sanity',
                'finite_drift', 'cascade', 'graph_indep', 'graph_indep_binary',
                'graph_correlated', 'graph_correlated_sanity']
for plot_name in simple_plots:
  plot_dict.update(bp.simple_algorithm_plot(plot_name))

# Graph plots also do cumulative distance
cumulative_plots = ['graph_indep', 'graph_correlated']
for plot_name in cumulative_plots:
  plot_dict.update(bp.cumulative_travel_time_plot(plot_name))

# Comparing action selection greedy/ts
plot_dict.update(bp.compare_action_selection_plot())

# Misspecified prior plots
plot_dict.update(bp.misspecified_plot())

# Ensemble plots
plot_dict.update(bp.ensemble_plot())

# Saving all plots to file

for plot_name, p in plot_dict.iteritems():
  file_path = os.path.join(_PLOT_FILEPATH, plot_name.lower() + '.png')
  file_path = file_path.replace(' ', '_')
  if 'ensemble' in file_path:
    p.save(file_path, height=8, width=6)
  else:
    p.save(file_path, height=8, width=8)

