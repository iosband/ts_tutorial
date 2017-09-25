"""Cascading bandit experiment.

Compare the performance of benchmark bandit algorithms on a cascading bandit.
See Figure 14 of https://arxiv.org/pdf/1707.02038.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from base.experiment import BaseExperiment
from base.plot import ResultsPlotter

from cascading.agent_cascading import CascadingBanditUCB1
from cascading.agent_cascading import CascadingBanditKLUCB
from cascading.agent_cascading import CascadingBanditTS

from cascading.env_cascading import CascadingBandit

num_items = 50
num_positions = 10
true_prior_a0 = 1
true_prior_b0 = 10

def correct_ts_init(num_items, num_positions):
  agent = CascadingBanditTS(num_items, num_positions,
                            a0=true_prior_a0, b0=true_prior_b0)
  return agent

agent_constructors = {'correct_ts': correct_ts_init,
                      'misspecified_ts': CascadingBanditTS,
                      'ucb1': CascadingBanditUCB1,
                      'kl_ucb': CascadingBanditKLUCB
                      }


# Running experiments (can be evaluated in parallel)
n_steps = 5000
n_seeds = 100

results = []
for agent_name, agent_constructor in agent_constructors.iteritems():
  for seed in range(n_seeds):
    environment = CascadingBandit(num_items, num_positions, true_prior_a0, true_prior_b0)
    unique_id = '|'.join([agent_name, str(seed)])
    agent = agent_constructor(num_items, num_positions)

    experiment = BaseExperiment(agent, environment, seed, unique_id=unique_id)
    experiment.run_experiment(n_steps)
    results.append(experiment.results)


# Plotting results
plotter = ResultsPlotter(results)
plotter.alg_seed_plot_variable()
