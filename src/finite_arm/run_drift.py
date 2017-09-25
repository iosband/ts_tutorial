"""Finit multi-armed bandit with drift.

Comparing the performance of Thompson sampling with forgetting factor to
Thompson sampling without forgetting factor in a nonstationary environment.
See Figure 12 https://arxiv.org/pdf/1707.02038.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from base.experiment import BaseExperiment
from base.plot import ResultsPlotter

from finite_arm.agent_finite import FiniteBernoulliBanditTS
from finite_arm.agent_finite import DriftingFiniteBernoulliBanditTS
from finite_arm.env_finite import DriftingFiniteArmedBernoulliBandit


agent_constructors = {'stationary_ts': FiniteBernoulliBanditTS,
                      'nonstationary_ts': DriftingFiniteBernoulliBanditTS}

# Running experiments (can be evaluated in parallel)
n_arm = 3
n_steps = 2000
n_seeds = 100

results = []
for agent_name, agent_constructor in agent_constructors.iteritems():
  for seed in range(n_seeds):
    environment = DriftingFiniteArmedBernoulliBandit(n_arm)
    unique_id = '|'.join([agent_name, str(seed)])
    agent = agent_constructor(n_arm)

    experiment = BaseExperiment(agent, environment, seed, unique_id=unique_id)
    experiment.run_experiment(n_steps)
    results.append(experiment.results)


# Plotting results
plotter = ResultsPlotter(results)
plotter.alg_seed_plot_variable()

