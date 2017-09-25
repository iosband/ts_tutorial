"""Binomial bridge bandit experiment.

Binomial bridge with only binary reward at the end --> no conjugate update.
See Figure 8 https://arxiv.org/pdf/1707.02038.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from base.experiment import BaseExperiment
from base.plot import ResultsPlotter

from graph.agent_indep_binary import BootstrapIndependentBBWithBinaryReward
from graph.agent_indep_binary import LaplaceIndependentBBWithBinaryReward
from graph.env_graph_bandit import IndependentBinomialBridgeWithBinaryReward


n_stages = 20
shape = 2
scale = 0.5
tol=0.1
alpha = 0.2
beta = 0.5

agent_constructors = {'bs': BootstrapIndependentBBWithBinaryReward,
                      'lp': LaplaceIndependentBBWithBinaryReward}


# Running experiments (can be evaluated in parallel)
n_steps = 500
n_seeds = 10

results = []
for agent_name, agent_constructor in agent_constructors.iteritems():
  for seed in range(n_seeds):
    environment = IndependentBinomialBridgeWithBinaryReward(n_stages, shape, scale)
    unique_id = '|'.join([agent_name, str(seed)])
    agent = agent_constructor(n_stages, shape, scale, tol, alpha, beta)

    experiment = BaseExperiment(agent, environment, seed, unique_id=unique_id)
    experiment.run_experiment(n_steps)
    results.append(experiment.results)


# Plotting results
plotter = ResultsPlotter(results)
plotter.alg_seed_plot_variable()

