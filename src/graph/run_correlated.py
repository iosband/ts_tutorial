"""Binomial bridge bandit experiment with correlated edges.

See Figure 7 https://arxiv.org/pdf/1707.02038.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from base.experiment import BaseExperiment
from base.plot import ResultsPlotter

from graph.agent_correlated import BootstrapCorrelatedBB
from graph.agent_correlated import CorrelatedBBTS
from graph.agent_correlated import IndependentBBTS
from graph.env_graph_bandit import CorrelatedBinomialBridge


n_stages = 20
mu0 = -0.5
sigma0 = 1
sigma_tilde = 1

agent_constructors = {'thoughtful ts': CorrelatedBBTS,
                      'misspecified ts': IndependentBBTS,
                      'bootstrap': BootstrapCorrelatedBB}


# Running experiments (can be evaluated in parallel)
n_steps = 500
n_seeds = 10

results = []
for agent_name, agent_constructor in agent_constructors.iteritems():
  for seed in range(n_seeds):
    environment = CorrelatedBinomialBridge(n_stages, mu0, sigma0, sigma_tilde)
    unique_id = '|'.join([agent_name, str(seed)])
    agent = agent_constructor(n_stages, mu0, sigma0, sigma_tilde)

    experiment = BaseExperiment(agent, environment, seed, unique_id=unique_id)
    experiment.run_experiment(n_steps)
    results.append(experiment.results)


# Plotting results
plotter = ResultsPlotter(results)
plotter.alg_seed_plot_variable()

