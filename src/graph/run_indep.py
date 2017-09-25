"""Binomial bridge bandit experiment with independent segments.

Compare the performance of Thompson sampling with 10% egreedy.
See Figure 6 https://arxiv.org/pdf/1707.02038.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from base.experiment import BaseExperiment
from base.plot import ResultsPlotter

from graph.agent_indep import IndependentBBEpsilonGreedy
from graph.agent_indep import IndependentBBTS
from graph.env_graph_bandit import IndependentBinomialBridge


n_stages = 20
mu0 = -0.5
sigma0 = 1
sigma_tilde = 1
epsilon = 0.01

agent_constructors = {'egreedy': IndependentBBEpsilonGreedy,
                      'ts': IndependentBBTS}

# Running experiments (can be evaluated in parallel)
n_steps = 500
n_seeds = 10

results = []
for agent_name, agent_constructor in agent_constructors.iteritems():
  for seed in range(n_seeds):
    environment = IndependentBinomialBridge(n_stages, mu0, sigma0, sigma_tilde)
    unique_id = '|'.join([agent_name, str(seed)])
    agent = agent_constructor(n_stages, mu0, sigma0, sigma_tilde, epsilon)

    experiment = BaseExperiment(agent, environment, seed, unique_id=unique_id)
    experiment.run_experiment(n_steps)
    results.append(experiment.results)


# Plotting results
plotter = ResultsPlotter(results)
plotter.alg_seed_plot_variable()

