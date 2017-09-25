"""A simple experiment comparing Thompson sampling to greedy algorithm.

Finite armed bandit with 3 arms.
Greedy algorithm premature and suboptimal exploitation.
See Figure 3 from https://arxiv.org/abs/1707.02038
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from base.experiment import BaseExperiment
from base.plot import ResultsPlotter

from finite_arm.agent_finite import FiniteBernoulliBanditEpsilonGreedy
from finite_arm.agent_finite import FiniteBernoulliBanditTS
from finite_arm.env_finite import FiniteArmedBernoulliBandit


n_arm = 3
probs = [0.9, 0.8, 0.7]
environment = FiniteArmedBernoulliBandit(probs)

agent_constructors = {'greedy': FiniteBernoulliBanditEpsilonGreedy,
                      'ts': FiniteBernoulliBanditTS}


# Running experiments (can be evaluated in parallel)
n_steps = 1000
n_seeds = 100

results = []
for agent_name, agent_constructor in agent_constructors.iteritems():
  for seed in range(n_seeds):
    unique_id = '|'.join([agent_name, str(seed)])
    agent = agent_constructor(n_arm)
    experiment = BaseExperiment(agent, environment, seed, unique_id=unique_id)
    experiment.run_experiment(n_steps)
    results.append(experiment.results)



###############################################################################
# Adhoc plotting with matplotlib

plotter = ResultsPlotter(results)
plotter.alg_seed_plot_variable()

plotter.proportion_action_plot('greedy')
plotter.proportion_action_plot('ts')
