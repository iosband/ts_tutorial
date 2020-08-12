"""Run an simple experiment locally without using config file.

This file is presented as a very simple entry point to code.
For running any meaningful experiments, we suggest `batch_runner.py` or
`local_runner.py`.
"""
import os
import sys

import numpy as np
import pandas as pd
import plotnine as gg

from base.experiment import BaseExperiment
from finite_arm.agent_finite import FiniteBernoulliBanditTS
from finite_arm.env_finite import FiniteArmedBernoulliBandit

sys.path.append(os.getcwd())

##############################################################################
# Running a single experiment

probs = [0.7, 0.8, 0.9]
n_steps = 1000
seed = 0

agent = FiniteBernoulliBanditTS(n_arm=len(probs))
env = FiniteArmedBernoulliBandit(probs)
experiment = BaseExperiment(
    agent, env, n_steps=n_steps, seed=seed, unique_id='example')

experiment.run_experiment()


experiment.run_experiment()

##############################################################################
# Simple display / plot of results

experiment.results.head()

p = (gg.ggplot(experiment.results)
     + gg.aes(x='t', y='instant_regret', colour='unique_id')
     + gg.geom_line())
print(p)
