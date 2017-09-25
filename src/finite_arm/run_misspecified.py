"""Finite multi-armed bandit with mis-specified prior.

Comparing the performance of Thompson sampling with an informed prior, with
Thompson sampling with an uninformed (incorrect) prior.
See Figure 11 https://arxiv.org/pdf/1707.02038.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from base.experiment import ExperimentWithMean

from finite_arm.agent_finite import FiniteBernoulliBanditTS
from finite_arm.agent_finite import DriftingFiniteBernoulliBanditTS
from finite_arm.env_finite import DriftingFiniteArmedBernoulliBandit

n_arm = 3
true_prior_success = [1, 1, 1]
true_prior_failure = [50, 100, 200]

def correct_ts_init(n_arm):
  assert n_arm == 3  # adhoc method for this experiment
  agent = FiniteBernoulliBanditTS(n_arm)
  agent.set_prior(true_prior_success, true_prior_failure)
  return agent

agent_constructors = {'correct_ts': correct_ts_init,
                      'misspecified_ts': FiniteBernoulliBanditTS}


# Running experiments (can be evaluated in parallel)
n_steps = 1000
n_seeds = 100

results = []
for agent_name, agent_constructor in agent_constructors.iteritems():
  for seed in range(n_seeds):
    environment = DriftingFiniteArmedBernoulliBandit(n_arm, gamma=0.0)
    environment.set_prior(true_prior_success, true_prior_failure)
    unique_id = '|'.join([agent_name, str(seed)])
    agent = agent_constructor(n_arm)

    experiment = ExperimentWithMean(agent, environment, seed, unique_id=unique_id)
    experiment.run_experiment(n_steps)
    results.append(experiment.results)



###############################################################################
# Adhoc plotting

df = pd.concat(results)
df['alg'] = df.unique_id.apply(lambda x: x.split('|')[0])
df['seed'] = df.unique_id.apply(lambda x: x.split('|')[1])


# Regret by algorithm
regret = (df.groupby(['t', 'alg'])
            .agg({'instant_regret': np.mean})
            .reset_index()
            .pivot(index='t', columns='alg')
            .reset_index(drop=True))
plt.plot(regret['instant_regret']['correct_ts'], 'b')
plt.plot(regret['instant_regret']['misspecified_ts'], 'r')
plt.ylim([0.0, 0.02])
plt.show()


# Action means
new_col_list = ['mean_0','mean_1','mean_2']
for n, col in enumerate(new_col_list):
    df[col] = df['posterior_mean'].apply(lambda x: x[n])

# Proportion of action selection
mean_0 = (df.groupby(['t', 'alg'])
            .agg({'mean_0': np.mean})
            .reset_index()
            .pivot(index='t', columns='alg')
            .reset_index(drop=True))
mean_1 = (df.groupby(['t', 'alg'])
            .agg({'mean_1': np.mean})
            .reset_index()
            .pivot(index='t', columns='alg')
            .reset_index(drop=True))
mean_2 = (df.groupby(['t', 'alg'])
            .agg({'mean_2': np.mean})
            .reset_index()
            .pivot(index='t', columns='alg')
            .reset_index(drop=True))

plt.plot(mean_0['mean_0']['correct_ts'], 'b')
plt.plot(mean_1['mean_1']['correct_ts'], 'b')
plt.plot(mean_2['mean_2']['correct_ts'], 'b')

plt.plot(mean_0['mean_0']['misspecified_ts'], 'r')
plt.plot(mean_1['mean_1']['misspecified_ts'], 'r')
plt.plot(mean_2['mean_2']['misspecified_ts'], 'r')

plt.ylim([0.0, 0.05])
plt.show()

