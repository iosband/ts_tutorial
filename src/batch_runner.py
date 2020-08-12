"""Run an experiment for given config, job_id and save_path.

This is the main function to run an experiment at large scale.
For example, to run the first configuration from finite_arm/config_simple:
```
  ipython batch_runner.py -- \
    --config finite_arm.config_simple --job_id 0 --save_path /tmp/
```

The config file defines the selection of agents/environments/seeds that we want
to run. This script is a lightweight wrapper to select one of these jobs, run
that specific job and then save the output to .csv file.

The code in base.plot handles collating the .csv files once you have all the
job_id's you want to run. Note that the `params` file (that maps all parameters
to job_id) is only written with `job_id == 0` by default.

If you do not want to write the experiments to .csv then please look at
`local_runner.py` instead. This is a convenience script that keeps everything
in memory, rather than writing the results out to .csv.

Valid (config, n_jobs) for reproducing the TS tutorial paper:
 - (finite_arm.config_simple, 20000)
 - (finite_arm.config_simple_rand, 20000)
 - (finite_arm.config_simple_sanity, 30000)
 - (finite_arm.config_misspecified, 20000)
 - (finite_arm.config_drift, 20000)
 - (cascading.config_cascading, 40000)
 - (ensemble_nn.config_nn, 20000)
 - (graph.config_indep, 5000)
 - (graph.config_indep_binary, 3000)
 - (graph.config_correlated, 3000)
 - (graph.config_correlated_sanity, 3000)
 - (pricing.config_pricing, 2000)

"""
import argparse
import importlib
import os
import sys

from base import config_lib

sys.path.append(os.getcwd())


##############################################################################
# Main function

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run batch experiment')
  parser.add_argument('--config', help='config', type=str)
  parser.add_argument('--job_id', help='job_id', type=int)
  parser.add_argument('--save_path', help='save_path', type=str)
  settings = parser.parse_args()

  # Loading in the experiment config.
  experiment_config = importlib.import_module(settings.config)
  config = experiment_config.get_config()

  # Running the experiment.
  job_config = config_lib.get_job_config(config, settings.job_id)
  experiment = job_config['experiment']
  experiment.run_experiment()

  # Saving results to csv.
  file_name = ('exp=' + config.name
               + '|id=' + str(settings.job_id) + '.csv')
  file_path = os.path.join(settings.save_path, file_name)
  with open(file_path, 'w') as f:
    experiment.results.to_csv(f, index=False)

  # Save the parameters if it is the first job.
  if settings.job_id == 0:
    params_df = config_lib.get_params_df(config)
    file_name = 'exp=' + config.name + '|params.csv'
    file_path = os.path.join(settings.save_path, file_name)
    with open(file_path, 'w') as f:
      params_df.to_csv(f, index=False)
