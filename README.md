A Tutorial on Thompson Sampling - Accompanying Code
===================================================

**Authors:** Dan Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, Zheng Wen

This is a git repository to release and share the code from our paper [A Tutorial on Thompson Sampling](https://arxiv.org/abs/1707.02038).
Its purpose is to allow for reproducibility of results and also to elucidate some practical programming elements of the underlying algorithms.


## Installation

First clone the git repository.
```
git clone https://github.com/iosband/ts_tutorial.git
```

Next, install the dependencies:
- [plotnine](https://github.com/has2k1/plotnine) a grammar of graphic plotting library for Python.
- [cvxpy](http://www.cvxpy.org/en/latest/install/index.html) a package for convex optimization, only used for the dynamic pricing example.

If you are using the (recommended) [Anaconda](https://anaconda.org/anaconda/python) python distribution then you can accomplish this simply by running:
```
conda install -c conda-forge plotnine
conda install -c cvxgrp cvxpy libgcc
```
We are working on packaging this tutorial so that in future versions this might all happen automatically from a single command line call.

For now, all code is written assuming a path from `ts_tutorial/src`.
To run a a simple experiment of Thompson sampling versus greedy decision making:
```
cd ts_tutorial/src
ipython simple_example.py
```

## Code structure

The code format/style is very simple and relies on nothing more than raw python / numpy.

We break the experiments/algorithms into two key parts:

- `agent`: handles learning + action selection of a bandit algorithm.
- `environment`: evaluates rewards/outcomes from an action.

We then say that an `experiment` collects the data from an `agent` with an `environment` together with an optional random seed.

We create an abstract skeleton for these classes in `src/base` and implement separate children classes in each of the `src/finite_arm`, `src/cascading`, `src/ensemble_nn`, `src/pricing` and `src/graph` subfolders.

Since many of our experiments involve large numbers of seeds/parameters we break down the jobs into pieces that can be parallelized easily.
To do this we specify the complete selection of agents/environments/seeds that we will run in a `config` file.
You can find the config file in each of these subfolders that describe exactly what is run for each Figure in the paper.

To run a specific job agent/environemt/seed from a config file you simply specify the `job_id`, which is the unique integer location for that parameter combination.
For example, the command:

```
ipython batch_runner.py --config finite_arm.config_simple --job_id 0 --save_path /tmp/
```

runs the first experiment from finite_arm/config_simple (agent='greedy', seed=0) on a 3-armed bandit problem.
It then saves the results to `/tmp/exp=finite_simple|id=0.csv`, you can recover the parameters for job_id=0 (and any other integer) by looking in `/tmp/exp=finite_simple|params.csv`, which is also generated automatically.


## Plotting results

Once you have run all of the experiments, you can regenerate the plots from the paper by running `batch_analysis.py`.


## Playing around with code locally

If you are just playing around with the code locally (and happy to run fewer seeds) you may find `local_runner.py` to be better for your needs than `batch_runner.py`, since it keeps all data in memory rather than writing to .csv file.


