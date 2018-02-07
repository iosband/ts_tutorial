A Tutorial on Thompson Sampling - Accompanying Code
===================================================

**Authors:** Dan Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, Zheng Wen

This is a git repository to release and share the code from our paper [A Tutorial on Thompson Sampling](https://arxiv.org/abs/1707.02038).

All of the figures and results contained in the paper are generated through running the code contained in this repository.



## Installation

First clone the git repository.
```
git clone https://github.com/iosband/ts_tutorial.git
```

Next, install the dependencies:
- [plotnine](https://github.com/has2k1/plotnine) a grammar of graphic plotting library for Python.
- [cvxpy](http://www.cvxpy.org/en/latest/install/index.html) a package for convex optimization, only used for the dynamic pricing example.

All code is written assuming a path from `ts_tutorial/src`.

For an simple experiment of Thompson sampling versus greedy decision making run:
```
cd ts_tutorial/src
ipython simple_example.py
```

## Code structure

The code format/style is very simple and relies on nothing more than raw python / numpy.

We break the experiments/algorithms into three key parts:

- `agent`: handles learning + action selection of a bandit algorithm.
- `environment`: evaluates rewards/outcomes from an action.
- `experiment`: contains agent, environment and seed.

We create an abstract skeleton for these classes in `src/base` and implement separate children classes in each of the `src/finite_arm`, `src/cascading`, `src/ensemble_nn`, `src/pricing` and `src/graph` subfolders.

To run large sweeps over many agents and seeds we define a `config` file.
The config decribes the selection of agents, environments and seeds that we want to run.
`batch_runner.py` will run a specific element of the `config` sweep that is specified by the `job_id` = the unique integer identifier for the parameter combination.
For example,
```
ipython batch_runner.py --config finite_arm.config_simple --job_id 0 --save_path /tmp/
```

runs the first experiment from finite_arm/config_simple (agent='greedy', seed=0) on a 3-armed bandit problem.
It then saves the results to `/tmp/exp=finite_simple|id=0.csv`, you can recover the parameters for job_id=0 (and any other integer) by looking in `/tmp/exp=finite_simple|params.csv`, which is also generated automatically.



## Reproducing paper figures

We present the exact code used to generate the figures for each plot in `reproduce_figures.py`.

Since many of our experiments involve large numbers of seeds/parameters we suggest that users may only want to run some `RUN_FRACTION < 1` proportion of the jobs used in the paper.



## Playing around with code locally

If you are just playing around with the code locally (and happy to run fewer seeds) you may find `local_runner.py` to be better for your needs than `batch_runner.py`, since it keeps all data in memory rather than writing to .csv file.

Researchers or students who want to get a better feel for the underlying code might use this as a starting point for exploring new methods and approaches.


