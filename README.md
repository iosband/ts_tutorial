A Tutorial on Thompson Sampling - Accompanying Code
===================================================

**Authors:** Dan Russo, Benjamin Van Roy, Abbas Kazerouni, Ian Osband, Zheng Wen

This is a git repository to release and share the code from our paper [A Tutorial on Thompson Sampling](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf).

All of the figures and results contained in the paper are generated through running the code contained in this repository.

Note that as of 29th June 2018 the version of this paper posted to arXiv is out of date, we are working to address this issue.
Please contact the authors if you have any trouble/questions.

## Installation

First clone the git repository.
```
git clone https://github.com/iosband/ts_tutorial.git
```

Our code is designed for iPython 2.7 (although it will likely work with other versions too).
We recommend [Anaconda iPython](https://ipython.org/install.html) as a starting point.
For the most part our code relies only on standard scientific python tools (numpy/scipy/pandas).
However, we do have two other dependencies that are slightly less standard:

- [plotnine](https://github.com/has2k1/plotnine) a grammar of graphic plotting library for Python.
- [cvxpy](http://www.cvxpy.org/en/latest/install/index.html) a package for convex optimization, only used for the dynamic pricing example.

All code is written assuming a path from `ts_tutorial/src`.

For an simple experiment of Thompson sampling versus greedy decision making run:
```
cd ts_tutorial/src
ipython simple_example.py
```


## Reproducing paper figures

We present the exact code used to generate the figures for each plot in `reproduce_figures.py`.

This is a command line script that can be called from the `src` directory:
```
cd ts_tutorial/src

# For instructions on how to use the script.
ipython reproduce_figures.py --help

# Reproduces Figure 3 with 1% of the seeds of the paper, save output to /tmp/
ipython reproduce_figures.py --figure 3 --run_frac 0.01 --data_path /tmp/ --plot_path /tmp/
```

Reproducing the number of experiments and seeds used in the paper can be extremely computationally expensive.
For this reason we suggest that either use a very small `run_frac` or delve slightly deeper into the code to set up some parallel/distributed workflow.
Since the process of developing distributed computation can be complicated and idiosyncratic we leave that to individual users, but present an outline of the high level code below.

## Code structure

Our code is meant to be as simple as possible, optimizing for ease of use rather than speed.

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

runs the first experiment from `finite_arm/config_simple`, job_id=0 corresponds to (agent='greedy', seed=0) on a 3-armed bandit problem.
It then saves the results to `/tmp/exp=finite_simple|id=0.csv`, you can recover the parameters for job_id=0 (and any other integer) by looking in `/tmp/exp=finite_simple|params.csv`, which is also generated automatically.

This script (`batch_runner.py`) is designed to be used in a distributed computation framework.
In this way, each job can be run on separate machines and the resulting output can be saved to a shared file system.
The code in `base/plot.py` is designed to collate the resulting `.csv` files from many different job_id's and produce the plots from the paper.
An example of this use case can be found in `batch_analysis.py`.



## Playing around with code locally

If you are just playing around with the code locally (and happy to run fewer seeds) you may find `local_runner.py` to be better for your needs than `batch_runner.py`, since it keeps all data in memory rather than writing to .csv file.

Researchers or students who want to get a better feel for the underlying code might use this as a starting point for exploring new methods and approaches.


