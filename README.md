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

All code is written assuming a path from `ts_tutorial/src`.
To run a a simple experiment of Thompson sampling versus greedy decision making:
```
cd ts_tutorial/src
ipython finite_arm/run_simple.py
```

This should run through a simple experiment and output plots similar to [Figure 3](https://arxiv.org/abs/1707.02038)

## Code structure

The code format/style is very simple and relies on nothing more than raw python / numpy + some pandas / matplotlib for plotting.

We break the experiments/algorithms into two key parts:

- `agent`: handles learning + action selection of a bandit algorithm.
- `environment`: evaluates rewards/outcomes from an action.

We then say that an `experiment` collects the data from an `agent` with an `environment` together with an optional random seed.

We create an abstract skeleton for these classes in `src/base` and implement separate children classes in each of the `src/finite_arm`, `src/cascading_bandit` and `src/graph` subfolders.

In each of these subfolders you will find a `run_*.py` file that runs a specific experiment.
Where possible, we try to include a small description of the experiment, what type of results you should expect to see and a link to relevant section of the tutorial paper.

Most of the format/style is taken from the simple RL library [TabulaRL](https://github.com/iosband/TabulaRL) which we adapt for the specific case of multi-armed bandit for clarity.
The code for the finite horizon MDPs is available from [TabulaRL](https://github.com/iosband/TabulaRL) directly.


## Reproducing large scale results

Many of the plots in our paper are averaged over thousands of random seeds.
Reproducing these full results on a single machine would probably take a long time...

However, most of the individual "experiments" do not take very long and can be parallelized without problem.
For information on how to get this set up with Amazon EC2 we recommend [Star Cluster](http://star.mit.edu/cluster/).
