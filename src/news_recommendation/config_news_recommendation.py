'''Config file for the news recommendation problem.'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools

from base.config_lib import Config
from base.experiment import ExperimentNoAction
from news_recommendation.env_recommendation import NewsRecommendation
from news_recommendation.agent_recommendation import GreedyNewsRecommendation
from news_recommendation.agent_recommendation import EpsilonGreedyNewsRecommendation
from news_recommendation.agent_recommendation import LaplaceTSNewsRecommendation
from news_recommendation.agent_recommendation import LangevinTSNewsRecommendation

def get_config():
  """Generates the config for the experiment."""
  name = 'news_recommendation'
  num_articles = 3
  dim = 7
  theta_mean = 0
  theta_std = 1
  epsilon1 = 0.01
  epsilon2 = 0.05
  batch_size = 50
  step_count=200
  step_size= 1/200
   
  alpha=0.2
  beta=0.5
  tol=0.0001
     
  

  agents = collections.OrderedDict(
      [('greedy',
        functools.partial(GreedyNewsRecommendation,
                          num_articles, dim, theta_mean, theta_std, epsilon1,
                          alpha,beta,tol)),
       ('Langevin TS',
        functools.partial(LangevinTSNewsRecommendation,
                          num_articles, dim, theta_mean, theta_std, epsilon1,
                          alpha,beta,tol,batch_size,step_count,step_size)),
       (str(epsilon1)+'-greedy',
        functools.partial(EpsilonGreedyNewsRecommendation,
                          num_articles, dim, theta_mean, theta_std, epsilon1,
                          alpha,beta,tol)),
       (str(epsilon2)+'-greedy',
        functools.partial(EpsilonGreedyNewsRecommendation,
                          num_articles, dim, theta_mean, theta_std, epsilon2,
                          alpha,beta,tol)),
       ('Laplace TS',
        functools.partial(LaplaceTSNewsRecommendation,
                          num_articles, dim, theta_mean, theta_std, epsilon1,
                          alpha,beta,tol))
       ]
  )

  environments = collections.OrderedDict(
      [('env',
        functools.partial(NewsRecommendation,
          num_articles, dim, theta_mean, theta_std))]
  )
      
  experiments = collections.OrderedDict(
      [(name, ExperimentNoAction)]
  )
  n_steps = 5000
  n_seeds = 10000
  config = Config(name, agents, environments, experiments, n_steps, n_seeds)
  return config

