import sys
import os
from simulation.Env import CircuitEnv
from algorithm.agent import DQNagent, SACagent, MIACagent
from runner import Runner
import random
import time
import numpy as np
import torch

config = {}
config['seed'] = 0
config['max_timesteps'] = 1e7
config['eval_interval'] = 20
config['env_config'] = {}
config['agent_config'] = {}
config['env_config']['task_name'] = 'cleantable'
config['env_config']['max_episode_length'] = 30
config['agent_config']['gamma'] = 0.99
config['agent_config']['update_interval'] = 5


np.random.seed(config['seed'])
random.seed(config['seed'])
torch.manual_seed(config['seed'])
logger = open("result_{}.txt".format(config['seed']), 'a')

env = CircuitEnv()
env.reset()
config['env_config']['action_space'] = env.get_action_space()
config['env_config']['obs_space'] = env.get_obs_space()

agent = DQNagent(config=config)
runner = Runner(agent=agent, env=env, config=config)
eps = 0
while runner.total_timesteps < config['max_timesteps']:
	episode, info = runner.sample(bool_eval=False)
	agent.memorize(episode)
	print("cur_timesteps {}, value loss {}, return {}".format(runner.total_timesteps, info['loss'], info['return']))
	logger.write("cur_timesteps {}, value loss {}, return {}".format(runner.total_timesteps, info['loss'], info['return']))
	logger.write("\n")
	logger.flush()
	eps += 1
	if eps % config['eval_interval'] == 0:
		runner.sample(bool_eval=True)