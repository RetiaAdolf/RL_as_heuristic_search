import sys
import os
from simulation.Env import CircuitEnv
from algorithm.agent import DQNagent, SACagent, MIACagent
from runner import Runner
import random
import time
import numpy as np
import torch
import pickle

def search(target_output):
	config = {}
	config['seed'] = 0
	config['max_timesteps'] = 1e7
	config['eval_interval'] = 20
	config['env_config'] = {}
	config['agent_config'] = {}
	config['env_config']['task_name'] = 'cleantable'
	config['env_config']['max_episode_length'] = 30
	config['agent_config']['gamma'] = 0.99
	config['agent_config']['update_interval'] = 3


	np.random.seed(config['seed'])
	random.seed(config['seed'])
	torch.manual_seed(config['seed'])

	env = CircuitEnv(target_output=target_output)
	env.reset()
	config['env_config']['action_space'] = env.get_action_space()
	config['env_config']['obs_space'] = env.get_obs_space()

	agent = SACagent(config=config)
	runner = Runner(agent=agent, env=env, config=config)
	eps = 0
	while runner.total_timesteps < config['max_timesteps']:
		episode, info = runner.sample(bool_eval=False)
		agent.memorize(episode)
		#print("cur_timesteps {}, value loss {}, return {}".format(runner.total_timesteps, info['loss'], info['return']))
		#logger.write("cur_timesteps {}, value loss {}, return {}".format(runner.total_timesteps, info['loss'], info['return']))
		#logger.write("\n")
		#logger.flush()
		eps += 1
		if eps % config['eval_interval'] == 0:
			_, eval_info = runner.sample(bool_eval=True)
			if "result" in eval_info.keys():
				#print(eval_info["result"])
				return runner.total_timesteps, env.get_nodes()
		if "result" in info.keys():
			#print(info["result"])
			return runner.total_timesteps, env.get_nodes()

	return config['max_timesteps'], env.get_nodes()

if __name__ == '__main__':
	with open('./simulation/SimOutput.pickle', 'rb') as f:
		dic = pickle.load(f)
	logger = open("search_result.txt", 'a')
	num = 0
	total_nodes = 0
	min_nodes = 9999999
	max_nodes = -9999999
	for i,key in enumerate(dic.keys()):
		output = dic[key]
		if i % 500 == 0:
			t, node = search(output)
			print("searched output: {}, used timesteps: {}, result: {}".format(output, t, node))
			logger.write("searched output: {}, used timesteps: {}, result: {}".format(output, t, node))
			logger.write("\n")
			logger.flush()

			num += 1
			total_nodes += node
			min_nodes = min(node, min_nodes)
			max_nodes = max(node, max_nodes)
			print("avg_nodes: {}, min_nodes: {}, max_nodes: {}".format(total_nodes * 1.0 / num, min_nodes, max_nodes))

	logger.write("avg_nodes: {}, min_nodes: {}, max_nodes: {}".format(total_nodes * 1.0 / num, min_nodes, max_nodes))
	logger.write("\n")
	logger.flush()
