from itertools import chain, combinations
from utils.Utils import Simulator
import numpy as np
import random
import subprocess
import os

class TopoEnv(object):
	def __init__(self, env_id=0):

		self.env_id = env_id

		self.nodes = ['OUT', 'IN', 'net6']
		self.edges = []
		for i in range(len(self.nodes)):
			for j in range(i+1, len(self.nodes)):
				edge = 'R_L_{} ({} {}) resistor r=150M'.format(len(self.edges), self.nodes[i], self.nodes[j])
				self.edges.append(edge)
		self.actions = list(chain.from_iterable(combinations(self.edges, r) for r in range(len(self.edges)+1)))


		self.action_dim = len(self.actions)
		self.state_dim = 10

		netlist_path = './simulation/netlist_base'
		f = open(netlist_path,'r')
		self.line_before = []
		self.line_after = []
		bool_before = True
		for line in f:
			if 'Vsupply' in line:
				bool_before = False
			if bool_before:
				self.line_before.append(line)
			else:
				self.line_after.append(line)

	def get_action_space(self):
		return self.action_dim

	def get_obs_space(self):
		return self.state_dim

	def get_action_mask(self):
		return np.ones(self.action_dim)

	def reset(self):
		return np.ones(self.state_dim)


	def step(self, action):
		topo_lines = []
		topo_lines += self.line_before
		add_edges = self.actions[action]
		for edge in add_edges:
			topo_lines.append(edge + '\n')
		topo_lines += self.line_after

		return topo_lines


if __name__ == '__main__':
	Env = TopoEnv()
	Env.reset()
	rand_action = random.randint(0, Env.get_action_space()-1)
	topo = Env.step(rand_action)

	env_id = 7
	overwrite_path = '/mnt/mydata/RL_{}/run/simulation/surrogate_model/gj_opa/maestro/results/maestro/.tmpADEDir_root/surrogate_model:gj_opa:1/surrogate_model_gj_opa_schematic_spectre/netlist'.format(env_id)
	f = open(overwrite_path, 'w')
	for line in topo:
		f.write(line)
	f.close()

	Sim = Simulator()
	output = Sim.reset([12,12,0], env_id)
	print(output)