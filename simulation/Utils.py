import numpy as np
import random
import pickle
import os
def __normalization__(data, data_range):
	data_min = data_range[:, 0]
	data_max = data_range[:, 1]
	normalized_data = (data - data_min) / (data_max - data_min)
	return normalized_data

def __random_target__(target_range, output_type):
	rand_output = []
	for output_range in target_range:
		if output_type == "float":
			rand_output.append(round(random.uniform(output_range[0],output_range[1]), 2))
		elif output_type == "int":
			rand_output.append(random.randint(output_range[0],output_range[1]))
	return np.array(rand_output)


class Simulator(object):
	"""docstring for Simulator"""
	def __init__(self):
		super(Simulator, self).__init__()
		__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
		with open(__location__ + '/SimOutput.pickle', 'rb') as f:
			self.data = pickle.load(f)
		self.nodes = set()

	def step(self, action, env_id):
		M3_W, M7_W, IN_OFST = action
		key = (M3_W, M7_W, IN_OFST)
		self.nodes.add(key)
		value = self.data[key]
		return value

	def get_nodes(self):
		return len(self.nodes)