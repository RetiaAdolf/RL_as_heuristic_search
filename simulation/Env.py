import random
import numpy as np
import copy
import time
import itertools
from .Utils import __normalization__
from .Utils import Simulator
import pickle


__input_num__ = 3
__action_num__ = 3
__ouput_num__ = 4

class CircuitEnv(object):
	"""docstring for Env"""
	def __init__(self, env_id=0):
		super(CircuitEnv, self).__init__()


		self.Sim = Simulator()
		self.env_id = env_id
		self.action_dim = __input_num__ ** __action_num__
		self.state_dim = __ouput_num__* 2 + __input_num__
		self.joint_action_mapping = list(itertools.product([-1,0,1],repeat=3))
		
		self.input_range = np.array([[12, 60], [12, 60], [0, 50]])
		self.output_range = np.array([[28.31, 78.12], [-11.63, -0.33], [4.41, 178.46], [1.64, 16.92]])

		self.target_output = None
		self.cur_input = None
		self.cur_output = None
		self.done = None


	def get_action_space(self):
		return self.action_dim


	def get_obs_space(self):
		return self.state_dim

	def get_action_mask(self):
		return np.ones(self.action_dim)

	def reset(self):

		target_output = [50.59, -5.6052, 65.30, 7.625]

		self.target_output = np.array(target_output).round(2)
		self.cur_input = np.array([36, 36, 25])
		self.cur_output = self.Sim.step(self.cur_input, self.env_id)
		self.done = 0

		state = self.__get_state__(self.cur_input, self.cur_output, self.target_output)
		mask = self.get_action_mask()

		return state, mask


	def __get_state__(self, input, output, target):
		normalized_input = __normalization__(input, self.input_range)
		normalized_output = __normalization__(output, self.output_range)
		normalized_target = __normalization__(target, self.output_range)

		return np.concatenate([normalized_input, normalized_output, normalized_target])

	def __get_reward__(self, output, target):
		normalized_output = __normalization__(output, self.output_range)
		normalized_target = __normalization__(target, self.output_range)
		relative_output = normalized_output - normalized_target

		reward = 0
		count = 0

		for element in relative_output:
			if element > 0:
				reward -= element
			else:
				count += 1
				reward -= element

		if count == 4:
			reward += 1

		return reward, count

	def __modify_input__(self, cur_input, action_id):
		actions = np.array(self.joint_action_mapping[action_id])
		modified_action = cur_input + actions
		clipped_action = np.clip(modified_action, self.input_range[:, 0], self.input_range[:, 1])
		return clipped_action

	def step(self, action):

		start_time = time.time()
		self.target_output = self.target_output
		self.cur_input = self.__modify_input__(self.cur_input, action)
		self.cur_output = self.Sim.step(self.cur_input, self.env_id)
		self.done = self.done

		state = self.__get_state__(self.cur_input, self.cur_output, self.target_output)
		mask = self.get_action_mask()
		reward, count = self.__get_reward__(self.cur_output, self.target_output)
		info = {"script": "cur input: {}, cur ouput: {}, target output: {}, reward: {}".format(self.cur_input, self.cur_output, self.target_output, reward)}
		
		if count == 4 and self.done == 0:
			self.done = 1

		return state, mask, reward, self.done



if __name__ == '__main__':
	input_range = np.array([[12, 60], [12, 60], [0, 50]])
	modified_action = np.array([11, 47, 36])
	print(np.clip(modified_action, input_range[:, 0], input_range[:, 1]))