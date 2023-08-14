import os
import torch
from torch.distributions import Categorical
from torch.optim import Adam
from .model import Critic
from collections import deque
import copy
import random
import numpy as np

def soft_update(target, source, tau):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(param.data)

class DQNagent(object):
	def __init__(self, config):

		self.config = config

		self.lr = 3e-4
		self.gamma = config['agent_config']['gamma']
		self.eps = 1.0
		self.eps_min = 0.1
		self.target_update_interval = 200

		self.device = torch.device("cuda")
		self.batch_size = 64
		self.learning_start = 1000
		self.buffer_size = 50000
        
		self.input_dim = self.config['env_config']['obs_space']
		self.hidden_dim = 64
		self.action_dim = self.config['env_config']['action_space']

		self.critic = Critic(input_dim=self.input_dim,
                             action_dim=self.action_dim,
                             hidden_dim=self.hidden_dim).to(device=self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		self.buffer = deque(maxlen=self.buffer_size)
		self.last_update_target = 0

	def sample_action(self, obs, mask, t, bool_eval=False):
		

		obs = torch.tensor([obs], dtype=torch.float, device=self.device) #1v
		mask = torch.tensor([mask], dtype=torch.float, device=self.device) #1a

		qval = self.critic(obs, mask) #1a

		if bool_eval:
			action = qval.max(dim=-1)[1] 
		else:
			action = qval.max(dim=-1)[1] 
			rand_dist = Categorical(probs=mask / mask.sum(dim=-1, keepdims=True))
			rand_action = rand_dist.sample()
			if random.random() < self.eps:
				action = rand_action
		if t > self.learning_start:
			self.eps = max(self.eps * 0.9995, self.eps_min)
		return action.detach().cpu().item()

	def memorize(self, data):
		for d in data:
			transition = (d['obs'],
						  d['mask'],
						  d['action'],
						  d['reward'],
						  d['next_obs'],
						  d['next_mask'],
						  d['done'])
			self.buffer.append(transition)


	def learn(self, cur_timesteps):
		if len(self.buffer) < self.batch_size:
			return 0
		else:
			batch = random.sample(self.buffer, self.batch_size)
			(obs, mask, action, reward, next_obs, next_mask, done) = zip(*batch)

			batch = (np.array(obs), np.array(mask),np.array(action),np.array(reward),
					 np.array(next_obs),np.array(next_mask),np.array(done))
			log_data =  self.update_parameters(batch)
			if (cur_timesteps - self.last_update_target) >= self.target_update_interval:
				self.last_update_target = cur_timesteps
				hard_update(self.critic_target, self.critic)             
			return log_data

	def update_parameters(self, batch):
		# Sample a batch from memory
		obs, mask, action, reward, next_obs, next_mask, done = batch

		obs = torch.tensor(obs, dtype=torch.float, device=self.device) #bv
		mask = torch.tensor(mask, dtype=torch.float, device=self.device) #ba
		action = torch.tensor(action, dtype=torch.long, device=self.device) #b
		reward = torch.tensor(reward, dtype=torch.float, device=self.device) #b
		next_obs = torch.tensor(next_obs, dtype=torch.float, device=self.device) #bv
		next_mask = torch.tensor(next_mask, dtype=torch.float, device=self.device) #ba
		done = torch.tensor(done, dtype=torch.float, device=self.device) #b

		with torch.no_grad():
			next_action = self.critic(next_obs, next_mask).max(dim=-1)[1]
			next_qvals = self.critic_target(next_obs, next_mask)
			next_qvals = torch.gather(next_qvals, dim=-1, index=next_action.unsqueeze(-1)).squeeze(-1)
			next_qvals = reward + (1 - done) * self.gamma * (next_qvals)
		next_qvals = next_qvals.detach()

		qvals = self.critic(obs, mask)
		q_selected = torch.gather(qvals, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
		loss = ((q_selected - next_qvals) ** 2)
		loss = loss.mean()
		self.critic_optim.zero_grad()
		loss.backward()
		self.critic_optim.step()

		return loss.item()