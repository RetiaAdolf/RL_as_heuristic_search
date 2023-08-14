import os
import torch
from torch.distributions import Categorical
from torch.optim import Adam
from .model import Critic, Actor
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

class MIACagent(object):
	def __init__(self, config):

		self.config = config

		self.alpha = 1
		self.alpha_lr = 3e-4
		self.lr = 3e-4
		self.gamma = config['agent_config']['gamma']
		self.eps = 1.0
		self.eps_min = 0.1
		self.target_update_interval = 200

		self.device = torch.device("cuda")
		self.batch_size = 256
		self.freeze_timesteps = 1000
		self.buffer_size = 50000
		self.on_policy_buffer_size = self.batch_size
		
		self.input_dim = self.config['env_config']['obs_space']
		self.hidden_dim = 64
		self.action_dim = self.config['env_config']['action_space']

		self.critic = Critic(input_dim=self.input_dim,
							 action_dim=self.action_dim,
							 hidden_dim=self.hidden_dim).to(device=self.device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

		self.policy = Actor(input_dim=self.input_dim,
							 action_dim=self.action_dim,
							 hidden_dim=self.hidden_dim).to(device=self.device)
		self.policy_optim = Adam(self.policy.parameters(), lr=self.lr)

		self.policy_avg = Actor(input_dim=self.input_dim,
							 action_dim=self.action_dim,
							 hidden_dim=self.hidden_dim).to(device=self.device)

		self.policy_avg_optim = Adam(self.policy_avg.parameters(), lr=self.lr)

		self.buffer = deque(maxlen=self.buffer_size)
		self.on_policy_buffer = deque(maxlen=self.on_policy_buffer_size)
		self.last_update_target = 0

	def sample_action(self, obs, mask, t, bool_eval=False):
		

		obs = torch.tensor([obs], dtype=torch.float, device=self.device) #1v
		mask = torch.tensor([mask], dtype=torch.float, device=self.device) #1a

		logit = self.policy(obs, mask) #1a

		if bool_eval:
			dist = Categorical(logits=logit)
			action = dist.sample()		
		else:
			dist = Categorical(logits=logit)
			action = dist.sample()
			rand_dist = Categorical(probs=mask / mask.sum(dim=-1, keepdims=True))
			rand_action = rand_dist.sample()
			if random.random() < self.eps:
				action = rand_action
		if t > self.freeze_timesteps:
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
						  d['done'],)
			self.buffer.append(transition)
			self.on_policy_buffer.append(transition)


	def learn(self, cur_timesteps):
		if len(self.buffer) < self.batch_size or len(self.on_policy_buffer) < self.batch_size:
			return 0
		else:
			batch = random.sample(self.buffer, self.batch_size)
			(obs, mask, action, reward, next_obs, next_mask, done) = zip(*batch)

			batch = (np.array(obs), np.array(mask),np.array(action),np.array(reward),
					 np.array(next_obs),np.array(next_mask),np.array(done))
			log_data =  self.update_parameters(batch)

			on_policy_batch = random.sample(self.on_policy_buffer, self.batch_size)
			(obs, mask, action, reward, next_obs, next_mask, done) = zip(*on_policy_batch)
			on_policy_batch = (np.array(obs), np.array(mask),np.array(action),np.array(reward),
					 np.array(next_obs),np.array(next_mask),np.array(done))
			_ = self.update_parameters_on_policy(on_policy_batch)
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
			next_logit = self.policy(next_obs, next_mask)
			next_logit_avg = self.policy_avg(torch.zeros_like(next_obs), next_mask)

			next_dist = Categorical(logits=next_logit)
			next_dist_avg = Categorical(logits=next_logit_avg)

			next_action = next_dist.sample()

			next_pi = torch.gather(next_dist.probs, dim=-1, index=next_action.unsqueeze(-1)).squeeze(-1)
			next_log_pi = (next_pi + 1e-10).log()

			next_pi_avg = torch.gather(next_dist_avg.probs, dim=-1, index=next_action.unsqueeze(-1)).squeeze(-1)
			next_log_pi_avg = (next_pi_avg + 1e-10).log()

			next_qvals = self.critic_target(next_obs, next_mask)
			next_qvals = torch.gather(next_qvals, dim=-1, index=next_action.unsqueeze(-1)).squeeze(-1)
			next_qvals = reward + (1 - done) * self.gamma * (next_qvals - self.alpha * (next_log_pi - next_log_pi_avg))
		next_qvals = next_qvals.detach()

		qvals = self.critic(obs, mask)
		q_selected = torch.gather(qvals, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
		loss = ((q_selected - next_qvals) ** 2)
		loss = loss.mean()
		self.critic_optim.zero_grad()
		loss.backward()
		self.critic_optim.step()

		logit = self.policy(obs, mask)
		dist = Categorical(logits=logit)
		pi = dist.probs
		log_pi = (pi + 1e-10).log()

		with torch.no_grad():
			logit_avg = self.policy_avg(torch.zeros_like(obs), mask)
			dist_avg = Categorical(logits=logit_avg)
			pi_avg = dist_avg.probs
			log_pi_avg = (pi_avg + 1e-10).log()
		log_pi_avg = log_pi_avg.detach()

		qvals = self.critic(obs, mask)
		qvals = qvals.detach()
		policy_loss = (pi * (self.alpha * (log_pi - log_pi_avg) - qvals)).sum(dim=-1)
		policy_loss = policy_loss.mean()
		self.policy_optim.zero_grad()
		policy_loss.backward()
		self.policy_optim.step()
		
		self.alpha = min((1 - self.alpha_lr) * self.alpha + self.alpha_lr * (loss.item()), 1)

		return loss.item()


	def update_parameters_on_policy(self, batch):
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
			logit = self.policy(obs, mask)
			dist = Categorical(logits=logit)
			pi = dist.probs
		pi = pi.detach()
        
		logit_avg = self.policy_avg(torch.zeros_like(obs), mask)
		dist_avg = Categorical(logits=logit_avg)
		pi_avg = dist_avg.probs
		log_pi_avg = (pi_avg + 1e-10).log()

		policy_avg_loss = -(pi * log_pi_avg).sum(dim=-1)
		policy_avg_loss = policy_avg_loss.mean()
		self.policy_avg_optim.zero_grad()
		policy_avg_loss.backward()
		self.policy_avg_optim.step()

		return policy_avg_loss.item()