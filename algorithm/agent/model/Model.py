import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Critic(nn.Module):
	def __init__(self, input_dim, action_dim, hidden_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.linear1 = nn.Linear(input_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear3 = nn.Linear(hidden_dim, action_dim)

	def forward(self, obs, mask):
		
		x1 = F.relu(self.linear1(obs))
		x1 = F.relu(self.linear2(x1))
		x1 = self.linear3(x1)
		x1[mask == 0] = -1e10 
        
		return x1


class Actor(nn.Module):
	def __init__(self, input_dim, action_dim, hidden_dim):
		super(Actor, self).__init__()

		# Q1 architecture
		self.linear1 = nn.Linear(input_dim, hidden_dim)
		self.linear2 = nn.Linear(hidden_dim, hidden_dim)
		self.linear3 = nn.Linear(hidden_dim, action_dim)

		#self.apply(weights_init_)

	def forward(self, obs, mask):
		x1 = F.relu(self.linear1(obs))
		x1 = F.relu(self.linear2(x1))
		x1 = self.linear3(x1)
		x1[mask == 0] = -1e10 
        
		return x1