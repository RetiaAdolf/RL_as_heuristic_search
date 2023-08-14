import numpy as np
import copy
class Runner():
	"""docstring for Runner"""
	def __init__(self, agent, env, config):
		super(Runner, self).__init__()
		self.agent = agent
		self.env = env
		self.config = config
		self.max_timesteps = self.config['env_config']['max_episode_length']
		self.total_timesteps = 0
		self.update_interval = self.config['agent_config']['update_interval']

	def sample(self, bool_eval=False):
		eps_data = []
		eps_loss = [0]
		eps_return = 0
		obs, mask = self.env.reset()
		for t in range(self.max_timesteps):
			action = self.agent.sample_action(obs=obs, mask=mask, t=self.total_timesteps, bool_eval=bool_eval)
			next_obs, next_mask, reward, done, info = self.env.step(action)
			
			data = {"obs":obs, 
					"mask":mask,
					"action":action, 
					"reward":reward, 
					"next_obs":next_obs, 
					"next_mask":next_mask,
					"done":done,}
			eps_data.append(copy.deepcopy(data))

			obs = next_obs
			mask = next_mask
			eps_return += reward

			if bool_eval:
				print("t: {}, action: {}".format(t, info['script']))
			else:
				self.total_timesteps += 1
				if t % self.update_interval == 0:
					loss = self.agent.learn(self.total_timesteps)
					eps_loss.append(loss)
			if done:
				break
		return eps_data, {'return': eps_return, 'loss': np.mean(eps_loss)}
