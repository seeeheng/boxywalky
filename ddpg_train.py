import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time

from ddpg_agent import Agent
from ddpg_logger import DDPGLogger

env = gym.make('BipedalWalker-v3')
env.seed(10)

#-------------------------------------------------------------------------------------------------------------#

class NormalizeAction(gym.ActionWrapper):
	def action(self, action):
		action = (action + 1) / 2  
		action *= (self.action_space.high - self.action_space.low)
		action += self.action_space.low
		return action

	def reverse_action(self, action):
		action -= self.action_space.low
		action /= (self.action_space.high - self.action_space.low)
		a
		ction = action * 2 - 1
		return actions

def ddpg_distance_metric(actions1, actions2):
	"""
	Compute "distance" between actions taken by two policies at the same states
	Expects numpy arrays
	"""
	diff = actions1-actions2
	mean_diff = np.mean(np.square(diff), axis=0)
	dist = sqrt(np.mean(mean_diff))
	return dist

class AdaptiveParamNoiseSpec(object):
	def __init__(self, initial_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01):
		"""
		Note that initial_stddev and current_stddev refer to std of parameter noise, 
		but desired_action_stddev refers to (as name notes) desired std in action space
		"""
		self.initial_stddev = initial_stddev
		self.desired_action_stddev = desired_action_stddev
		self.adaptation_coefficient = adaptation_coefficient

		self.current_stddev = initial_stddev

	def adapt(self, distance):
		if distance > self.desired_action_stddev:
			# Decrease stddev.
			self.current_stddev /= self.adaptation_coefficient
		else:
			# Increase stddev.
			self.current_stddev *= self.adaptation_coefficient

	def get_stats(self):
		stats = {
			'param_noise_stddev': self.current_stddev,
		}
		return stats

	def __repr__(self):
		fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
		return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)

param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05,desired_action_stddev=0.3, adaptation_coefficient=1.05)
env = NormalizeAction(env)

#-------------------------------------------------------------------------------------------------------------#

agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=10, train=False)

def ddpg(n_episodes=100000, max_t=700, load=False):
	scores_deque = deque(maxlen=100)
	scores = []
	max_score = -np.Inf

	if load:
		agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
		agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

	for i_episode in range(1, n_episodes+1):
		logger.update_time_and_episodes(time.time(),i_episode)
		logger.check(i_episode)
		state = env.reset()
		score = 0
		agent.perturb_actor_parameters(param_noise)

		for t in range(max_t):
			agent.noise.reset()
			action = agent.act(state,param_noise)
			next_state, reward, done, _ = env.step(action)
			agent.step(state, action, reward, next_state, done)
			state = next_state
			score += reward
			if done:
				break 
		scores_deque.append(score)
		scores.append(score)
		print('\rEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_deque), score), end="")
		if i_episode % 100 == 0:
			torch.save(agent.actor_local.state_dict(), 'checkpoints/checkpoint_actor_{}.pth'.format(i_episode))
			torch.save(agent.critic_local.state_dict(), 'checkpoints/checkpoint_critic_{}.pth'.format(i_episode))
			print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))   
		logger.update_scores(score)
	return scores

start_time = time.time()
logger = DDPGLogger(start_time)
scores = ddpg(load=False)


# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(1, len(scores)+1), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()
