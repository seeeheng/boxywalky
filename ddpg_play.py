import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as pltthon
import time

# %matplotlib inline
from ddpg_agentTest import Agent

env = gym.make('BipedalWalker-v2')
env.seed(10)
agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=10)

agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

for i in range(10):
    state = env.reset()
    score = 0

    while True:    
        action = agent.act(state)
        env.render()
        next_state, reward, done, _ = env.step(action)
        score += reward
        state = next_state
        if done:
            break

    print("Trial {}, Score: {}".format(i+1,score))

            
env.close()