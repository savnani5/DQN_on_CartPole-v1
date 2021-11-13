import gym
import time
import ipdb
import numpy as np


env = gym.make("CartPole-v1")
observation = env.reset()

done = False
while not done:
    env.render()
    time.sleep(0.1)
    ipdb.set_trace()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    
    if done:
        observation = env.reset()

env.close()
    

