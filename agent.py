import gym
import time
from tqdm import tqdm
import ipdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
import torch.optim as optim
from dataclasses import dataclass
from typing import Any
from random import sample, random
import wandb
from collections import deque

@dataclass
class Sarsd:
    """State--Action--Reward--next_state--done"""
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool


class DQNagent:
    """ Deep Q Network Agent class"""
    
    def __init__(self, model):
        self.model = model

    def get_actions(self, observations):
        # observations shape = (N,4)
        q_vals = self.model(observations)
        
        # q_vals shape = (N,2)
        return q_vals.max(-1)[1]

class Model(nn.Module):
    """Pytorch Perceptron model"""

    def __init__(self, obs_shape, num_actions):
        super(Model, self).__init__()
        assert len(obs_shape) == 1, "This network only works for flat observations"
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_shape[0], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_actions),
        )
        self.opt = optim.Adam(self.net.parameters(), lr=0.0001)

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    """Experience Replay Buffer queue"""
    
    def __init__(self, buffer_size = 100000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
    
    def insert(self, sarsd):
        self.buffer.append(sarsd)

    def sample(self, num_samples):
        assert num_samples <= len(self.buffer)
        return sample(self.buffer, num_samples)


def update_tgt_model(m, tgt):
    tgt.load_state_dict(m.state_dict())

def train_step(model, state_transitions, tgt, num_actions, device):
    curr_states = torch.stack([torch.Tensor(s.state) for s in state_transitions]).to(device)
    rewards = torch.stack([torch.Tensor([s.reward]) for s in state_transitions]).to(device)
    
    # Condition for ending the game 
    mask = torch.stack([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions]).to(device)
    next_states = torch.stack([torch.Tensor(s.next_state) for s in state_transitions]).to(device)
    actions = [s.action for s in state_transitions]                        # Should be one hot encoded

    with torch.no_grad():
        qvals_next = tgt(next_states).max(-1)[0]

    model.opt.zero_grad()
    qvals = model(curr_states)
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)
    
    # ipdb.set_trace() 
    loss = ((rewards + mask[:, 0]*qvals_next - torch.sum(qvals*one_hot_actions, -1))**2).mean()
    loss.backward()
    model.opt.step()
    return loss


def main(test=False, chkpt=None, device="cuda"):
    
    if not test:
        wandb.init(project="dqn", name="cartpole")
    min_rb_size = 10000
    sample_size = 2500
    eps_min = 0.01
    eps_decay = 0.999995


    env_steps_before_train = 100
    tgt_model_update = 150

    env = gym.make("CartPole-v1")
    last_observation = env.reset()

    # Loading the model
    m = Model(env.observation_space.shape, env.action_space.n).to(device)
    if chkpt:
        m.load_state_dict(torch.load(chkpt))
    
    tgt = Model(env.observation_space.shape, env.action_space.n).to(device)
    update_tgt_model(m, tgt)

    rb = ReplayBuffer()
    steps_since_train = 0
    epochs_since_tgt = 0

    step_num = -1*min_rb_size

    episode_rewards = []
    rolling_reward = 0 

    tq = tqdm()
    try:
        while True:
            if test:
                env.render()
                time.sleep(0.05)

            tq.update(1)

            eps = eps_decay**step_num
            if test: 
                eps = 0

            if random() < eps:
                action = env.action_space.sample()
            else:
                action = m(torch.Tensor(last_observation).to(device)).max(-1)[-1].item()
            
            
            observation, reward, done, info = env.step(action)
            rolling_reward += reward
            reward = reward/100.0       # Scaling the reward

            rb.insert(Sarsd(last_observation, action, reward, observation, done))
            last_observation = observation
            
            if done:
                episode_rewards.append(rolling_reward)
                
                if test:
                    print(rolling_reward)
                
                rolling_reward = 0 
                observation = env.reset()

            steps_since_train +=1
            step_num +=1

            if (not test) and len(rb.buffer) > min_rb_size and steps_since_train > env_steps_before_train:
                epochs_since_tgt +=1
                loss = train_step(m, rb.sample(sample_size), tgt, env.action_space.n, device)
                wandb.log({'loss': loss.detach().cpu().item(), 
                            'eps': eps, 
                            'avg_reward': np.mean(episode_rewards)}, step=step_num)
                # print(step_num, loss.detach().item())
                episode_rewards = []

                if epochs_since_tgt > tgt_model_update:
                    print("Updating target model")
                    update_tgt_model(m, tgt)
                    epochs_since_tgt = 0
                    torch.save(tgt.state_dict(), f"models/{step_num}.pth")

                steps_since_train = 0

    except KeyboardInterrupt:
        pass

    env.close()
        

if __name__ == "__main__":
    # main(True, "models/259167.pth")
    main()
