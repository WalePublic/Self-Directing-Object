# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 2023

@author: talkt
"""

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from collections import deque


#Create the architecture of the neural network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fcl = nn.Linear(input_size,30)
        self.fc2 = nn.Linear(30, nb_action)
        

    # Forward propagation function that determines how car advances
    def forward(self, state):
        x = tnf.relu(self.fcl(state))
        q_values = self.fc2(x)
        return q_values
    
#Implement experience replay
class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []  #TODO. why not use deque instead
        
    def push(self, event):
        self.memory.append(event)
        
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map( lambda x: Variable(torch.cat(x, 0)), samples)
    
# Implement Deep Q Learning    
class Dqn():
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = deque()
        self.model = Network(input_size, nb_action)
        self.r_memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state):
        probs = tnf.softmax(self.model(Variable(state, volatile = True))*30) 
        #Temperate = 7, to remove the ai set to zero
        action = probs.multinomial(num_samples=1)
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = tnf.smooth_l1_loss(outputs, target)
        
        #back propagate
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph=True)
        self.optimizer.step()
    
    # Update all states
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.r_memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]),torch.LongTensor([self.last_reward])))
        action = self.select_action(new_state)
        
        if len(self.r_memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.r_memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        
        if len(self.reward_window) > 1000:
            self.reward_window.popleft()
            
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'model_state': self.model.state_dict(), 
                    'optimizer_state': self.optimizer.state_dict(),
                    }, 'latest_state.pth')
        
    def load(self):
        if os.path.isfile('latest_state.pth'):
            print("loading saved state")
            saved_state = torch.load('latest_state.pth')
            self.model.load_state_dict(saved_state['model_state'])
            self.optimizer.load_state_dict(saved_state['optimizer_state'])
        else:
            print("No previous state found.")