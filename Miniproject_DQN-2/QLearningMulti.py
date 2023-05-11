

from constants import *
from utils import Utils
from epidemic_env.agent import Agent
from QLearningModel import QLearningModel , ReplayMemory
import torch
import random
import torch.nn as nn




BATCH_SIZE = 2048

class QAgentMulti(Agent):

    def __init__(self, env,device,input_size):
        self.env = env

        self.policy_network = QLearningModel(input_size=input_size,output_size=5)
        self.target_network = QLearningModel(input_size=input_size,output_size=5)
        self.input_size = input_size
        self.memory = ReplayMemory(20000)
        self.device = device

    
    def load_model(self, savepath):

        self.policy_network.load_state_dict(torch.load(savepath+str("policy.pt")))
        self.target_network.load_state_dict(torch.load(savepath+str("target.pt")))
        return
      
        

    def save_model(self, savepath):
        torch.save(self.policy_network.state_dict(), savepath+str("policy")+str(".pt"))
        torch.save(self.target_network.state_dict(), savepath+str("target")+str(".pt"))
        return

    def remember(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)


    def optimize_model(self, optimizer,episode,gamma=0.9):

        batch_size = BATCH_SIZE
        if len(self.memory) < BATCH_SIZE:
            batch_size = len(self.memory)

        transitions = self.memory.sample(batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Utils.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        
      
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

      

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)


        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]
        # Compute the expected Q values

        expected_state_action_values = (next_state_values * gamma) + reward_batch.squeeze(1)

 
        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)


        optimizer.step()
        if episode % 5 == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
            
        

       
    
    def reset(self):
        # This should be called when the environment is reset
        self.policy_network = QLearningModel(input_size=self.input_size,output_size=5)
        self.target_network = QLearningModel(input_size=self.input_size,output_size=5)
        self.memory = ReplayMemory(20000)

    def act(self, obs,eps,eps_decay=False,eps_min=None,num_episodes=None,episode=None):
        # this takes an observation and returns an action
        # the action space can be directly sampled from the env
        sample = random.random()
        if eps_decay : 
            eps_threshold = max(eps_min,eps *(num_episodes - episode ) / num_episodes)
        else : 
            eps_threshold = eps

        if sample < 1 - eps_threshold:
            with torch.no_grad():
               
                
                output =  self.policy_network(obs).max(1)[1].view(1, 1)
                return output
      
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)


    
        
