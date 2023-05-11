
from epidemic_env.agent import Agent
from epidemic_env.env import Env

from constants import *



class Russo(Agent):
    def __init__(self,  env:Env,threshhold=20000,
                ):
        """
        Russo's Policy agent implementation. Confine entire country for 4 week of nb_infected > threshhold .
        """
        self.env = env
        self.threshhold = threshhold
        self.counter = 0 # start with no confinement, track number of weeks of confinement


        
    def load_model(self, savepath):
        # This is where one would define the routine for loading a pre-trained model
        pass

    def save_model(self, savepath):
        # This is where one would define the routine for saving the weights for a trained model
        pass

    def optimize_model(self):
        # This is where one would define the optimization step of an RL algorithm
        return 0
    
    def reset(self,):
        # This should be called when the environment is reset
        self.counter= 0 
    
    def act(self, obs):
        # this takes an observation and returns an action
        # the action space can be directly sampled from the env
       
        if self.counter < 4 and self.counter > 0 :#and self.env.last_action == Constants.ACTION_CONFINE :
            action = Constants.ACTION_CONFINE
            self.counter += 1
        else:
            if obs.total.infected[-1]  > self.threshhold : 
             
                action = Constants.ACTION_CONFINE
                self.counter = 1
            else:
                action = Constants.ACTION_NULL
                self.counter = 0

        return action
          