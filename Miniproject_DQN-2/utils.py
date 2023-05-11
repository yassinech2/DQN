
import numpy as np
import torch
from epidemic_env.env import Observation, ModelDynamics
from constants import *
from collections import namedtuple

class Utils : 
    def action_preprocessor(a:torch.Tensor,  dyn:ModelDynamics):
      
        last_action = dyn.get_action().copy()

        if a == Constants.ACTION_NULL:

            return last_action
        
        if a == Constants.ACTION_CONFINE:
            last_action['confinement'] = not last_action['confinement']
        elif a == Constants.ACTION_ISOLATE:
            last_action['isolation'] = not last_action['isolation']
        elif a == Constants.ACTION_VACCINATE:
            last_action['vaccinate'] = not last_action['vaccinate']
        elif a == Constants.ACTION_HOSPITAL:
            last_action['hospital'] = not last_action['hospital']
            
      
        return  last_action
        
    def observation_preprocessor(obs: Observation, dyn:ModelDynamics):
        infected = Constants.SCALE * np.array([np.array(obs.city[c].infected)/obs.pop[c] for c in dyn.cities])
        # compute power 1/4 of the infected
        infected = np.power(infected, 1/4)

        dead = Constants.SCALE * np.array([np.array(obs.city[c].dead)/obs.pop[c] for c in dyn.cities])
        # compute power 1/4 of the dead
        dead = np.power(dead, 1/4)
        
        return torch.Tensor(np.stack((dead, infected))).unsqueeze(0)

    def observation_preprocessor_action(obs: Observation, dyn:ModelDynamics):
        infected = Constants.SCALE * np.array([np.array(obs.city[c].infected)/obs.pop[c] for c in dyn.cities])
        # compute power 1/4 of the infected
        infected = np.power(infected, 1/4)
        dead = Constants.SCALE * np.array([np.array(obs.city[c].dead)/obs.pop[c] for c in dyn.cities])
        # compute power 1/4 of the dead
        dead = np.power(dead, 1/4)

        _res = torch.flatten(torch.Tensor(np.stack((dead, infected))))

        action = [int(obs.action['confinement']), int(obs.action['isolation']), int(obs.action['hospital']),int(obs.action['vaccinate'])]
        


 

       

        # concat res and action 
        return torch.cat((_res, torch.Tensor(action))).unsqueeze(0)
      

    Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
