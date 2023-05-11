import torch.nn as nn
import random
from collections import namedtuple, deque
import torch.nn.functional as F
from  utils  import Utils






class QLearningModel(nn.Module):

    def __init__(self,input_size=6 ,  output_size=2):
        super(QLearningModel, self).__init__()
        self.fc0 = nn.Linear(input_size, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_size)

    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    



class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Utils.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    