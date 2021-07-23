import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action

        #act_dim = self.dim_action * n_agent
        #ToDo Zastanow sie co powinien krytyk dostawac czy per land czy nie.
        act_dim = 1
        self.FC1 = nn.Linear(dim_observation+act_dim, 1024)
        self.FC2 = nn.Linear(1024, 4048)
        self.FC3 = nn.Linear(4048, 512)
        self.FC4 = nn.Linear(512, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):


        obs = th.flatten(obs)
        acts = th.flatten(acts)
        combined = th.cat([obs, acts], 0)
        result = F.relu(self.FC1(combined))
        combined = th.cat([result, acts], 0)
        result = F.relu(self.FC2(result))
        return self.FC4(F.relu(self.FC3(result)))

    # obs = th.flatten(obs)
    # acts = th.flatten(acts)
    # combined = th.cat([obs, acts], 0)
    # result = F.relu(self.FC1(combined))
    # combined = th.cat([result, acts], 0)
    # result = F.relu(self.FC2(result))
    # return self.FC4(F.relu(self.FC3(result)))

class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()

        self.FC1 = nn.Linear(dim_observation, 1024)
        self.FC2 = nn.Linear(1024, 2048)
        self.FC3 = nn.Linear(2048, 512)
        self.FC4 = nn.Linear(512, 1)

    # action output between -2 and 2
    def forward(self, obs):
        obs = th.flatten(obs)
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.relu(self.FC3(result))
        result = self.FC4(result)
        result = th.sigmoid(result)
        return result
