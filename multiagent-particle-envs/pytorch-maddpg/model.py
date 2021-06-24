import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, dim_action):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        self.dim_observation = dim_observation
        self.dim_action = dim_action
        obs_dim = 18 * n_agent
        #act_dim = self.dim_action * n_agent
        #ToDo Zastanow sie co powinien krytyk dostawac czy per land czy nie.
        act_dim = 1
        self.FC1 = nn.Linear(108, 1024)
        self.FC2 = nn.Linear(1024+act_dim, 512)
        self.FC3 = nn.Linear(512, 300)
        self.FC4 = nn.Linear(300, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        obs = th.flatten(obs)
        acts = th.flatten(acts)
        result = F.relu(self.FC1(obs))
        combined = th.cat([result, acts], 0)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result))) # Q_value estimation


class Actor(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(Actor, self).__init__()

        self.FC1 = nn.Linear(dim_observation, 500)
        self.FC2 = nn.Linear(500, 128)
        self.FC3 = nn.Linear(128, 1)

    # action output between -2 and 2
    def forward(self, obs):
        obs = th.flatten(obs)
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = th.sigmoid(F.relu(self.FC3(result))) # actions
        return result
