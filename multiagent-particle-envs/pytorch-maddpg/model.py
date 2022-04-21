import torch as th
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, n_agent, dim_observation, batch_size):
        super(Critic, self).__init__()
        self.n_agent = n_agent
        # self.dim_observation = dim_observation
        # self.dim_action = dim_action
        self.dim_observation = 48
        self.dim_action = 4

        #act_dim = self.dim_action * n_agent
        #ToDo Zastanow sie co powinien krytyk dostawac czy per land czy nie.
        act_dim = 1
        print(f"critic dims {dim_observation*12+dim_observation}")
        self.FC1 = nn.Linear(32+dim_observation, 512)
        self.FC2 = nn.Linear(512, 512)
        self.FC3 = nn.Linear(1024, 512)
        self.FC4 = nn.Linear(512, 1)

    # obs: batch_size * obs_dim
    def forward(self, obs, acts):
        # obs = self.norm1(obs)
        # acts = self.norm2(acts)

        obs = th.flatten(obs.float())
        acts = th.flatten(acts).float()

        combined = th.cat([obs, acts], 0)

        result = F.relu(self.FC1(combined))
        result = F.relu(self.FC2(result))
        result =  self.FC4(result)


        return result

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
        print(f"agent dims {dim_observation*18}")
        self.FC1 = nn.Linear(32, 1024)
        self.FC2 = nn.Linear(1024, 2048)
        self.FC3 = nn.Linear(1024, 512)

        self.FC4 = nn.Linear(2048, dim_observation)

    # action output between -2 and 2
    def forward(self, obs):
        obs = th.flatten(obs)
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))


        result = self.FC4(result)
        result = th.sigmoid(result)
        return result
class MultiPurposemetaLearner(nn.Module):
    #in
    def __init__(self):
        super().__init__()
        self.number_of_agents = 4
        self.FC1 = nn.Linear(16, 512)
        self.FC2 = nn.Linear(512, 1028)


        self.FC4_ = nn.Linear(1028, 2056)


    def forward(self,obs):
        obs_visual = obs
        obs = th.flatten(obs)
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))

        result = F.softmax(result,dim=1)

        #Possibly multiply
        result = self.FC1(obs)
        result = self.FC1(obs)
