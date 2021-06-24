from model import Critic, Actor
import torch as th
from copy import deepcopy
from memory import ReplayMemory, Experience
from torch.optim import Adam
from randomProcess import OrnsteinUhlenbeckProcess
import torch.nn as nn
import numpy as np
from params import scale_reward


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)

def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)
class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size,
                 capacity, episodes_before_train):
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs,
                               dim_act) for _ in range(n_agents)]
        # self.actors = [a.apply(weights_init_uniform) for a in self.actors]
        # self.critics = [c.apply(weights_init_uniform) for c in self.critics]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for _ in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update_policy(self, all_agents):
        # do not train until exploration is enough
        if self.episode_done <= self.batch_size:
            return None, None

        ByteTensor = th.cuda.ByteTensor if self.use_cuda else th.ByteTensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        c_loss = []
        a_loss = []
        for agent_index, agent in enumerate(all_agents):
            transitions = self.memory.sample(self.batch_size)  # S,a, S',r
            batch = Experience(*zip(*transitions))
            non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))

            # next_state = list(batch.next_states)
            # non_final_mask = next_state
            # state_batch: batch_size x n_agents x dim_obs
            # state_batch = th.stack(batch.states).type(FloatTensor)
            # action_batch = th.stack(batch.actions).type(FloatTensor)
            # reward_batch = th.stack(batch.rewards).type(FloatTensor)

            # state_batch = th.stack(batch.states).type(FloatTensor)
            # action_batch = batch.actions
            # reward_batch = th.stack(batch.rewards).type(FloatTensor)
            S = np.stack([x[agent_index] for x in batch.states ])
            S = S.reshape(-1, *S.shape[2:])
            S_prime = np.stack([x[agent_index] for x in batch.next_states ])
            S_prime = S_prime.reshape(-1, *S_prime.shape[2:])
            action = np.stack([x[agent_index] for x in batch.actions ])
            action = action.reshape(-1, *action.shape[2:])
            reward_batch = np.stack([x[agent_index] for x in batch.rewards])
            reward_batch = reward_batch.reshape(-1, *reward_batch.shape[2:])
            reward_batch = th.from_numpy(reward_batch)
            S_with_action = zip(th.from_numpy(S), th.from_numpy(action))
            print()

            # : (batch_size_non_final) x n_agents x dim_obs
            # ToDo Turn it on for batching or different local obs
            # non_final_next_states = th.stack(
            #     [th.from_numpy(s) for s in batch.next_states
            #      if s is not None]).type(FloatTensor)
            # non_final_next_states = th.from_numpy(list(batch.next_states)[0]).unsqueeze(0)
            # for current agent
            # whole_state = state_batch.view(self.batch_size, -1)
            # whole_action = action_batch.view(self.batch_size, -1)
            # whole_state = state_batch
            # whole_action = action_batch
            self.critic_optimizer[agent_index].zero_grad()
            current_Q = [
                self.critics[agent_index](current_land.float(), land_action) for
                current_land, land_action in S_with_action]  # S,a
            current_Q = th.stack([x for x in current_Q]).squeeze()
            # Consier for batching
            # ToDo Turn it on for batching or different local obs
            # non_final_next_actions = [
            #     self.actors_target[i](non_final_next_states[:,
            #                                                 i,
            #                                                 :]) for i in range(
            #                                                     self.n_agents)]

            non_final_next_actions = [self.actors_target[agent_index](th.from_numpy(land_prime).unsqueeze(0).float())
                                      for land_prime in S_prime]
            # S' => a'
            non_final_next_actions = th.stack([x for x in non_final_next_actions]).squeeze()
            S_prime_action_prime = zip(th.from_numpy(S_prime), non_final_next_actions)
            # ToDo Turn it on for batching or different local obs
            # non_final_next_actions = th.stack(non_final_next_actions)
            # ToDo Turn it on for batching or different local obs
            # non_final_next_actions = (
            #     non_final_next_actions.transpose(0,
            #                                      1).contiguous())
            # ToDo Turn it on for batching or different local obs
            # target_Q = th.zeros(
            #     self.batch_size).type(FloatTensor)
            #
            # target_Q[non_final_mask] = self.critics_target[agent](
            #     non_final_next_states.view(-1, self.n_agents * self.n_states),
            #     non_final_next_actions.view(-1,
            #                                 self.n_agents * self.n_actions)
            # ).squeeze()
            # ToDo Turn it on for batching or different local obs
            # target_Q = th.zeros(
            #
            #     self.batch_size).type(FloatTensor)
            # ToDO SUSPICIOS WHY THEY DO [non_final_mask]
            # target_Q[non_final_mask] = self.critics_target[agent_index](
            #     non_final_next_states,
            #     non_final_next_actions
            # ).squeeze()

            target_Q = [self.critics_target[agent_index](
                future_land.unsqueeze(0).float(),
                future_action
            ) for future_land, future_action in S_prime_action_prime]  # S' a'
            target_Q = th.stack([x for x in target_Q]).squeeze()
            # scale_reward: to scale reward in Q functions
            # ToDo Turn it on for batching or different local obs
            # target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
            #         reward_batch[:, agent_index].unsqueeze(1) * scale_reward)
            # current_Q = np.array(current_Q)
            # target_Q = np.array(target_Q)
            target_Q = (target_Q * self.GAMMA) + (
                    reward_batch )
            # current_Q = np.expand_dims(current_Q, 0)
            # target_Q = np.expand_dims(target_Q, 0)
            #
            # current_Q = th.stack([x.view(1, -1) for x in current_Q[0]])
            # target_Q = th.stack([x.view(1, -1) for x in target_Q[0]])

            loss_Q = nn.MSELoss()(current_Q.float(), target_Q.float().detach())
            #print(loss_Q.item())
            loss_Q.backward()
            self.critic_optimizer[agent_index].step()

            # self.actor_optimizer[agent_index].zero_grad()
            # state_i = state_batch[:, agent_index, :]
            # action_i = self.actors[agent_index](state_i)
            # ac = action_batch.clone()
            # ac[:, agent_index, :] = action_i
            # whole_action = ac.view(self.batch_size, -1)
            # actor_loss = -self.critics[agent_index](whole_state, whole_action)
            # actor_loss = actor_loss.mean()
            # actor_loss.backward()
            # self.actor_optimizer[agent_index].step()
            # c_loss.append(loss_Q)
            # a_loss.append(actor_loss)

            self.actor_optimizer[agent_index].zero_grad()

            # state_i = state_batch
            # action_i = self.actors[agent_index](state_i)
            # ac = action_batch.clone().squeeze()
            # ac[agent_index] = action_i
            # whole_action = ac.view(self.batch_size, -1)
            # actor_loss = -self.critics[agent_index](whole_state, whole_action)
            # actor_loss = actor_loss.mean()
            # actor_loss.backward()
            # self.actor_optimizer[agent_index].step()
            # c_loss.append(loss_Q)
            # a_loss.append(actor_loss)
            state_i = S
            action_i = [self.actors[agent_index](th.from_numpy(state).unsqueeze(0).float()) for state in state_i]

            state_i_with_action_i = zip(state_i,action_i)
            actor_loss = [-self.critics[agent_index](th.from_numpy(whole_state).float(), whole_action) for whole_state,whole_action in state_i_with_action_i]
            #ToDO ask LEANDRO WHAT SHOULD HAPPEN HERE

            actor_loss = th.stack([x.view(1, -1) for x in actor_loss]).mean()
            actor_loss.backward()
            self.actor_optimizer[agent_index].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)
            self.steps_done += 1
        if self.steps_done % 10 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, state_batch, all_agents):
        # state_batch: n_agents x state_dim
        actions = []

        # action_dict = {i:[] for i,_ in enumerate(all_agents)}
        # print([" "+str(agent.agent_id)+" " for agent in all_agents])

        for agent_index, agent in enumerate(all_agents):
            # using global

            sb = state_batch[agent_index]
            agent_actions = []
            for i, land in enumerate(agent.land_cells_owned):
                land_obs = th.from_numpy(sb[i]).float()

                agent_actions.append(self.actors[agent_index](land_obs.unsqueeze(0)).squeeze().data.cpu())

            actions.append(agent_actions)
        return actions

    #     act += th.from_numpy(
    #         np.random.randn(2) * self.var[i]).type(FloatTensor)
    #
    #     if self.episode_done > self.episodes_before_train and\
    #        self.var[i] > 0.05:
    #         self.var[i] *= 0.999998
    #     act = th.clamp(act, -1.0, 1.0)
    #
    #     actions[i, :] = act
    # self.steps_done += 1
    def averageOfList(self, numOfList):
        avg = sum(numOfList) / len(numOfList)
        return avg
