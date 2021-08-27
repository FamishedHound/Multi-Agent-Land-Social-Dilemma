from random import choice

from model import Critic, Actor
import torch as th
from copy import deepcopy
from memory import ReplayMemory, Experience
from torch.optim import Adam, SGD
from randomProcess import OrnsteinUhlenbeckProcess
import torch.nn as nn
import numpy as np
from params import scale_reward
import matplotlib.pyplot as plt


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
                 capacity, episodes_before_train,worlds_all_agents):
        self.actors = [Actor(len(a.land_cells_owned), dim_act) for a in worlds_all_agents]
        self.critics = [Critic(len(a.land_cells_owned), len(a.land_cells_owned),
                               dim_act) for a in worlds_all_agents]
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
        loss_new = nn.BCEWithLogitsLoss()
        self.GAMMA = 0.99
        self.tau = 0.99
        self.lst1 = []
        self.lst2 = []
        self.var = [1.0 for _ in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(),
                                     lr=1e-4) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=1e-4) for x in self.actors]
        self.loss_list = []
        self.loss_q = []
        self.lst3 = []
        self.lst4 = []
        self.lst5 = []
        self.lst6 = []
        self.lst7 = []
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

    def update_policy(self, all_agents, epsilon, randy_random=None):
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
            S = np.stack([x[agent_index] for x in batch.states])
            # S = S.reshape(-1, *S.shape[2:])

            S_prime = np.stack([x[agent_index] for x in batch.next_states])

            # S_prime = S_prime.reshape(-1, *S_prime.shape[2:])
            action = np.stack([x[agent_index] for x in batch.actions])

            # action = action.reshape(-1, *action.shape[2:])
            # print(f"actions now {action}")
            reward_batch = np.stack([x[agent_index] for x in batch.rewards])
            # reward_batch = reward_batch[:,agent_index]
            # reward_batch = reward_batch.reshape(-1, *reward_batch.shape[2:])
            reward_batch = th.from_numpy(reward_batch).cuda()
            S_with_action = zip(th.from_numpy(S).cuda(), th.from_numpy(action).cuda())
            # print(f" reward { reward_batch}")
            # print(f"S prime {S_prime}")

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

            # current_Q = th.empty((self.batch_size, S.shape[1]))
            current_Q = th.empty((self.batch_size))
            for i, data in enumerate(S_with_action):
                batch_states, batch_actions = data
                current_Q[i]=self.critics[agent_index](batch_states.unsqueeze(0), batch_actions.unsqueeze(0))
                # for q, data in enumerate(zip(batch_states, batch_actions)):
                #     land, action = data
                #     current_Q[i, q] = self.critics[agent_index](land.float(), action)

            # current_Q = [
            #     self.critics[agent_index](current_land.float(), land_action) for
            #     current_land, land_action in S_with_action]  # S,a
            # current_Q = th.stack([x for x in current_Q]).squeeze().cuda()
            # Consier for batching
            # ToDo Turn it on for batching or different local obs
            # non_final_next_actions = [
            #     self.actors_target[i](non_final_next_states[:,
            #                                                 i,
            #                                                 :]) for i in range(
            #                                                     self.n_agents)]
            # non_final_next_actions = th.empty((self.batch_size, S.shape[1]))
            # non_final_next_actions = th.empty((self.batch_size,1))
            non_final_next_actions=th.empty((self.batch_size,len(agent.land_cells_owned)))
            for i, land_batch in enumerate(S_prime):

                non_final_next_actions[i] = self.actors_target[agent_index](th.from_numpy(land_batch).float().cuda()) #ToDo Was actors_target

                # for q, land in enumerate(land_batch):
                #     # if randy_random == 0:
                #     #     action= th.tensor(0.2)
                #     # else:
                #     #     action = th.tensor(0.6)
                #     non_final_next_actions[i, q] = self.actors_target[agent_index](th.from_numpy(land).unsqueeze(
                #         0).float().cuda())  # th.tensor(choice([0.1,0.2,0.3,0.4,0.5,0.6]))#

            # non_final_next_actions = [self.actors_target[agent_index](th.from_numpy(land_prime).unsqueeze(0).float().cuda())
            #                           for land_prime in S_prime]
            # S' => a'

            # print(f"{non_final_next_actions} output of future actions")
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
            # target_Q = th.empty((self.batch_size, S.shape[1])).cuda()
            target_Q = th.empty((self.batch_size))
            for i, data in enumerate(S_prime_action_prime):
                batch_land, batch_action = data
                target_Q[i] = self.critics_target[agent_index](
                        batch_land.float().cuda(),
                        batch_action.cuda())
                # for q, data in enumerate(zip(batch_land, batch_action)):
                #     land, action = data
                #     target_Q[i, q] = self.critics_target[agent_index](
                #         land.unsqueeze(0).float().cuda(),
                #         action.cuda())

            # target_Q = [self.critics_target[agent_index](
            #     future_land.unsqueeze(0).float().cuda(),
            #     future_action.cuda()
            # ) for future_land, future_action in S_prime_action_prime ]  # S' a'
            # target_Q = th.stack([x for x in target_Q]).squeeze()
            # scale_reward: to scale reward in Q functions
            # ToDo Turn it on for batching or different local obs
            # target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
            #         reward_batch[:, agent_index].unsqueeze(1) * scale_reward)
            # current_Q = np.array(current_Q)
            # target_Q = np.array(target_Q)
            target_Q = target_Q.view(self.batch_size, -1).cuda()
            reward_batch = reward_batch.view(self.batch_size, -1).cuda()
            current_Q = current_Q.view(self.batch_size, -1).cuda()
            target_Q = (target_Q * self.GAMMA) + (
                reward_batch.cuda())
            # current_Q = np.expand_dims(current_Q, 0)
            # target_Q = np.expand_dims(target_Q, 0)
            #
            # current_Q = th.stack([x.view(1, -1) for x in current_Q[0]])
            # target_Q = th.stack([x.view(1, -1) for x in target_Q[0]])

            loss_Q = nn.MSELoss()(current_Q.float().cuda(), target_Q.float().detach())
            # print(loss_Q.item())
            self.loss_q.append(loss_Q)
            loss_Q.backward()
            #print(f"lossQ {loss_Q}")
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
            state_i = S.copy()
            # action_i = [self.actors[agent_index](th.from_numpy(state).unsqueeze(0).float().cuda()) for state in state_i]
            # action_i = th.empty((self.batch_size, S.shape[1])).cuda()
            action_i = th.empty((self.batch_size,len(agent.land_cells_owned)))
            for i, land_batch in enumerate(state_i):
                action_i[i] = self.actors[agent_index](
                         th.from_numpy(land_batch).float().cuda())
                # for q, land in enumerate(land_batch):
                #     action_i[i, q] = self.actors[agent_index](
                #         th.from_numpy(land).unsqueeze(0).float().cuda())
            state_i_with_action_i = zip(state_i, action_i)
            # actor_loss = [-self.critics[agent_index](th.from_numpy(whole_state).float().cuda(), whole_action.cuda()) for
            #               whole_state, whole_action in state_i_with_action_i]
            actor_loss = th.empty((self.batch_size)).cuda()
            for i, data in enumerate(state_i_with_action_i):
                batch_land, batch_action = data
                actor_loss[i] = -self.critics[agent_index](
                             th.from_numpy(batch_land).float().cuda(),
                            batch_action.cuda())
                # for q, data in enumerate(zip(batch_land, batch_action)):
                #     land, action = data
                #     actor_loss[i, q] = \
                #         -self.critics[agent_index](
                #             th.from_numpy(land).unsqueeze(0).float().cuda(),
                #             action.cuda())
            # ToDO ask LEANDRO WHAT SHOULD HAPPEN HERE

            # actor_loss = th.stack([x.view(1, -1) for x in actor_loss]).mean()
            # print(f"{actor_loss} actor loss")
            actor_loss = actor_loss.view(self.batch_size, -1).mean()
            self.loss_list.append(actor_loss)
            actor_loss.backward()
            # print(actor_loss)
            self.actor_optimizer[agent_index].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)
            #print(f" EPPPPPPPPPPPPPPPPPPPPPPSILON {epsilon}")
            self.steps_done += 1
            # self.Q_test(agent_index)
            # self.lst1.append(main)
            # # self.lst2.append(self.Q_test(agent_index)[1])
            # #
            # self.lst2.append(other[0])
            # self.lst3.append(other[1])
            # self.lst4.append(other[2])
            # self.lst5.append(other[3])
            # self.lst6.append(other[4])
            # if epsilon < 0 :
            #     plt.plot(self.lst1[100:], color='black')
            #     plt.plot(self.lst2[100:])
            #     plt.plot(self.lst3[100:])
            #     plt.plot(self.lst4[100:])
            #     plt.plot(self.lst5[100:])
            #     plt.plot(self.lst6[100:])
            #
            #     plt.show()

        update = 60
        if epsilon <= 0.2:
            pass
            # plt.plot(self.loss_list)
            # plt.show()
            # plt.plot(self.loss_q, color='black')
            # plt.show()
        if self.steps_done % update == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)
        #print("end of ITERATION")
        return c_loss, a_loss

    def Q_test(self, agent_index):
        good = th.from_numpy(np.array([[[1]], [[0.2]], [[0.2]]]))
        rest = [(th.from_numpy(np.array([[[1]], [[i]], [[i]]])), th.tensor(i)) for i in [0.1, 0.3, 0.4, 0.5, 0.6]]
        action_good = th.tensor(0.2)
        action_bad = th.tensor(0.5)
        good_result_Q = self.actors[agent_index](
            good.unsqueeze(0).float().cuda())
        # other = []
        # for x,y in rest:
        #     other.append(self.actors[agent_index](
        #         x.unsqueeze(0).float().cuda(),
        #         y.cuda()))
        # bad_result_Q = self.critics_target[agent_index](
        #     bad.unsqueeze(0).float().cuda(),
        #     action_bad.cuda())
        print(f" good action {good_result_Q}")
        # print(f"bad Q {bad_result_Q}")
        # return good_result_Q,other#,bad_result_Q

    def select_action(self, state_batch, all_agents):
        # state_batch: n_agents x state_dim
        actions = []

        # action_dict = {i:[] for i,_ in enumerate(all_agents)}
        # print([" "+str(agent.agent_id)+" " for agent in all_agents])

        for agent_index, agent in enumerate(all_agents):
            # using global

            sb = np.asarray(state_batch[agent_index])
            agent_actions = []
            decisions = self.actors[agent_index](
                    th.from_numpy(np.array(sb)).unsqueeze(0).float().cuda()).squeeze().data.cpu()
            tensor_actions = [th.tensor(x) for x in decisions.tolist()]
            actions.append(tensor_actions)
            print()
            # for i, land in enumerate(sb):
            #     # land_obs = th.rand((3,2,2))
            #     fluid_state = np.asarray(land).copy()
            #     decision = self.actors[agent_index](
            #         th.from_numpy(np.array(land)).float().unsqueeze(0).cuda()).squeeze().data.cpu()
            #     agent_actions.append(decision)
            #     # x,y = np.where(sb[i][0]==1)
            #     # x= x.item()
            #     # y = y.item()
            #     # fluid_state[1][x,y] = decision
            #     # for j in range(sb.shape[0]):
            #     #     sb[j][1] = fluid_state[1]
            # actions.append(agent_actions)

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
