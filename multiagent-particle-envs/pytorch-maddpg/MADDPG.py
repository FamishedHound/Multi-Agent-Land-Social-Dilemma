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
                 capacity, episodes_before_train, worlds_all_agents, use_pretrained):
        # if not use_pretrained:
        #     self.initialise_networks(batch_size, dim_act, worlds_all_agents)
        # else:
        #     pass
        # implement loading the weights
        if use_pretrained:
            self.load_the_weights(batch_size, dim_act, worlds_all_agents, "after_curriculum")
        else:
            self.initialise_networks(batch_size, dim_act, worlds_all_agents)
        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episodes_before_train
        loss_new = nn.BCEWithLogitsLoss()
        self.GAMMA = 0.60
        self.tau = 0.95
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

    def initialise_networks(self, batch_size, dim_act, worlds_all_agents):
        self.actors = [Actor(len(a.land_cells_owned), dim_act) for a in worlds_all_agents]
        self.critics = [Critic(len(a.land_cells_owned), len(a.land_cells_owned),
                               batch_size) for a in worlds_all_agents]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

    def flush_memory(self):
        self.memory.flush_memory()

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
            # while transitions.count(None) !=0:
            #     print(" I WAS NONE")
            #     print(transitions)
            #     transitions = self.memory.sample(self.batch_size)
            # print(f"types {[type(x) for x in transitions]}")
            batch = Experience(*zip(*transitions))
            non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))

            S = np.stack([x[agent_index] for x in batch.states]).squeeze(1)

            S_prime = np.stack([x[agent_index] for x in batch.next_states]).squeeze(1)
            global_states = np.asarray(batch.global_state)
            global_states_prime = np.asarray(batch.next_global_state)

            action = np.stack([x[agent_index] for x in batch.actions])

            reward_batch = np.stack([x[agent_index] for x in batch.rewards])

            reward_batch = th.from_numpy(reward_batch).cuda()
            S_with_action = zip(th.from_numpy(global_states).cuda(), th.from_numpy(action).cuda())

            self.critic_optimizer[agent_index].zero_grad()

            current_Q = th.empty((self.batch_size))
            for i, d in enumerate(zip(global_states, action)):
                state, actions = d
                current_Q[i] = self.critics[agent_index](th.from_numpy(state).cuda(), th.from_numpy(actions).cuda())

            non_final_next_actions = th.empty((self.batch_size, len(agent.land_cells_owned)))
            for i, land_batch in enumerate(S_prime):
                non_final_next_actions[i] = self.actors_target[agent_index](
                    th.from_numpy(land_batch).float().cuda())  # ToDo Was actors_target

            S_prime_action_prime = zip(th.from_numpy(global_states_prime), non_final_next_actions)

            target_Q = th.empty((self.batch_size)).cuda()
            for i, d in enumerate(zip(global_states_prime, non_final_next_actions)):
                state, actions = d
                target_Q[i] = self.critics_target[agent_index](
                    th.from_numpy(state).cuda(),
                    actions.cuda())

            target_Q = (target_Q * self.GAMMA) + (
                reward_batch.cuda())

            loss_Q = nn.MSELoss()(current_Q.float().cuda(), target_Q.float().detach())
            # print(loss_Q.item())
            # self.loss_q.append(loss_Q)
            loss_Q.backward()
            # print(f"lossQ {loss_Q}")
            self.critic_optimizer[agent_index].step()

            self.actor_optimizer[agent_index].zero_grad()

            state_i = S.copy()

            action_i = th.empty((self.batch_size, len(agent.land_cells_owned)))
            for i, land_batch in enumerate(state_i):
                action_i[i] = self.actors[agent_index](
                    th.from_numpy(land_batch).float().cuda())

            state_i_with_action_i = zip(global_states.copy(), action_i)
            # actor_loss = [-self.critics[agent_index](th.from_numpy(whole_state).float().cuda(), whole_action.cuda()) for
            #               whole_state, whole_action in state_i_with_action_i]
            actor_loss = th.empty((self.batch_size)).cuda()
            for i, data in enumerate(state_i_with_action_i):
                batch_land, batch_action = data
                actor_loss[i] = -self.critics[agent_index](
                    th.from_numpy(batch_land).float().cuda(),
                    batch_action.cuda())

            actor_loss = actor_loss.view(self.batch_size, -1).mean()
            self.loss_list.append(actor_loss)
            actor_loss.backward()
            # print(actor_loss)
            self.actor_optimizer[agent_index].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)
            # print(f" EPPPPPPPPPPPPPPPPPPPPPPSILON {epsilon}")
            self.steps_done += 1

        update = 100
        if epsilon <= 0.2:
            pass

            # plt.show()
        if self.steps_done % update == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                hard_update(self.critics_target[i], self.critics[i])
                hard_update(self.actors_target[i], self.actors[i])
        # print("end of ITERATION")
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
            # print(decisions)
            tensor_actions = [th.tensor(x) for x in decisions.tolist()]
            actions.append(tensor_actions)

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

    def save_weights_of_networks(self, checkpoint):
        for i, networks in enumerate(zip(self.actors, self.critics, self.actors_target, self.critics_target)):
            a, at, c, ct = networks
            th.save(a.state_dict(),
                    f"C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\pytorch-maddpg\\saved_models\\network_{i}_actor_{checkpoint}.pt")
            th.save(at.state_dict(),
                    f"C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\pytorch-maddpg\\saved_models\\network_{i}_actor_target_{checkpoint}.pt")
            th.save(c.state_dict(),
                    f"C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\pytorch-maddpg\\saved_models\\network_{i}_critic_{checkpoint}.pt")
            th.save(ct.state_dict(),
                    f"C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\pytorch-maddpg\\saved_models\\network_{i}_critic_target_{checkpoint}.pt")

    def load_the_weights(self, batch_size, dim_act, worlds_all_agents, checkpoint):
        self.initialise_networks(batch_size, dim_act, worlds_all_agents)
        '''
        self.actors = []
        self.actors_target
        
        '''
        for i, networks in enumerate(zip(self.actors, self.critics, self.actors_target, self.critics_target)):
            a, at, c, ct = networks
            a.load_state_dict(th.load(
                f"C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\pytorch"
                f"-maddpg\\saved_models\\network_{i}_actor_{checkpoint}.pt"))
            a.eval()
            at.load_state_dict(th.load(
                f"C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\pytorch"
                f"-maddpg\\saved_models\\network_{i}_actor_target_{checkpoint}.pt"))
            at.eval()
            c.load_state_dict(th.load(
                f"C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\pytorch"
                f"-maddpg\\saved_models\\network_{i}_critic_{checkpoint}.pt"))
            c.eval()
            ct.load_state_dict(th.load(
                f"C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\multiagent-particle-envs\\pytorch"
                f"-maddpg\\saved_models\\network_{i}_critic_target_{checkpoint}.pt"))
            ct.eval()
