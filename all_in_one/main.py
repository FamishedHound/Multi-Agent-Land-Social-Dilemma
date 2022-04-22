import random
import warnings
import random
from typing import List
import gym
import sys

import pygame
from gym import spaces
import numpy as np
import torch as th
import pygame
from random import choice
import math
import random

from scipy.spatial import distance
from model import Critic, Actor
import torch as th
from copy import deepcopy
from memory import ReplayMemory, Experience
from torch.optim import Adam, SGD
from randomProcess import OrnsteinUhlenbeckProcess
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import torch.nn.functional as F
class Grid:
    def __init__(self):
        from LandCell import LandCell
        self.all_cells = {(x, y): LandCell(x, y) for x in range(GlobalParamsGame.MAX_CELLS_NUMER) for y in
                          range(GlobalParamsGame.MAX_CELLS_NUMER)}
        self.image = pygame.image.load(
            'C:\\Users\\LukePC\PycharmProjects\\polinators_social_dilema\\images\\117480-middle.png')
        pygame.font.init()
        self.font = pygame.font.SysFont("Grobold", 25,italic=True)
    def get_cell(self,cords):
        for k,v in self.all_cells.items():
            if k == cords:
                return v
    def drawGrid(self):







        for cell in self.all_cells.values():



            self.image = pygame.transform.scale(self.image, (40,40)).convert_alpha()
            self.image.set_alpha(40)
            txt_surf = self.font.render(str(cell.bag_pointer_actual), True, cell.owner.color2)
            txt_surf2 = self.font.render(f"agent : {str(cell.owner.id)}" , True, cell.owner.color2)
            txt_surf3 = self.font.render(f"money : {str(cell.owner.money)}", True, cell.owner.color2)
            txt_surf4 = self.font.render(f"alpha : {str(cell.owner.alpha)}", True, cell.owner.color2)
            if cell.is_owned:

                pygame.draw.rect(GlobalParamsGame.SCREEN, cell.owner.color, cell.get_rect())
                pygame.draw.rect(GlobalParamsGame.SCREEN, cell.owner.color2, cell.get_rect(), 2)
                GlobalParamsGame.SCREEN.blit(txt_surf, (cell.get_rect().x+4,cell.get_rect().y+10))
                GlobalParamsGame.SCREEN.blit(txt_surf2, (cell.get_rect().x + 20, cell.get_rect().y + 50))
                GlobalParamsGame.SCREEN.blit(txt_surf3, (cell.get_rect().x + 20, cell.get_rect().y + 80))
                GlobalParamsGame.SCREEN.blit(txt_surf4, (cell.get_rect().x + 20, cell.get_rect().y + 110))
            if cell.was_pollinated:
                GlobalParamsGame.SCREEN.blit(self.image, (cell.get_rect().x+70,cell.get_rect().y))


    def draw_rect_alpha(self, surface, color, rect):
        shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
        surface.blit(shape_surf, rect)

class LandCell:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x * GlobalParamsGame.BLOCKSIZE, y * GlobalParamsGame.BLOCKSIZE,
                                GlobalParamsGame.BLOCKSIZE, GlobalParamsGame.BLOCKSIZE)
        self.is_owned = False
        self.owner = None
        self.x = x
        self.y = y
        self.is_pollinator = False
        self.bag_pointer_declared = 0
        self.bag_pointer_actual = 0
        self.was_pollinated = False
        self.dead = False
        self.last_income = 0
    def get_rect(self):
        return self.rect

    def set_owner(self, owner):
        self.owner = owner
        owner.no_already_assigned_lands+=1

    def set_owned(self, is_owned: bool):
        self.is_owned = is_owned

    def euclidian_distance(self):
        pass
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

class Agent:
    def __init__(self, id, pos_x: int, pos_y: int, number_of_lands: int, agent_type=None):
        self.agent_id = (pos_x, pos_y)
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.agent_type = agent_type
        self.id = id
        self.land_cells_owned: List[LandCell] = []
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.predefined_number_of_lands = number_of_lands
        self.no_already_assigned_lands = 0
        self.money = 0
        self.is_dead = False
        self.last_income = None

        self.money = GlobalEconomyParams.STARTING_GOLD

    def __eq__(self, other):
        return self.pos_x == other.pos_x and self.pos_y == other.pos_y
class LandAgent(Agent):

    def __init__(self, id, pos_x: int, pos_y: int, number_of_lands: int, agent_type, pollinators_processor):
        super().__init__(id, pos_x, pos_y, number_of_lands)
        self.utility = 0
        self.agent_type = agent_type
        self.pollinators_processor = pollinators_processor
        self.money_past = []
        self.average_past = []
        self.pollination_memory = {}
        self.my_fees = len(self.land_cells_owned) * GlobalEconomyParams.LAND_UPCOST
        self.income = 0
        self.observation_counter = 0
        self.emergency_counter = 0
    # If I have pollinator where it can pollinate
    # Look out for pollinators from neighbours and free ride

    def make_a_decision(self):
        if not self.pollination_memory:
            self.pollination_memory = {(k.x, k.y): [] for k in self.land_cells_owned}
        self.take_observation_of_pollination()
        lands_to_process = [x for x in self.land_cells_owned]
        my_pollinators = self.find_my_pollinators()
        closest_pols = []
        for pollinator in my_pollinators:
            closest_pols = self.find_closest_lands_in_my_farm((pollinator.x, pollinator.y), 2)
            for x in closest_pols:
                if pollinator.bag_pointer_actual >= x.bag_pointer_actual:
                    x.bag_pointer_declared = 0

        remaining_lands = [x for x in lands_to_process if x not in closest_pols and x not in my_pollinators]

        self.analyze_current_situation(remaining_lands)

        self.money_past.append(self.income)

    def analyze_current_situation(self, lands_to_analyze):
        self.emergency_counter_measure()

        if self.emergency_counter<3:
            for k, v in self.pollination_memory.items():
                for land in lands_to_analyze:
                    landCell_pollinator = self.pollinators_processor.get_pollinator(k)
                    if landCell_pollinator.x == land.x and \
                            landCell_pollinator.y == land.y:

                        proportion = sum(v) / len(v)
                        if proportion <= 0.1:
                            landCell_pollinator.bag_pointer_declared = 100
                        elif proportion > 0.1 and proportion <= 0.3:
                            landCell_pollinator.bag_pointer_declared += 20
                        else:
                            landCell_pollinator.bag_pointer_declared = 0
    def emergency_counter_measure(self):

        if self.money_past and self.money_past[-1] < self.my_fees:
            self.emergency_counter+=1
        elif self.emergency_counter>=0:
            self.emergency_counter-=1

        if self.emergency_counter==3:
            for land in self.land_cells_owned:
                if land.bag_pointer_actual < 100:
                    land.bag_pointer_declared=0


    def take_observation_of_pollination(self):
        for land in self.land_cells_owned:
            if land.was_pollinated:
                self.pollination_memory[(land.x, land.y)].append(1)
            else:
                self.pollination_memory[(land.x, land.y)].append(0)

    def find_closest_lands_in_my_farm(self, current_point, closeness):
        closest_lands = list(
            filter(
                lambda c: distance.euclidean((c.x, c.y), current_point) <= closeness and distance.euclidean((c.x, c.y),
                                                                                                           current_point) != 0,
                self.land_cells_owned))

        return closest_lands

    def find_my_pollinators(self):
        return [land for land in self.land_cells_owned if land.bag_pointer_actual > 0]

    @staticmethod
    def average(lst):
        return sum(lst) / len(lst)

class PolinattorsProcessor:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.buffer_lands = []
        self.all_polinattors = set([
            (random.randint(0, GlobalParamsGame.MAX_CELLS_NUMER - 1),
             random.randint(0, GlobalParamsGame.MAX_CELLS_NUMER - 1))
            for _ in range(GlobalParamsAi.NUMBER_OF_RANDOM_POLLINATORS)])
        for polinattors in self.all_polinattors:
            self.get_pollinator(polinattors).is_pollinator = True
            self.get_pollinator(polinattors).bag_pointer_declared = 10
            self.get_pollinator(polinattors).bag_pointer_actual = 10
            self.grid.all_cells[polinattors].is_pollinator = True
            self.grid.all_cells[polinattors].bag_pointer_declared = 10
            self.grid.all_cells[polinattors].bag_pointer_actual = 10


    def set_active_pollinator(self, land):
        self.all_polinattors.add((land.x, land.y))

    def get_pollinator(self, cords):

        for x, y in self.grid.all_cells.keys():

            if x == cords[0] and y == cords[1]:
                return self.grid.all_cells[(x, y)]

    def find_closest_pollinator_to_land(self, current_point, closeness):
        closest_pollinators = list(
            filter(lambda c: distance.euclidean(c, current_point) < closeness, self.all_polinattors))
        distances = list(map(lambda c: distance.euclidean(c, current_point), closest_pollinators))

        return closest_pollinators, distances

    def clear_pollinators(self):
        to_delete = []
        for land in self.all_polinattors:
            if self.grid.all_cells[(land[0], land[1])].bag_pointer_actual == 0 or self.grid.all_cells[
                (land[0], land[1])].bag_pointer_actual == -1:
                to_delete.append(land)

        for x in to_delete:
            self.all_polinattors.remove(x)
        for l,v in self.grid.all_cells.items():
            if v.bag_pointer_actual>0:
                self.all_polinattors.add(l)
    def check_for_failed_pollinators_during_exploration(self):

        for land in self.all_polinattors:
            if self.grid.all_cells[(land[0], land[1])].bag_pointer_actual != 0:
                return False
        return True
    def logits(self,x):
        k=8.4
        xo=0.3
        return  1 / (1 + math.exp(-k*(x-xo)))

    # I assume that if you have bees you get pollinated
    def get_pollinated(self, land):

        polliator_distance_dict = {c: distance.euclidean(c, (land.x, land.y)) for c in
                                   self.all_polinattors if c != (land.x, land.y)}
        pollinators_within_certain_distance = dict(
            filter(lambda elem: self.distance_less_than(elem[1], 1.9), polliator_distance_dict.items()))
        weights = []
        #bag_sizes = []
        for other_pollinator,dist in pollinators_within_certain_distance.items():
            bag_size_actual = self.get_pollinator(other_pollinator).bag_pointer_actual
            result = self.sample_pollination(dist)
            weights.append(result*bag_size_actual/100)
            #bag_sizes.append(bag_size_actual)
        result_from_this_land = self.sample_pollination(0) #ToDo for learning purpose
        weights.append(result_from_this_land*land.bag_pointer_actual/100)
        sum_of_weights = sum(weights)
        randy_random = random.uniform(0, 1)
        probability = self.logits(sum_of_weights)
        return randy_random < probability

        # neighbourhood_actual_pollinators = [self.get_pollinator(k).bag_pointer_actual for k in
        #                                     pollinators_within_certain_distance.keys()]
        # cumulative_neighbour_polinattors = sum(neighbourhood_actual_pollinators)
        # if cumulative_neighbour_polinattors > 100:
        #     cumulative_neighbour_polinattors = 100

    @staticmethod
    def distance_less_than(number, less_than):
        return number <= less_than

    @staticmethod
    def sample_pollination(x,mode=0):


        # if x ==80:
        #     return True
        # else:
        #     return False
            # probablity = 0.7627864 + (-1.579016e-7 - 0.7627864)/(1 + (x/84.04566)**13.87343)
        # probablity =  1.52239 + (-0.001725851 - 1.52239)/(1 + (x/98.6658)**7.729825)
        # if probablity <0:
        #      probablity=0

        c=0.5
        a=1.1
        weight = c*math.exp(-x/a)
        # probablity =0
        # if x==100:
        #     probablity = 0.35
        # elif x==90:
        #     probablity=0.3
        # elif x==80:
        #     probablity=0.2
        # elif x==70:
        #     probablity=0.1
        # elif x==60:
        #     probablity = 0.05
        # elif x==50:
        #     probablity=0
        # elif x==40:
        #     probablity=0
        # else:
        #     probablity=0

        # #
        return weight



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
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def flush_memory(self):
        self.position = 0
        self.memory=random.sample(self.memory,16)
        #self.capacity=50
    def __len__(self):
        return len(self.memory)

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

class MADDPG_agent(Agent):
    def __init__(self, id, pos_x: int, pos_y: int, number_of_lands: int, agent_type):
        super().__init__(id, pos_x, pos_y, number_of_lands, agent_type)

        if id ==0:
            self.alpha=0.6
        elif id==1:
            self.alpha = 0.6
        elif id==2:
            self.alpha=0.6
        else:
            self.alpha=0.6
        self.trust_factor = round(random.uniform(0.0,1),2)
    def select_action(self, neural_net_output_number):
        a_bag_numbers = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        result = min(range(len(a_bag_numbers)), key=lambda i: abs(a_bag_numbers[i] - neural_net_output_number)) * 10
        return result
    #ToDo be wary of these action[0][0] it migth be tottally wroooong if 1 agent 1 land schema changes

    def get_random_action(self):
        a =[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        return random.choice(a)
    def make_a_decision(self, action,epsilon):

        decisions = []
        for i,land in enumerate(self.land_cells_owned):

            bad_size_declared = self.select_action(action[i])
            land.bag_pointer_declared = bad_size_declared
            land.bag_pointer_actual = bad_size_declared
            decisions.append(bad_size_declared)
        return decisions



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
        self.GAMMA = 0.1
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

        update = 250  # ToDo Was 100

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


class GlobalParamsGame:

    BLACK = (0, 0, 0)
    WHITE = (229, 255, 204)
    WINDOW_HEIGHT = 800
    WINDOW_WIDTH = 800
    BLOCKSIZE = 200
    SCREEN = pygame.display.set_mode((WINDOW_HEIGHT, WINDOW_WIDTH))
    MAX_CELLS_NUMER = int(WINDOW_HEIGHT / BLOCKSIZE)

class GlobalParamsAi:
    NUMBER_OF_AGENTS = 4
    NUMBER_OF_RANDOM_POLLINATORS = 1

class GlobalEconomyParams:
    LAND_UPCOST = 40
    STARTING_GOLD =10000
    MAXIMAL_INCOME = 100
class EconomyManager:
    def __init__(self, all_agents, pollinators_processor):
        self.upcost_price = GlobalEconomyParams.LAND_UPCOST
        self.all_agents = all_agents
        # self.starting_gold_multiplier = GlobalEconomyParams.STARTING_GOLD_MULTIPLIER
        self.land_fee = GlobalEconomyParams.LAND_UPCOST

        self.polinator_processor = pollinators_processor

    def deduce_land_fee(self):
        for a in self.all_agents:

            income = self.add_income(a)
            a.money += income
            a.income = income
            fee_to_pay = len(a.land_cells_owned) * self.land_fee
            if fee_to_pay > a.money:
                self.handle_dead_situation(a)

            else:
                a.money -= fee_to_pay

            a.last_income = income
    def handle_dead_situation(self,agent):
        agent.money = GlobalEconomyParams.STARTING_GOLD
        agent.is_dead=False
        agent.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        agent.color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    def add_income(self, agent):
        total_gross_income = 0

        for land in agent.land_cells_owned:

            if self.polinator_processor.get_pollinated(land):# ToDo Changed
                #print(f"land {(land.x, land.y)}")
            #if self.get_new_pollinated(land):
                this_land_income = (100 - land.bag_pointer_actual) / 100 * GlobalEconomyParams.MAXIMAL_INCOME
                total_gross_income += this_land_income
                land.was_pollinated = True
                land.last_income = this_land_income
            else:
                land.was_pollinated = False
                land.last_income = 0

        return total_gross_income
    def get_new_pollinated(self,land):
        if land.bag_pointer_actual>0  and land.bag_pointer_actual<100:
            return True
        return False
class AgentProcessor:
    def __init__(self, grid: Grid, pollinators_processor):
        self.all_agents = []
        counter_agent_id = 0
        self.grid = grid
        self.agents_pos_memory = set()
        # ToDo rewrite this shit
        self.seperate_land()
        # while len(self.all_agents) != GlobalParamsAi.NUMBER_OF_AGENTS:
        #     x, y = self.generate_two_random_numbers_that_does_not_hold_agent()
        #     cell = self.grid.get_cell((x, y))
        #
        #     new_agent = MADDPGAGENT(counter_agent_id, x,
        #                             y, 25, "RuleBasedAgent")
        #     new_agent.land_cells_owned.append(cell)
        #     self.set_ownership_of_land_piece(new_agent, cell)
        #     if new_agent not in self.all_agents:
        #         self.all_agents.append(new_agent)
        #     counter_agent_id += 1
        self.grid = grid

    def generate_two_random_numbers_that_does_not_hold_agent(self):
        while True:

            x = random.randint(0, GlobalParamsGame.MAX_CELLS_NUMER - 1)
            y = random.randint(0, GlobalParamsGame.MAX_CELLS_NUMER - 1)
            if (x, y) not in self.agents_pos_memory:
                self.agents_pos_memory.add((x, y))
                return x, y

    def seperate_land(self):

        # self.generate_agents_intial_positions()
        self.distribute_unoccupied_land()

    # def generate_agents_intial_positions(self):
    #     for agent in self.all_agents:
    #         for cell in self.grid.all_cells.values():
    #             if LandCell(agent.pos_x, agent.pos_y) == cell:
    #                 agent.land_cells_owned.append(cell)
    #                 self.set_ownership_of_land_piece(agent, cell)

    def set_ownership_of_land_piece(self, agent, cell):
        cell.set_owned(True)
        cell.set_owner(agent)

    def clear_empty_agents(self):
        for agent in self.all_agents:
            if len(agent.land_cells_owned) == 0:
                self.all_agents.remove(agent)

    def distribute_unoccupied_land(self):
        done = False
        counter = 0

        for i in range(GlobalParamsAi.NUMBER_OF_AGENTS):
            # if i == 0:
            #     agent =  MADDPGAGENT(i, 0,0, 25, "RuleBasedAgent")
            #     cells = [self.grid.all_cells[(0, 0)], self.grid.all_cells[(0, 1)], self.grid.all_cells[(0, 2)],
            #              self.grid.all_cells[(0, 3)],self.grid.all_cells[(1, 0)], self.grid.all_cells[(1, 1)], self.grid.all_cells[(1, 2)],
            #               self.grid.all_cells[(2, 0)],self.grid.all_cells[(3, 0)],self.grid.all_cells[(2, 1)], self.grid.all_cells[(2, 2)], self.grid.all_cells[(3, 1)],
            #               self.grid.all_cells[(3, 2)],self.grid.all_cells[(1, 3)], self.grid.all_cells[(2, 3)], self.grid.all_cells[(3, 3)]]
            #     for land in cells:
            #         agent.land_cells_owned.append(land)
            #         self.set_ownership_of_land_piece(agent, land)
            #     self.all_agents.append(agent)
            if i == 0:
                agent =  MADDPG_agent(i, 0,0, 25, "RuleBasedAgent")
                cells = [self.grid.all_cells[(0, 0)], self.grid.all_cells[(0, 1)], self.grid.all_cells[(0, 2)],
                         self.grid.all_cells[(0, 3)]]
                for land in cells:
                    agent.land_cells_owned.append(land)
                    self.set_ownership_of_land_piece(agent, land)
                self.all_agents.append(agent)
            if i==1:
                agent = MADDPG_agent(i, 1, 0, 25, "RuleBasedAgent")
                cells = [self.grid.all_cells[(1, 0)], self.grid.all_cells[(1, 1)], self.grid.all_cells[(1, 2)],
                         self.grid.all_cells[(2, 0)],self.grid.all_cells[(3, 0)]]
                for land in cells:
                    agent.land_cells_owned.append(land)
                    self.set_ownership_of_land_piece(agent, land)
                self.all_agents.append(agent)
            if i==2:
                agent = MADDPG_agent(i, 2, 1, 25, "RuleBasedAgent")
                cells = [self.grid.all_cells[(2, 1)], self.grid.all_cells[(2, 2)], self.grid.all_cells[(3, 1)],
                         self.grid.all_cells[(3, 2)]]
                for land in cells:
                    agent.land_cells_owned.append(land)
                    self.set_ownership_of_land_piece(agent, land)
                self.all_agents.append(agent)
            if i == 3:
                agent = MADDPG_agent(i, 1, 3, 25, "RuleBasedAgent")
                cells = [self.grid.all_cells[(1, 3)], self.grid.all_cells[(2, 3)], self.grid.all_cells[(3, 3)]]
                for land in cells:
                    agent.land_cells_owned.append(land)
                    self.set_ownership_of_land_piece(agent, land)
                self.all_agents.append(agent)

            counter += 1
        # for x in self.grid.all_cells.values():
        #     if not x.is_owned:
        #         exit("not all lands where distributed")


'''
OLD logic for seperating the land 

        while not done:
            done = True
            for x in self.grid.all_cells.values():
                if not x.is_owned:
                    done = False
            # Fixing the scenario for 4x4 grid and 4 agents
# agent = agent
                # if agent.predefined_number_of_lands > len(agent.land_cells_owned):
                #     counters = 0
                #     while counters < len(agent.land_cells_owned):
                # 
                #         curr_land = agent.land_cells_owned[counters]
                #         if curr_land.x + 1 < GlobalParamsGame.MAX_CELLS_NUMER:
                #             cell = self.grid.all_cells[(curr_land.x + 1, curr_land.y)]
                #             if not cell.is_owned:
                #                 agent.land_cells_owned.append(cell)
                #                 self.set_ownership_of_land_piece(agent, cell)
                #                 break
                #         if curr_land.x - 1 >= 0:
                #             cell = self.grid.all_cells[(curr_land.x - 1, curr_land.y)]
                #             if not cell.is_owned:
                #                 agent.land_cells_owned.append(cell)
                #                 self.set_ownership_of_land_piece(agent, cell)
                #                 break
                #         if curr_land.y + 1 < GlobalParamsGame.MAX_CELLS_NUMER:
                #             cell = self.grid.all_cells[(curr_land.x, curr_land.y + 1)]
                #             if not cell.is_owned:
                #                 agent.land_cells_owned.append(cell)
                #                 self.set_ownership_of_land_piece(agent, cell)
                #                 break
                #         if curr_land.y - 1 >= 0:
                #             cell = self.grid.all_cells[(curr_land.x, curr_land.y - 1)]
                #             if not cell.is_owned:
                #                 agent.land_cells_owned.append(cell)
                #                 self.set_ownership_of_land_piece(agent, cell)
                #                 break
                # 
                #         counters += 1


'''


class ActionProcessor:

    def __init__(self, all_agents, pollinator_processor: PolinattorsProcessor):
        self.all_agents = all_agents
        self.action_space = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.pollinators_processor = pollinator_processor
        self.epsilon = 1

    def all_agents_make_a_move(self, actions):

        counter = 0
        #print(f"epsilon is {self.epsilon}")
        lands_reward = []
        for i, agent in enumerate(self.all_agents):
            #print(f"these are actions {actions}")
            lands_reward.append(agent.make_a_decision(actions[i], self.epsilon))

            counter += 1
        return lands_reward

    def make_random_action(self, agent):
        for land in agent.land_cells_owned:
            action_space_random = random.randint(0, len(self.action_space) - 1)
            land.bag_pointer_declared = self.action_space[action_space_random]
            self.pollinators_processor.buffer_lands.append(land)


def measure_sum_of_distances_between_matrices(a, b):
    distances = []
    for x, y in zip(a, b):
        for z, c in zip(x, y):
            distances.append(abs(round(z.item(),1) - round(c.item(),1)))

    return sum(distances)

def make_random_action(all_agents):
    action = []
    for agent in all_agents:
        agent_actions = [th.tensor(random.choice([random.uniform(0, 102)]) / 100) for _ in
                         range(len(agent.land_cells_owned))]

        action.append(agent_actions)
    return action
class EnvironmentalManager:
    def __init__(self, pollinators_processor: PolinattorsProcessor):
        self.pollinators_processor = pollinators_processor
        self.all_polinattors = self.pollinators_processor.all_polinattors

    def calculate_euclidian_distance(self, a, b):
        return np.linalg.norm(a - b)

    def process_declared_lands(self):

        lands_to_process = [land for land in self.pollinators_processor.grid.all_cells.values()  ]
        for land in lands_to_process:

            if land.bag_pointer_actual != -1:
               # land.bag_pointer_actual = land.bag_pointer_declared
               # if land.bag_pointer_actual > 0:
               #     self.pollinators_processor.all_polinattors.add((land.x, land.y))
                #turned off for NOW
                current_point = (land.x, land.y)
                closest_pollinators, distances = self.pollinators_processor.find_closest_pollinator_to_land(current_point,
                                                                                                            3)
                if closest_pollinators:
                    self.calculate_environmental_bag(zip(closest_pollinators, distances),
                                                     self.pollinators_processor.get_pollinator(current_point))



    def calculate_environmental_bag(self, closest_pollinators_with_distance, land):
        #ToDo enable spreading
        for closests_pollinator, euclidian_distance in closest_pollinators_with_distance:
            probability = math.exp(-1 * euclidian_distance)
            #land.bag_pointer_actual = land.bag_pointer_declared



            self.sample_pollinator_to_create_new_one(land, probability,
                                                     self.pollinators_processor.get_pollinator(closests_pollinator))

    def sample_pollinator_to_create_new_one(self, land, probability, pollinator):
        randy_random = random.uniform(0, 1)
        actual_bag = land.bag_pointer_actual
        declared_bag = land.bag_pointer_declared
        if declared_bag > actual_bag and actual_bag < 100:
            land.bag_pointer_actual = land.bag_pointer_declared #ToDo for debugging
            # if randy_random <= probability:
            #
            #     result =0
            #
            #     probability_how_much_we_get = 10 * (1 + 0.03) ** pollinator.bag_pointer_actual / 100
            #     randy_random = uniform(0, 1)
            #     if randy_random >0 and randy_random <0.7:
            #         result +=10
            #     elif randy_random>0.7 and randy_random <0.8 and actual_bag <=80:
            #         result += 20
            #     elif randy_random>0.8 and randy_random <0.85 and actual_bag <=70:
            #         result += 30
            #     else:
            #         result += 10
            #     if actual_bag + result <= declared_bag:
            #         land.bag_pointer_actual += result
            #     else:
            #         land.bag_pointer_actual = declared_bag

        elif declared_bag < actual_bag:
            land.bag_pointer_actual = declared_bag
        if actual_bag > 0:
            self.pollinators_processor.all_polinattors.add((land.x, land.y))

class gymDriver(gym.Env):
    def __init__(self):
        self.grid = Grid()
        self.polinattor_processor = PolinattorsProcessor(grid=self.grid)
        self.n_agents = GlobalParamsAi.NUMBER_OF_AGENTS
        self.agent_processor = AgentProcessor(grid=self.grid, pollinators_processor=self.polinattor_processor)
        #self.agent_processor.seperate_land()
        self.agent_processor.clear_empty_agents()
        self.action_processor = ActionProcessor(all_agents=self.agent_processor.all_agents,
                                                pollinator_processor=self.polinattor_processor)
        self.observation_space = spaces.Box(0, 1, shape=[1])
        self.clockobject = pygame.time.Clock()
        self.action_space = spaces.Box(-1, 1, shape=[4])
        self.environmental_manager = EnvironmentalManager(self.polinattor_processor)
        #self.environmental_manager.process_declared_lands()
        self.economy_manager = EconomyManager(self.agent_processor.all_agents, self.polinattor_processor)
        self.counter = 0
        self.index = 0

    def reset(self):

        self.polinattor_processor = PolinattorsProcessor(grid=self.grid)

        self.n_agents = len(self.agent_processor.all_agents)

        return self._create_observation([0, 0, 0, 0])

    def _get_reward_for_gov(self):
        agent_lands = np.zeros(4)
        for j, agent in enumerate(self.agent_processor.all_agents):

            cumulative_lands = 0
            for land in agent.land_cells_owned:
                cumulative_lands += land.bag_pointer_actual / 100

            agent_lands[j] = cumulative_lands / len(agent.land_cells_owned)
        return agent_lands

    def get_global_pollinators_reward(self):
        pollinators_reward = 0

        for land in self.grid.all_cells.values():
            if land.was_pollinated:
                pollinators_reward += (land.bag_pointer_actual / 100) / len(self.grid.all_cells)
        # print(f"Global pollinators reward {pollinators_reward}")
        return pollinators_reward

    def _get_reward(self, incentive, render):
        agents_rewards = []
        global_pollinators_reward = self.get_global_pollinators_reward()
        reward_without_incentive =[]
        for j, agent in enumerate(self.agent_processor.all_agents):

            cumulative_reward = 0

            for i, land in enumerate(agent.land_cells_owned):
                single_reward = land.last_income / 100 - GlobalEconomyParams.LAND_UPCOST / 100 #ToDo no minus anymore no upcost

                cumulative_reward += single_reward

            money_reward = cumulative_reward / len(agent.land_cells_owned)
            #          0-1   0.9  * 0.9   + 0.1 * 0.35
            reward_internal = agent.alpha * (money_reward) + (1 - agent.alpha) * global_pollinators_reward/2 #ToDo was /2
            final_reward = reward_internal+incentive[j] # incentive 0-1 you can divide by 2 everything#ToDo just money reword for debugging
            # agent.money += incentive[j] * 1000
            if render:
                print(
                    f"AGENT:{agent.id} his alpha is {agent.alpha} money :{money_reward}"
                    f" env :{round(global_pollinators_reward/2, 2)}"
                    f" incentive : {incentive[j] }"
                    f" final_reward : {final_reward} ")
            if final_reward<0:
                final_reward=0
            agents_rewards.append(final_reward)
            reward_without_incentive.append(reward_internal)
        return agents_rewards,reward_without_incentive

    def _create_observation(self, incentive):
        board_size = int(GlobalParamsGame.WINDOW_HEIGHT / GlobalParamsGame.BLOCKSIZE)
        land_per_agent_obs = []
        # ToDo HERE
        for j, agent in enumerate(self.agent_processor.all_agents):
            single_agent_obs = []
            empty_obs_local_declared = np.zeros((board_size, board_size))
            empty_obs_local_actual = np.zeros((board_size, board_size))
            incentive_np = np.zeros((board_size, board_size))

            for land in agent.land_cells_owned:
                empty_obs_local_declared[land.x, land.y] = land.bag_pointer_declared / 100
                empty_obs_local_actual[land.x, land.y] = land.bag_pointer_actual / 100

                incentive_np[land.x, land.y] = incentive[j]

            empty_obs_local_declared=np.rot90(empty_obs_local_declared)
            empty_obs_local_actual=np.rot90(empty_obs_local_actual)
            incentive_np=np.rot90(incentive_np)

            single_agent_obs.append(
                np.array([empty_obs_local_actual,                       #ToDo deleting incentive
                          incentive_np]))  # '''*incentive[j]'''')  # ToDo Deleting actual bag for debugging purposes
            # single_agent_obs.append(
            #         np.array([empty_obs_local_actual]))
            land_per_agent_obs.append(single_agent_obs)
        empty_obs_declared, empty_obs_actual, incentive_global = self.get_global_state_without_position(board_size,
                                                                                                        incentive)  # ToDo moved from THERE TO HERE
        empty_obs_actual = np.rot90(empty_obs_actual)
        incentive_global = np.rot90(incentive_global)
        # land_per_agent_obs.append(np.array([empty_obs_declared]))
        # print("bla bla {}".format(incentive_global))
        global_obs = np.array([empty_obs_actual, incentive_global])
        return land_per_agent_obs, global_obs

    def get_global_state_without_position(self, board_size, incentive):
        empty_obs_declared = np.zeros((board_size, board_size))
        empty_obs_actual = np.zeros((board_size, board_size))
        incentive_np_global = np.zeros((board_size, board_size))
        for j, agent in enumerate(self.agent_processor.all_agents):
            for land in agent.land_cells_owned:
                empty_obs_declared[land.x, land.y] = land.bag_pointer_declared / 100
                empty_obs_actual[land.x, land.y] = land.bag_pointer_actual / 100
                incentive_np_global[land.x, land.y] = incentive[j]
        return empty_obs_declared, empty_obs_actual, incentive_np_global

    def render(self, mode='human'):
        self.grid.drawGrid()
        process_pygame_events()
        pygame.display.update()

    def step(self, action=None, randy_random=None, incentive=None, render=False):
        if isinstance(action, np.ndarray):
            action = [action.tolist()]
        if randy_random:
            for i,output in enumerate(action):
                printer =[]
                for x in output:
                    printer.append(int(round(x.item(),2)*100))
                print(f"agent {i} state is {printer}           average: {sum(printer)/len(printer)*100}")
            print(f"AGENTS WILL DO THE FOLLOWING ACTION {action}")
        self.action_processor.all_agents_make_a_move(action)
        self.polinattor_processor.clear_pollinators()
        self.environmental_manager.process_declared_lands()
        self.economy_manager.deduce_land_fee()

        done = False
        if self.polinattor_processor.check_for_failed_pollinators_during_exploration():
            self.reset()
            done = True

        if render:
            self.render()
        reward,reward_without_incentive = self._get_reward(incentive, render)

        #

        observation = self._create_observation(incentive)
        # print(f"rewards per agent {reward}")
        #self.create_actions_channels(action, observation, reward)
        return observation, (reward,reward_without_incentive), done, None

    def create_actions_channels(self, action, img, reward):
        import pickle
        # plt.imshow(img,cmap='gray',vmax=1,vmin=0)
        # plt.show()
        (obs_, global_state_) = img
        action_np = np.zeros(np.array(img).shape)
        action = self.flatten(action)
        action = th.from_numpy(np.reshape(action, (4, 4))).float().unsqueeze(0).unsqueeze(0) #for now no concrete mapping to agents or lands GAN will have to  figure it out
        alone = np.pad(np.array([[1, 2], [2, 3]]), 2, self.pad_with, padder=0)

        # action = torch.ones_like(torch.from_numpy(img)).repeat(4, 1, 1) * torch.from_numpy(action) \
        #     .unsqueeze(1) \
        #     .unsqueeze(2)

        state_action = th.cat([th.from_numpy(global_state_).float(),action.squeeze(0)],dim=0).unsqueeze(0)

        if self.index > 0:
            with open(f"C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\train\\Sa_images\\state_s_{self.index - 1}.pickle", 'wb') as handle:
                state_without_action = th.from_numpy(global_state_).float().unsqueeze(0)
                #future_state = torch.from_numpy(state_without_action).unsqueeze(0)
                pickle.dump(state_without_action, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\train\\S_images\\/state_s_{self.index}.pickle', 'wb') as handle:
            pickle.dump(state_action, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.index += 1

        self.previous_gan_action = action

    def flatten(self, t):
        return [item for sublist in t for item in sublist]

    def pad_with(self,vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 10)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
import sys
def process_pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()


def make_random_action(all_agents):
    action = []
    import torch as th
    import random
    for agent in all_agents:
        agent_actions = [th.tensor(random.choice([random.uniform(0, 102)]) / 100) for _ in
                         range(len(agent.land_cells_owned))]

        action.append(agent_actions)
    return action
class goverment_agent():
    def __init__(self, agent_networks, q_value_networks, all_agents):
        board_size = int(GlobalParamsGame.WINDOW_HEIGHT / GlobalParamsGame.BLOCKSIZE)

        self.target = [0.35, 0.35, 0.35, 0.35]  # [round(x,1) for x in self.target]
        self.budget = 0
        self.agent_networks = agent_networks
        self.q_value_networks = q_value_networks
        self.all_agents = all_agents
        self.big_incentive = 0
        self.medium_incentive = 0
        self.low_incentive = 0
        self.interpreted_agent = None
        self.interpreted_obs = None
        self.counter_networks = 0
        self.decisions = {}
        self.new_state = None

    def set_this_year_budget(self, new_budget):
        self.budget = new_budget

    def distribute_incetive(self):
        incentive = []
        for j, agent in enumerate(self.all_agents):
            all_pollinators = 0
            for i, land in enumerate(agent.land_cells_owned):
                all_pollinators += land.bag_pointer_actual / 100
            if all_pollinators / len(agent.land_cells_owned) >= 0.8:
                incetive = -1
            elif all_pollinators / len(agent.land_cells_owned) >= 0.5:
                incetive = -0.6
            else:
                incetive = 0
            # incetive = random.uniform(-1,1)
            # print(f"agent {j} got {all_pollinators/len(agent.land_cells_owned)} and got this incentive {incetive}")
            incentive.append(incetive)
        return incentive

    def distribute_incentive_2(self):
        incentive = []
        for j, agent in enumerate(self.all_agents):
            all_pollinators = 0
            for i, land in enumerate(agent.land_cells_owned):
                all_pollinators += land.bag_pointer_actual / 100
            if all_pollinators / len(agent.land_cells_owned) >= 0.8:
                incetive = 0.6
            elif all_pollinators / len(agent.land_cells_owned) >= 0.5:
                incetive = 0.1
            else:
                incetive = -0.25
                # print(f"agent {j} got {all_pollinators/len(agent.land_cells_owned)} and got this incentive {incetive}")
            incentive.append(incetive)
        return incentive

    def distribute_incentive_3(self):
        incentive = []
        for j, agent in enumerate(self.all_agents):
            all_pollinators = 0
            final_incentive = 0
            for i, land in enumerate(agent.land_cells_owned):

                if land.bag_pointer_actual == 100:
                    final_incentive -= 0.25
                if i == 0 and land.bag_pointer_actual == 50:
                    final_incentive += 0.25
                if i == 1 and land.bag_pointer_actual == 20:
                    final_incentive += 0.25
                if i == 2 and land.bag_pointer_actual == 60:
                    final_incentive += 0.25

                # print(f"agent {j} got {all_pollinators/len(agent.land_cells_owned)} and got this incentive {incetive}")
            incentive.append(final_incentive)
        return incentive

    def distribute_incentive_4(self):
        incentive = []
        for j, agent in enumerate(self.all_agents):
            final_incentive = 0
            for i, land in enumerate(agent.land_cells_owned):

                if land.bag_pointer_actual != 100 and land.bag_pointer_actual != 0:
                    final_incentive += 0.2
                else:
                    final_incentive -= 0.1

                # print(f"agent {j} got {all_pollinators/len(agent.land_cells_owned)} and got this incentive {incetive}")
            incentive.append(final_incentive)
        return incentive

    def optimise_incentives(self, obs, agents, agent_networks, critics):

        import optuna
        self.agent_networks = agent_networks
        self.critics = critics
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # If you want agent observations you need their personal one not global oen which was an error
        final_incentive = []
        new_observation = []
        for i, a in enumerate(agents):
            self.counter_networks = i
            self.debugging_obs = np.array(obs)[self.counter_networks][0]
            self.interpreted_obs = np.array(obs)[self.counter_networks][0][0]
            self.interpreted_agent = a
            study = optuna.create_study(direction='minimize')
            study.optimize(self.objective, n_trials=500)
            best_incentive = study.best_params['x']
            final_incentive.append(best_incentive)
            new_obs_incentive = self.get_agents_land_positions(a,self.interpreted_obs,best_incentive)
            new_observation.append(np.stack((self.interpreted_obs, new_obs_incentive)))
            # print(f"before : \n {self.debugging_obs[1]} \n  after: \n {self.new_state[0][1]}")

            print(
                f"here are decisions for agent {self.counter_networks} that would result {self.agent_networks[self.counter_networks](th.from_numpy(self.debugging_obs).unsqueeze(0).unsqueeze(0).float().cuda()).mean().item()} here are optimised {self.decisions[self.counter_networks].mean()}")

        return final_incentive,np.stack((np.expand_dims(x, axis=0) for x in new_observation)).tolist()

    # make decision somehow impact the critic
    def objective(self, trial):
        x = trial.suggest_float('x', -1, 1)
        multiplier = 1
        new_incentive = self.get_agents_land_positions(self.interpreted_agent, self.interpreted_obs, x * multiplier)
        self.new_state = th.cat([th.from_numpy(self.interpreted_obs.copy()).float().unsqueeze(0),
                                 th.from_numpy(new_incentive.copy()).float().unsqueeze(0)], dim=0).unsqueeze(0)
        self.decisions[self.counter_networks] = self.agent_networks[self.counter_networks](
            self.new_state.cuda()).squeeze().data
        # self.critics[self.counter_networks](new_state.cuda(),decisions)
        return abs(self.target[self.counter_networks] - self.decisions[self.counter_networks].mean().item())

    def get_agents_land_positions(self, agent, other_array, incentive):
        incentive_arr = np.zeros_like(other_array)
        for land in agent.land_cells_owned:
            incentive_arr[land.x, land.y] = incentive
        incentive_arr = np.rot90(incentive_arr)
        # print(f"\n\n\n My incentive arr is agent : {self.counter_networks} \n obs {incentive_arr}\n\n\n\n")
        return incentive_arr

e_render = False

food_reward = 10.
poison_reward = -1.
encounter_reward = 0.01
n_coop = 2
world = gymDriver()

reward_record = []

np.random.seed(1234)
th.manual_seed(1234)
world.seed(1234)

n_states = 213
n_actions = 1
capacity = 9000 #ToDo was 500
batch_size = 16
n_episode = 500
max_steps = 100000000
episodes_before_train = 10
epsilon = 1
win = None
param = None
obs = world.reset()
# initialization of the objeects is after reset that's why it's here
dims = np.array(obs)
n_agents = world.n_agents
buffer_action = None
buffer_counter = 0
cum_reward = []
import matplotlib.pyplot as plt

(obs, global_state) = world.reset()
worlds_all_agents = world.agent_processor.all_agents
maddpg = MADDPG(world.n_agents, 12, n_actions, batch_size, capacity,
                episodes_before_train, worlds_all_agents,False)
maddpg_fake = MADDPG(world.n_agents, 12, n_actions, batch_size, capacity,
                episodes_before_train, worlds_all_agents,False)
FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor


meta = goverment_agent(None, None, worlds_all_agents)
counters = 0
incentive = [0,0,0,0]
warnings.filterwarnings("ignore")
flag_for_incentive = False
flag_for_real_action = False
incentive_reward = []
before_incentive_reward = []
difference_in_reward = []
reward =[]
incentive_tracker=[]
average = []
baseline = []
baseline_spending = []
reward_without_incentive_baseline = []

fake_real_difference = []
def handle_exploration():
    global epsilon
    if maddpg.episode_done >= batch_size:
        # if epsilon > 0.6 :
        #     epsilon -= 1e-3
        # elif epsilon <  0.6 and epsilon > 0.3:
        #     epsilon -= 1e-4
        # el
        #epsilon -= 1e-4  # 0.0001
        #ToDo static epsilon going 0
        if round(epsilon * 1000) % 211 == 0 and epsilon == 0.3:
            plt.plot(cum_reward)
            plt.show()
        if epsilon > 0.4:
            epsilon -= 1e-4 # 0.0001
        else:
            epsilon = 0.4


def cirriculum_learning(worlds_all_agents,action,epsilon,counters):

    if epsilon <= 0.8 and counters <= 100:
        incentive = meta.distribute_incetive()
    if counters >= 100 and counters < 1200:
        incentive = meta.distribute_incentive_2()
    if counters >= 1200 and counters < 1800:
        incentive = []
        for i, a in enumerate(worlds_all_agents):
            average_action = sum([x.item() for x in action[i]]) / len(action[i])
            print(f" Average :  {average_action} !! ")
            incentive.append(1 - (abs(meta.target[i] - average_action)))
            baseline.append(meta.target[i] - average_action)
    if counters >= 1800 and counters < 2000:
        incentive = meta.distribute_incentive_4()
    else:
        incentive=[0,0,0,0]
    return incentive

if __name__ == "__main__":
    for i_episode in range(n_episode):

        # obs = np.stack(obs)
        # if isinstance(obs, np.ndarray):
        #     obs = th.from_numpy(obs).float()
        total_reward = 0.0
        rr = np.zeros((n_agents,))
        for t in range(max_steps):
            # render every 100 episodes to speed up training
            # if i_episode % 100 == 0 and e_render:
            world.clockobject.tick(99)
            # world.render()

            # obs = obs.type(FloatTensor)
            randy_random = random.uniform(0, 1)

            # if epsilon<0.9 and epsilon>0.3:
            #     incentive = meta.distribute_incetive()
            # if epsilon<0.5 and counters<=19700:
            #     incentive = meta.distribute_incentive_4()
            #incentive = meta.optimise_incentives(obs, worlds_all_agents, maddpg.actors, maddpg.critics)
            #incentive = meta.optimise_incentives(obs, worlds_all_agents, maddpg.actors, maddpg.critics)
            #incentive = [0, 0, 0, 0]



            # print(f" action for the game {agents_actions}")
            # action = th.from_numpy(action_np)
            # obs_, reward, done, _ = world.step(action, randy_random_2)
            print("Epsilon is {}".format(epsilon))
            if epsilon <= 0.4:
                counters += 1
                # maddpg.save_weights_of_networks("before_curriculum")
                print("The COUNTER IS {}".format(counters))




            if counters>=3300:
                import pickle

                with open('important_pickles/fake_planned_20_80_personal_reward45.pkl', 'wb') as f:
                    pickle.dump(difference_in_reward, f)
                with open('important_pickles/fake_planned_20_80_incetives_given45.pkl', 'wb') as f:
                    pickle.dump(incentive_tracker, f)
                with open('important_pickles/fake_planned_20_80_distance_from_target45.pkl', 'wb') as f:
                    pickle.dump(average, f)
                with open('important_pickles/fake_planned_20_80_distance_from_target_baseline45.pkl', 'wb') as f:
                    pickle.dump(baseline, f)
                with open('important_pickles/fake_planned_20_80_spending45.pkl', 'wb') as f:
                    pickle.dump(baseline_spending, f)
                with open('important_pickles/fake_planned_20_80_baseline_reward_without_incentive45.pkl', 'wb') as f:
                    pickle.dump(reward_without_incentive_baseline, f)
                exit(1)


            if counters>=2100:
                import pickle
                with open('important_pickles/distance_real_fake.pkl', 'wb') as f:
                    pickle.dump(fake_real_difference, f)
                #incentive,new_ob =meta.optimise_incentives(obs,worlds_all_agents,maddpg.actors,maddpg.critics) #ToDo now Fake one is optimised
                incentive, new_ob = meta.optimise_incentives(obs, worlds_all_agents, maddpg_fake.actors, maddpg_fake.critics)
                obs = new_ob
            print(f"Optimiser gave the following incentive : {incentive}")
            if epsilon > randy_random:
                action = maddpg.select_action(obs, worlds_all_agents)
                action_fake = maddpg_fake.select_action(obs, worlds_all_agents) #ToDo fake maddpg
                flag_for_real_action = False
                agents_actions = ''.join(f'''agent {i} made actions {a} \n ''' for i, a in enumerate(action))
                agents_actions_fake = ''.join(f''' FAKE agent {i} made actions {a} \n ''' for i, a in enumerate(action_fake)) #ToDo Fake agent
                print(agents_actions)
                print(agents_actions_fake) #ToDo Fake agent
                action = make_random_action(worlds_all_agents)
                distance = measure_sum_of_distances_between_matrices(action,action_fake)
                fake_real_difference.append(distance)
            else:
                # maddpg.batch_size= 32
                flag_for_real_action = True
                action = maddpg.select_action(obs, worlds_all_agents)
                action_fake = maddpg_fake.select_action(obs, worlds_all_agents)  # ToDo fake maddpg
                distance = measure_sum_of_distances_between_matrices(action, action_fake)
                fake_real_difference.append(distance)

            print(f"DISTANCE BETWEEN FAKE AND REAL IS {distance}")
            if epsilon <= 0.8 and counters <= 100:
                incentive = meta.distribute_incetive()
            if counters >= 100 and counters < 1000:
                incentive = meta.distribute_incentive_2()
            if counters >= 1000 and counters < 2100:
                incentive = []
                spends = []
                avg = []

                for i, a in enumerate(worlds_all_agents):
                    average_action = sum([x.item() for x in action[i]]) / len(action[i])
                    print(f" Average :  {average_action} !! ")
                    spending = 1 - (abs(meta.target[i] - average_action))
                    incentive.append(spending)
                    avg.append(meta.target[i] - average_action)
                    spends.append(spending)
                if flag_for_real_action:
                    reward_without_incentive_baseline.append(sum(reward_without_incentive))
                    baseline.append(avg)
                    baseline_spending.append(sum(spends))


            # if counters >= 1000 and counters < 2000:
            #     incentive = meta.distribute_incentive_4()


                # average_action = sum([x.item() for x in action[0]]) / len(action[0])
                # print(f" Average :  {average_action} !! ")
                # incentive = [1-(abs(0.15-average_action))]

            (obs_, global_state_), (reward,reward_without_incentive), done, _ = world.step(action, flag_for_real_action, incentive, True)
            print(f" I had the following incentive {incentive} ")

            if counters>=2100 and flag_for_real_action:
                #ToDo Single agent tracker
                # incentive_tracker.append(incentive)
                # difference_in_reward.append(reward_without_incentive)
                # averages = sum([x.item() for x in action[0]]) / len(action[0])
                # average.append(averages)
                incentive_tracker.append(sum(incentive))
                difference_in_reward.append(sum(reward_without_incentive))
                averages = []
                for a in range(len(maddpg.actors)):
                    averages.append((meta.target[a] - sum([x.item() for x in action[a]]) / len(action[a])))
                average.append(averages)
            # if not flag_for_incentive and flag_for_real_action  :
            #     before_incentive_reward.append(sum(reward)/4)
            #     difference_in_reward.append(sum(reward_without_incentive)/4)
            # elif flag_for_incentive :
            #     incentive_reward.append(sum(reward)/4)
            #     difference_in_reward.append(sum(reward_without_incentive)/4)
            # if counters>=9000:
            #
            #     exit(1)
            # observation, reward, done, info = world.step(action, None, action_incentive)
            # model.forward(action_incentive, obs_)
            # print("reward: {}\nnew state: {}".format(reward, np.round(obs_, 2)))
            # done = False
            # model.learn(10000, progress_bar=True)
            # action = model.best_action()
            # observation, reward, done, info = env.step(action)
            # model.forward(action, observation)
            # reward = th.FloatTensor(reward).type(FloatTensor)
            # obs_ = np.stack(obs_)
            # obs_ = th.from_numpy(obs_).float()
            # if t != max_steps - 1:
            #     next_obs = obs_
            # else:
            #     next_obs = None
            next_obs = obs_
            next_global_state = global_state_
            # total_reward += reward.sum()
            # rr += reward.cpu()
            # obs_ = np.concatenate([np.expand_dims(obs[2], 0), obs_], 0)
            #ToDo switch around obs -> old version obs,obs_ , new obs_, obs
            maddpg.memory.push(obs, action, obs_, reward, global_state, global_state_)
            maddpg_fake.memory.push(obs, action, obs_, reward, global_state, global_state_) #ToDo fake maddpg
            obs = next_obs
            global_state = next_global_state
            cum_reward.append(sum(reward))
            c_loss, a_loss = maddpg.update_policy(worlds_all_agents, epsilon)
            c_loss, a_loss = maddpg_fake.update_policy(worlds_all_agents, epsilon)#ToDo fake maddpg
            maddpg.episode_done += 1
            maddpg_fake.episode_done += 1

            handle_exploration()
            print(f"actual epsilon {epsilon}")
        maddpg.episode_done += 1
        # print('Episode: %d, reward = %f' % (i_episode, total_reward))
        # reward_record.append(total_reward)

        if maddpg.episode_done == maddpg.episodes_before_train:
            print('training now begins...')
            print('MADDPG on WaterWorld\n' +
                  'scale_reward=%f\n' % scale_reward +
                  'agent=%d' % n_agents +
                  ', coop=%d' % n_coop +
                  ' \nlr=0.001, 0.0001, sensor_range=0.3\n' +
                  'food=%f, poison=%f, encounter=%f' % (
                      food_reward,
                      poison_reward,
                      encounter_reward))

    world.close()

