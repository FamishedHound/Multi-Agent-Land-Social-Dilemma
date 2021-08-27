from random import uniform, random, choice,randint

import gym
import sys
from time import sleep

import pygame

from game.economy.EconomyManager import EconomyManager
from game.logic.ActionProcessor import ActionProcessor
from game.logic.AgentProcessor import AgentProcessor
from game.logic.EnvironmentalManager import EnvironmentalManager
from game.logic.PolinattorsProcessor import PolinattorsProcessor
from game.visuals.Grid import Grid
from game import GlobalParamsGame
import numpy as np
import torch


class gymDriver(gym.Env):
    def __init__(self):
        self.grid = Grid()
        self.polinattor_processor = PolinattorsProcessor(grid=self.grid)
        self.n_agents = GlobalParamsGame.GlobalParamsAi.NUMBER_OF_AGENTS
        self.agent_processor = AgentProcessor(grid=self.grid, pollinators_processor=self.polinattor_processor)
        self.agent_processor.seperate_land()
        self.agent_processor.clear_empty_agents()
        self.action_processor = ActionProcessor(all_agents=self.agent_processor.all_agents,
                                                pollinator_processor=self.polinattor_processor)
        self.clockobject = pygame.time.Clock()

        self.environmental_manager = EnvironmentalManager(self.polinattor_processor)
        self.environmental_manager.process_declared_lands()
        self.economy_manager = EconomyManager(self.agent_processor.all_agents, self.polinattor_processor)
        self.counter = 0
    def reset(self):


        self.polinattor_processor = PolinattorsProcessor(grid=self.grid)

        self.n_agents = len(self.agent_processor.all_agents)

        return self._create_observation()

    # def _get_reward(self):
    #     agents_rewards = []
    #     for i, agent in enumerate(self.agent_processor.all_agents):
    #         agent_land_rewards = []
    #         reward = 0
    #         for land in agent.land_cells_owned:
    #             # reward = land.last_income/100
    #             if land.bag_pointer_declared == 50:
    #                 reward = -1
    #             elif land.bag_pointer_declared == 100:
    #                 reward =-1
    #             elif land.bag_pointer_declared == 20 :
    #
    #                 reward = 1
    #             else:
    #                 reward =-1
    #             agent_land_rewards.append(reward)
    #         agents_rewards.append(agent_land_rewards)
    #     return np.array(agents_rewards)
    def get_global_pollinators_reward(self):
        pollinators_reward = 0
        for i, agent in enumerate(self.agent_processor.all_agents):


            for land in agent.land_cells_owned:
                if land.bag_pointer_actual ==100:
                    pollinators_reward+=0.1
        print(f"Global pollinators reward {pollinators_reward}")
        return pollinators_reward
    def _get_reward(self):
        agents_rewards = []
        global_pollinators_reward =  self.get_global_pollinators_reward()

        for i, agent in enumerate(self.agent_processor.all_agents):
            agent_land_rewards = []
            cumulative_reward = 0
            x = [x.bag_pointer_actual for x in agent.land_cells_owned]
            # print("state being {}".format(x))
            # if x==[90,90,90,90]:
            #     reward = 1
            # elif x ==[100,100,100,100]:
            #     reward = - 1
            # else:
            #     reward = 0
            for i,land in enumerate(agent.land_cells_owned):

                single_reward = land.last_income/100
                if single_reward==0:
                    single_reward=-0.5
                cumulative_reward += single_reward
                # if reward ==0 :
                #     reward=0.2
                # if land.was_pollinated:
                #     reward=1
                # else:
                #     reward=0
                # if land.bag_pointer_declared == 60 :
                #     reward = 1
                # elif land.bag_pointer_declared > 60:
                #     reward = 60/land.bag_pointer_declared
                # else:
                #     reward = land.bag_pointer_declared/60
                #
            #agent_land_rewards.append(cumulative_reward)
            #land_rewards = [cumulative_reward for _ in range (len(agent.land_cells_owned))]


            agents_rewards.append(cumulative_reward/len(agent.land_cells_owned))
        return agents_rewards

    def _create_observation(self):
        board_size = int(GlobalParamsGame.GlobalParamsGame.WINDOW_HEIGHT / GlobalParamsGame.GlobalParamsGame.BLOCKSIZE)
        land_per_agent_obs = []
        empty_obs_actual, empty_obs_declared = self.get_global_state_without_position(board_size)
        for agent in self.agent_processor.all_agents:
            single_agent_obs = []

            # for land in agent.land_cells_owned:
            #     empty_obs_position = np.zeros((board_size, board_size))
            #     empty_obs_position[land.x, land.y] = 1


            single_agent_obs.append( np.array([empty_obs_declared]))#ToDo Deleting actual bag for debugging purposes
            land_per_agent_obs.append(single_agent_obs)
        #land_per_agent_obs.append(np.array([empty_obs_declared]))
        return land_per_agent_obs

    def get_global_state_without_position(self, board_size):
        empty_obs_declared = np.zeros((board_size, board_size))
        empty_obs_actual = np.zeros((board_size, board_size))
        for agent in self.agent_processor.all_agents:
            for land in agent.land_cells_owned:
                empty_obs_declared[land.x, land.y] = land.bag_pointer_declared / 100
                empty_obs_actual[land.x, land.y] = land.bag_pointer_actual / 100
        return empty_obs_actual, empty_obs_declared

    def render(self, mode='human'):
        self.grid.drawGrid()
        process_pygame_events()
        pygame.display.update()

    def step(self, action=None,randy_random=None):



        lands_picked = self.action_processor.all_agents_make_a_move(action)
        self.polinattor_processor.clear_pollinators()
        self.environmental_manager.process_declared_lands()
        self.economy_manager.deduce_land_fee()

        done = False
        if self.polinattor_processor.check_for_failed_pollinators_during_exploration():
            self.reset()
            done = True

        # if randy_random==0:
        #
        #     reward = [[1]]
        # else:
        #
        # reward = []
        # for agent_lands in lands_picked:
        #     reward_agent_temp = []
        #     for agent_land in agent_lands:
        #         if agent_land==70:
        #             reward_agent_temp.append(1)
        #         else:
        #             reward_agent_temp.append(0)
        #     reward.append(reward_agent_temp)



        # if np.array(self._create_observation())[0][0][1][0].item()==0.2:
        #     reward = [[1]]
        # else:
        #     reward = [[0]]
        self.render()
        reward = self._get_reward()
        if done:
            reward = []
            for agent_lands in lands_picked:
                reward_agent_temp = []

                reward.append(-10)
        state_of_game = [a.bag_pointer_actual for b in self.agent_processor.all_agents for i,a in enumerate(b.land_cells_owned) ]
        print(f"state of the game {state_of_game}")

        print(f"rewards per agent {reward}")
        if all(elem in state_of_game  for elem in [0,0,100,0]):
            print()
        return self._create_observation(), reward, done, None


def process_pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()


if __name__ == '__main__':
    gym_driver = gymDriver()
    gym_driver.reset()
    while True:
        gym_driver.clockobject.tick(5)
        gym_driver.render()
        gym_driver.step()
        gym_driver.counter += 1

        if gym_driver.counter % 20 == 0:
            gym_driver.reset()
