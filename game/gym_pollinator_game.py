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
    def _get_reward(self):
        agents_rewards = []
        for i, agent in enumerate(self.agent_processor.all_agents):
            agent_land_rewards = []
            reward = 0
            for land in agent.land_cells_owned:
                reward = land.last_income/100
                land=land
                # if land.bag_pointer_declared == 60 :
                #     reward = 1
                # elif land.bag_pointer_declared > 60:
                #     reward = 60/land.bag_pointer_declared
                # else:
                #     reward = land.bag_pointer_declared/60
                #


                agent_land_rewards.append(reward)
            agents_rewards.append(agent_land_rewards)
        return agents_rewards

    def _create_observation(self):
        board_size = int(GlobalParamsGame.GlobalParamsGame.WINDOW_HEIGHT / GlobalParamsGame.GlobalParamsGame.BLOCKSIZE)
        land_per_agent_obs = []
        for agent in self.agent_processor.all_agents:
            single_agent_obs = []
            for land in agent.land_cells_owned:
                empty_obs_position = np.zeros((board_size, board_size))
                empty_obs_declared = np.zeros((board_size, board_size))
                empty_obs_actual = np.zeros((board_size, board_size))
                empty_obs_position[land.x, land.y] = 1
                empty_obs_declared[land.x, land.y] = land.bag_pointer_declared/100
                empty_obs_actual[land.x, land.y] = land.bag_pointer_actual/100

                single_agent_obs.append( np.array([empty_obs_position,empty_obs_declared, empty_obs_actual]))
            land_per_agent_obs.append(single_agent_obs)
        return land_per_agent_obs

    def render(self, mode='human'):
        self.grid.drawGrid()
        process_pygame_events()
        pygame.display.update()

    def step(self, action=None,randy_random=None):
        done = False
        if self.polinattor_processor.check_for_failed_pollinators_during_exploration():
            self.reset()
            done = True

        self.polinattor_processor.clear_pollinators()
        lands_picked = self.action_processor.all_agents_make_a_move(action)
        self.environmental_manager.process_declared_lands()
        self.economy_manager.deduce_land_fee()
        bla = [f"agent {i} has this : " + str(a.money) +"\n" for i,a in enumerate(self.agent_processor.all_agents)]
        print(f"Actual cells {bla}")

        bla = self._create_observation()
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
                for agent_land in agent_lands:
                        reward_agent_temp.append(-1)
                reward.append(reward_agent_temp)
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
