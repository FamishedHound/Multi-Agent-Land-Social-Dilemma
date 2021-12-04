from random import uniform, random, choice, randint

import gym
import sys
from time import sleep

import pygame
from gym import spaces

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
        self.observation_space = spaces.Box(0, 1, shape=[1])
        self.clockobject = pygame.time.Clock()
        self.action_space = spaces.Box(-1, 1, shape=[4])
        self.environmental_manager = EnvironmentalManager(self.polinattor_processor)
        self.environmental_manager.process_declared_lands()
        self.economy_manager = EconomyManager(self.agent_processor.all_agents, self.polinattor_processor)
        self.counter = 0

    def reset(self):

        self.polinattor_processor = PolinattorsProcessor(grid=self.grid)

        self.n_agents = len(self.agent_processor.all_agents)

        return self._create_observation([0,0,0,0])

    def _get_reward_for_gov(self):
        agent_lands = np.zeros(4)
        for j,agent in enumerate(self.agent_processor.all_agents):

            cumulative_lands = 0
            for land in agent.land_cells_owned:
                cumulative_lands+=land.bag_pointer_actual/100

            agent_lands[j] = cumulative_lands/len(agent.land_cells_owned)
        return agent_lands
    def get_global_pollinators_reward(self):
        pollinators_reward = 0


        for land in self.grid.all_cells.values():

                pollinators_reward += (land.bag_pointer_actual/100)/len(self.grid.all_cells)
        #print(f"Global pollinators reward {pollinators_reward}")
        return pollinators_reward

    def _get_reward(self,incentive):
        agents_rewards = []
        global_pollinators_reward = self.get_global_pollinators_reward()

        for j, agent in enumerate(self.agent_processor.all_agents):

            cumulative_reward = 0

            for i, land in enumerate(agent.land_cells_owned):

                single_reward = land.last_income / 100

                cumulative_reward += single_reward

            money_reward = cumulative_reward / len(agent.land_cells_owned)
            final_reward = agent.alpha * (money_reward) + (1 - agent.alpha ) * global_pollinators_reward + incentive[j]/2 #incentive 0-1 you can divide by 2 everything
            #agent.money += incentive[j] * 1000
            # print(
            #     f"AGENT:{agent.id} his alpha is {agent.alpha} money :{money_reward}"
            #     f" env :{round(global_pollinators_reward,2)}"
            #     f"trust incentive : {incentive[j]/4}"
            #     f" final_reward : {final_reward} ")

            agents_rewards.append(final_reward)
        return agents_rewards

    def _create_observation(self,incentive):
        board_size = int(GlobalParamsGame.GlobalParamsGame.WINDOW_HEIGHT / GlobalParamsGame.GlobalParamsGame.BLOCKSIZE)
        land_per_agent_obs = []
        #ToDo HERE
        for j,agent in enumerate(self.agent_processor.all_agents):
            single_agent_obs = []
            empty_obs_local_declared = np.zeros((board_size, board_size))
            empty_obs_local_actual = np.zeros((board_size, board_size))
            incentive_np = np.zeros((board_size, board_size))

            for land in agent.land_cells_owned:
                empty_obs_local_declared[land.x, land.y] = land.bag_pointer_declared / 100
                empty_obs_local_actual[land.x, land.y] = land.bag_pointer_actual / 100
                incentive_np[land.x, land.y] = incentive[j]

            single_agent_obs.append(
                np.array([empty_obs_local_actual,incentive_np]))#'''*incentive[j]'''')  # ToDo Deleting actual bag for debugging purposes
            land_per_agent_obs.append(single_agent_obs)
        empty_obs_declared, empty_obs_actual, incentive_global = self.get_global_state_without_position(board_size,
                                                                                                        incentive) #ToDo moved from THERE TO HERE
        # land_per_agent_obs.append(np.array([empty_obs_declared]))
        #print("bla bla {}".format(incentive_global))
        global_obs = np.array([empty_obs_actual,incentive_global])
        return land_per_agent_obs, global_obs

    def get_global_state_without_position(self, board_size,incentive):
        empty_obs_declared = np.zeros((board_size, board_size))
        empty_obs_actual = np.zeros((board_size, board_size))
        incentive_np_global = np.zeros((board_size, board_size))
        for j,agent in enumerate(self.agent_processor.all_agents):
            for land in agent.land_cells_owned:
                empty_obs_declared[land.x, land.y] = land.bag_pointer_declared / 100
                empty_obs_actual[land.x, land.y] = land.bag_pointer_actual / 100
                incentive_np_global[land.x, land.y] = incentive[j]
        return empty_obs_declared, empty_obs_actual,incentive_np_global

    def render(self, mode='human'):
        self.grid.drawGrid()
        process_pygame_events()
        pygame.display.update()

    def step(self, action=None, randy_random=None,incentive=None,render=False):
        if isinstance(action,np.ndarray):
            action=[action.tolist()]
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
        reward = self._get_reward(incentive)

        state_of_game = [a.bag_pointer_actual for b in self.agent_processor.all_agents for i, a in
                         enumerate(b.land_cells_owned)]
        #print(f"state of the game {state_of_game}")

        #print(f"rewards per agent {reward}")

        return self._create_observation(incentive), reward, done, None


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
if __name__ == '__main__':
    gym_driver = gymDriver()
    gym_driver.reset()
    while True:
        gym_driver.clockobject.tick(5)
        gym_driver.render()
        actions = make_random_action(gym_driver.agent_processor.all_agents)
        gym_driver.step(actions,None,[0,0,0,0])
        gym_driver.counter += 1

        if gym_driver.counter % 20 == 0:
            gym_driver.reset()
