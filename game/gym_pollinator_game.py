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
                single_reward = land.last_income / 100 - GlobalParamsGame.GlobalEconomyParams.LAND_UPCOST / 100

                cumulative_reward += single_reward

            money_reward = cumulative_reward / len(agent.land_cells_owned)
            #          0-1   0.9  * 0.9   + 0.1 * 0.35
            reward_internal = agent.alpha * (money_reward) + (1 - agent.alpha) * global_pollinators_reward/2
            final_reward = money_reward + incentive[j] # incentive 0-1 you can divide by 2 everything#ToDo just money reword for debugging
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
        board_size = int(GlobalParamsGame.GlobalParamsGame.WINDOW_HEIGHT / GlobalParamsGame.GlobalParamsGame.BLOCKSIZE)
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
        action = torch.from_numpy(np.reshape(action, (4, 4))).float().unsqueeze(0).unsqueeze(0) #for now no concrete mapping to agents or lands GAN will have to  figure it out
        alone = np.pad(np.array([[1, 2], [2, 3]]), 2, self.pad_with, padder=0)

        # action = torch.ones_like(torch.from_numpy(img)).repeat(4, 1, 1) * torch.from_numpy(action) \
        #     .unsqueeze(1) \
        #     .unsqueeze(2)

        state_action = torch.cat([torch.from_numpy(global_state_).float(),action.squeeze(0)],dim=0).unsqueeze(0)

        if self.index > 0:
            with open(f"C:\\Users\\LukePC\\PycharmProjects\\polinators_social_dilema\\train\\Sa_images\\state_s_{self.index - 1}.pickle", 'wb') as handle:
                state_without_action = torch.from_numpy(global_state_).float().unsqueeze(0)
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
        gym_driver.step(actions, None, [0, 0, 0, 0])
        gym_driver.counter += 1

        if gym_driver.counter % 20 == 0:
            gym_driver.reset()
