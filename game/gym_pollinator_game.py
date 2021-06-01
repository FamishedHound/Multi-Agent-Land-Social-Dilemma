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


class gymDriver(gym.Env):
    def reset(self):
        self.polinattor_processor = PolinattorsProcessor(grid=self.grid)
        self.agent_processor = AgentProcessor(grid=self.grid, pollinators_processor=self.polinattor_processor)
        self.agent_processor.seperate_land()
        self.agent_processor.clear_empty_agents()
        self.action_processor = ActionProcessor(all_agents=self.agent_processor.all_agents,
                                                pollinator_processor=self.polinattor_processor)

    def render(self, mode='human'):
        self.grid.drawGrid()
        process_pygame_events()
        pygame.display.update()

    def step(self, action=None):
        self.action_processor.all_agents_make_a_move()
        self.environmental_manager.process_declared_lands()
        self.polinattor_processor.clear_pollinators()
        self.economy_manager.deduce_land_fee()

    def __init__(self):
        self.grid = Grid()

        self.polinattor_processor = PolinattorsProcessor(grid=self.grid)
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
def process_pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
if __name__ == '__main__':
    gym_driver = gymDriver()
    while True:
        gym_driver.clockobject.tick(99)
        gym_driver.render()
        gym_driver.step()