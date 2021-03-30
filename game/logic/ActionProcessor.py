import random

from ai.Agent import Agent
from game.logic.PolinattorsProcessor import PolinattorsProcessor


class ActionProcessor:

    def __init__(self, all_agents,pollinator_processor: PolinattorsProcessor):
        self.all_agents = all_agents
        self.action_space = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.pollinators_processor = pollinator_processor
    def all_agents_make_a_move(self):
        for agent in self.all_agents:
            self.make_random_action(agent)


    def make_random_action(self, agent):
        for land in agent.land_cells_owned:
            action_space_random = random.randint(0, len(self.action_space)-1)
            land.bag_pointer_declared = self.action_space[action_space_random]
            self.pollinators_processor.buffer_lands.append(land)