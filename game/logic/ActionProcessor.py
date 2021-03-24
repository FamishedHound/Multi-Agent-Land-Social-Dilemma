import random

from ai.Agent import Agent


class ActionProcessor:

    def __init__(self, all_agents):
        self.all_agents = all_agents
        self.action_space = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    def all_agents_make_a_move(self):
        for agent in self.all_agents:
            self.make_random_action(agent)

    def make_random_action(self, agent):
        for land in agent.land_cells_owned:
            land.bag_pointer_declared = self.action_space[random.randint(0, len(self.action_space)-1)]
