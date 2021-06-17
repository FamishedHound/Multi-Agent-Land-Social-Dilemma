import random

from ai.Agent import Agent
from game.logic.PolinattorsProcessor import PolinattorsProcessor


class ActionProcessor:

    def __init__(self, all_agents, pollinator_processor: PolinattorsProcessor):
        self.all_agents = all_agents
        self.action_space = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.pollinators_processor = pollinator_processor

    def all_agents_make_a_move(self, actions):
        agents_actions = zip(self.all_agents, actions)
        counter = 0
        for i, x in enumerate(agents_actions):
            agent, actions = x
            if not agent.is_dead and agent.agent_type == None:
                agent.make_a_decision(i, actions)
            elif not agent.is_dead:
                agent.make_a_decision(i, actions)
            counter += 1

    def make_random_action(self, agent):
        for land in agent.land_cells_owned:
            action_space_random = random.randint(0, len(self.action_space) - 1)
            land.bag_pointer_declared = self.action_space[action_space_random]
            self.pollinators_processor.buffer_lands.append(land)
