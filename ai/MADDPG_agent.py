import random

from ai.Agent import Agent


class MADDPGAGENT(Agent):
    def __init__(self, id, pos_x: int, pos_y: int, number_of_lands: int, agent_type):
        super().__init__(id, pos_x, pos_y, number_of_lands, agent_type)
        self.alpha = round(random.uniform(0.15,1),2)
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
            decisions.append(bad_size_declared)
        return decisions


