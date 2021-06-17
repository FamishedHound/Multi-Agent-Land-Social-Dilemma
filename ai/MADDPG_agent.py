from ai.Agent import Agent


class MADDPGAGENT(Agent):
    def __init__(self, id, pos_x: int, pos_y: int, number_of_lands: int, agent_type):
        super().__init__(id, pos_x, pos_y, number_of_lands, agent_type)

    def select_action(self, neural_net_output_number):
        a_bag_numbers = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        return min(range(len(a_bag_numbers)), key=lambda i: abs(a_bag_numbers[i] - neural_net_output_number)) * 10
    #ToDo be wary of these action[0][0] it migth be tottally wroooong if 1 agent 1 land schema changes
    def make_a_decision(self, agent_id,action):
        for i,land in enumerate(self.land_cells_owned):

            bad_size_declared = self.select_action(action[0][0])
            land.bag_pointer_declared = bad_size_declared



