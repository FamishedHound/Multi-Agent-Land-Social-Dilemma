import numpy as np

from game import GlobalParamsGame


class meta_agent():
    def __init__(self, agent_networks, q_value_networks):
        board_size = int(GlobalParamsGame.GlobalParamsGame.WINDOW_HEIGHT / GlobalParamsGame.GlobalParamsGame.BLOCKSIZE)
        self.target = np.random.uniform(0, 1, (board_size, board_size))
        self.budget = 0
        self.agent_networks = agent_networks
        self.q_value_networks = q_value_networks
        self.target = np.random.rand(board_size, board_size)

    def set_this_year_budget(self, new_budget):
        self.budget = new_budget
    def distribute_incetive(self):
        return [0.6,0.1,0.2,0.1]
