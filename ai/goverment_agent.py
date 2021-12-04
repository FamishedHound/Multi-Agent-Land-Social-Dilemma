import numpy as np

from game import GlobalParamsGame


class meta_agent():
    def __init__(self, agent_networks, q_value_networks,all_agents):
        board_size = int(GlobalParamsGame.GlobalParamsGame.WINDOW_HEIGHT / GlobalParamsGame.GlobalParamsGame.BLOCKSIZE)
        self.target = np.random.uniform(0, 1, (board_size, board_size))
        self.budget = 0
        self.agent_networks = agent_networks
        self.q_value_networks = q_value_networks
        self.target = np.random.rand(4)
        self.all_agents = all_agents

    def set_this_year_budget(self, new_budget):
        self.budget = new_budget

    def distribute_incetive(self):
        incentive = []
        for j, agent in enumerate(self.all_agents):
            all_pollinators = 0
            for i, land in enumerate(agent.land_cells_owned):
                all_pollinators+=land.bag_pointer_actual/100
            if all_pollinators/len(agent.land_cells_owned) >= 0.8:
                incetive = -1
            elif all_pollinators/len(agent.land_cells_owned) >= 0.5:
                incetive = -0.6
            else:
                incetive=0
                print(f"agent {j} got {all_pollinators/len(agent.land_cells_owned)} and got this incentive {incetive}")
            incentive.append(incetive)
        return incentive
    def step(self,action):
        pass
