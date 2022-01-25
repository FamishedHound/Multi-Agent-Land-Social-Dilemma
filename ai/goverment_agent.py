

import numpy as np

from game import GlobalParamsGame
import torch as th

class meta_agent():
    def __init__(self, agent_networks, q_value_networks,all_agents):
        board_size = int(GlobalParamsGame.GlobalParamsGame.WINDOW_HEIGHT / GlobalParamsGame.GlobalParamsGame.BLOCKSIZE)
        self.target = np.random.uniform(0,1, (4))
        self.target = [0.1,0.5,0.9,0.2]#[round(x,1) for x in self.target]
        self.budget = 0
        self.agent_networks = agent_networks
        self.q_value_networks = q_value_networks
        self.all_agents = all_agents
        self.big_incentive = 0
        self.medium_incentive = 0
        self.low_incentive = 0
        self.interpreted_agent = None
        self.interpreted_obs = None
        self.counter_networks = 0

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
                #print(f"agent {j} got {all_pollinators/len(agent.land_cells_owned)} and got this incentive {incetive}")
            incentive.append(incetive)
        return incentive
    def distribute_incentive_2(self):
        incentive = []
        for j, agent in enumerate(self.all_agents):
            all_pollinators = 0
            for i, land in enumerate(agent.land_cells_owned):
                all_pollinators += land.bag_pointer_actual / 100
            if all_pollinators / len(agent.land_cells_owned) >= 0.8:
                incetive = 0.6
            elif all_pollinators / len(agent.land_cells_owned) >= 0.5:
                incetive = 0.1
            else:
                incetive = -0.25
                # print(f"agent {j} got {all_pollinators/len(agent.land_cells_owned)} and got this incentive {incetive}")
            incentive.append(incetive)
        return incentive
    def optimise_incentives(self,obs,agents,agent_networks,epsilon):

        import optuna
        self.agent_networks = agent_networks
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if epsilon==0.2:
            final_incentive = []
            for i,a in enumerate(agents):
                self.counter_networks = i
                self.interpreted_obs=np.array(obs)[0][0][0]
                self.interpreted_agent=a
                study = optuna.create_study(direction='minimize')
                study.optimize(self.objective, n_trials=100)
                final_incentive.append(study.best_params['x'])

            return final_incentive
        else:
            return [0,0,0,0]


    def objective(self,trial):
        x = trial.suggest_float('x', -1, 1)

        new_incentive = self.get_agents_land_positions(self.interpreted_agent, self.interpreted_obs, x)
        new_state = th.cat([th.from_numpy(self.interpreted_obs).float().unsqueeze(0), th.from_numpy(new_incentive).float().unsqueeze(0)], dim=0).unsqueeze(0)
        decisions = self.agent_networks[self.counter_networks](
           new_state.cuda()).squeeze().data.cpu()

        return abs(self.target[self.counter_networks]-decisions.mean().item())
    def get_agents_land_positions(self,agent,other_array,incentive ):
        incentive_arr = np.zeros_like(other_array)
        for land in agent.land_cells_owned:
            incentive_arr[land.x,land.y] = incentive
        return incentive_arr