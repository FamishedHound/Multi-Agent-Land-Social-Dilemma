import random

import numpy as np

from game import GlobalParamsGame
import torch as th

class meta_agent():
    def __init__(self, agent_networks, q_value_networks,all_agents):
        board_size = int(GlobalParamsGame.GlobalParamsGame.WINDOW_HEIGHT / GlobalParamsGame.GlobalParamsGame.BLOCKSIZE)
        self.target = np.random.uniform(0,1, (4))
        self.target = [0.35,0.9,0.1,0.5]#[round(x,1) for x in self.target]
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
        self.decisions = {}
        self.new_state = None
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
            #incetive = random.uniform(-1,1)
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
    def distribute_incentive_3(self):
        incentive = []
        for j, agent in enumerate(self.all_agents):
            all_pollinators = 0
            final_incentive = 0
            for i, land in enumerate(agent.land_cells_owned):

                if land.bag_pointer_actual==100:
                    final_incentive-=0.25
                if i==0 and  land.bag_pointer_actual==50:
                    final_incentive+=0.25
                if i == 1 and land.bag_pointer_actual == 20:
                    final_incentive += 0.25
                if i==2 and land.bag_pointer_actual == 60:
                    final_incentive+=0.25

                # print(f"agent {j} got {all_pollinators/len(agent.land_cells_owned)} and got this incentive {incetive}")
            incentive.append(final_incentive)
        return incentive
    def distribute_incentive_4(self):
        incentive = []
        for j, agent in enumerate(self.all_agents):
            final_incentive = 0
            for i, land in enumerate(agent.land_cells_owned):

                if land.bag_pointer_actual != 100 and land.bag_pointer_actual!=0:
                    final_incentive += 0.2
                else:
                    final_incentive -=0.1


                # print(f"agent {j} got {all_pollinators/len(agent.land_cells_owned)} and got this incentive {incetive}")
            incentive.append(final_incentive)
        return incentive
    def optimise_incentives(self,obs,agents,agent_networks,critics):

        import optuna
        self.agent_networks = agent_networks
        self.critics = critics
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        #If you want agent observations you need their personal one not global oen which was an error
        final_incentive = []
        for i,a in enumerate(agents):
            self.counter_networks = i
            self.debugging_obs = np.array(obs)[self.counter_networks][0]
            self.interpreted_obs=np.array(obs)[self.counter_networks][0][0]
            self.interpreted_agent=a
            study = optuna.create_study(direction='minimize')
            study.optimize(self.objective, n_trials=220)
            final_incentive.append(study.best_params['x'])
            #print(f"before : \n {self.debugging_obs[1]} \n  after: \n {self.new_state[0][1]}")
            print(f"here are decisions for agent {self.counter_networks} that would result {self.agent_networks[self.counter_networks](th.from_numpy(self.debugging_obs).unsqueeze(0).unsqueeze(0).float().cuda()).mean().item()} here are optimised {self.decisions[self.counter_networks].mean()}")
        return final_incentive


    #make decision somehow impact the critic
    def objective(self,trial):
        x = trial.suggest_float('x', -1, 1)
        multiplier = 1
        new_incentive = self.get_agents_land_positions(self.interpreted_agent, self.interpreted_obs, x*multiplier)
        self.new_state = th.cat([th.from_numpy(self.interpreted_obs.copy()).float().unsqueeze(0), th.from_numpy(new_incentive.copy()).float().unsqueeze(0)], dim=0).unsqueeze(0)
        self.decisions[self.counter_networks] = self.agent_networks[self.counter_networks](
           self.new_state.cuda()).squeeze().data
        # self.critics[self.counter_networks](new_state.cuda(),decisions)
        return abs(self.target[self.counter_networks]-self.decisions[self.counter_networks].mean().item())
    def get_agents_land_positions(self,agent,other_array,incentive ):
        incentive_arr = np.zeros_like(other_array)
        for land in agent.land_cells_owned:

            incentive_arr[land.x,land.y] = incentive
        incentive_arr = np.rot90(incentive_arr)
        return incentive_arr