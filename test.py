

def optimise_incentives(self, obs, agents, agent_networks, epsilon):
    import optuna
    self.agent_networks = agent_networks
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if epsilon == 0.2:
        final_incentive = []
        for i, a in enumerate(agents):
            self.counter_networks = i
            self.interpreted_obs = np.array(obs)[0][0][0]
            self.interpreted_agent = a
            study = optuna.create_study()
            study.optimize(self.objective, n_trials=2000)
            final_incentive.append(study.best_params['x'])

        return final_incentive
    else:
        return [0, 0, 0, 0]


def objective(self, trial):
    x = trial.suggest_float('x', -1, 1)

    new_incentive = self.get_agents_land_positions(self.interpreted_agent, self.interpreted_obs, x)
    new_state = th.cat(
        [th.from_numpy(self.interpreted_obs).float().unsqueeze(0), th.from_numpy(new_incentive).float().unsqueeze(0)],
        dim=0).unsqueeze(0)
    decisions = self.agent_networks[self.counter_networks](
        new_state.cuda()).squeeze().data.cpu()

    return self.target[self.counter_networks] / decisions.mean().item()


def get_agents_land_positions(self, agent, other_array, incentive):
    incentive_arr = np.zeros_like(other_array)
    for land in agent.land_cells_owned:
        incentive_arr[land.x, land.y] = incentive
    return incentive_arr