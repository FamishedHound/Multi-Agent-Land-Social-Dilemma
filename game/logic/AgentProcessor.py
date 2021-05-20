from ai.Agent import Agent
from ai.rule_based_agent import RuleBasedAgent
from ai.within_my_land_ageent import LandAgent
from game.GlobalParamsGame import GlobalParamsAi, GlobalParamsGame
from game.visuals.Grid import Grid
import random

from game.visuals.LandCell import LandCell


class AgentProcessor:
    def __init__(self, grid: Grid, pollinators_processor):
        self.all_agents = []
        counter_agent_id = 0
        # ToDo rewrite this shit
        while len(self.all_agents) != GlobalParamsAi.NUMBER_OF_AGENTS:
            # typee = random.uniform(0, 1)
            #
            # new_agent = Agent(counter_agent_id, random.randint(0, GlobalParamsGame.MAX_CELLS_NUMER),
            #                   random.randint(0, GlobalParamsGame.MAX_CELLS_NUMER), 25)
            # if 0.3 < typee:
            new_agent = LandAgent(counter_agent_id, random.randint(0, GlobalParamsGame.MAX_CELLS_NUMER),
                                       random.randint(0, GlobalParamsGame.MAX_CELLS_NUMER), 25, "RuleBasedAgent",
                                       pollinators_processor)
            if new_agent not in self.all_agents:
                self.all_agents.append(new_agent)
            counter_agent_id += 1
        self.grid = grid

    def seperate_land(self):

        self.generate_agents_intial_positions()
        self.distribute_unoccupied_land()

    def generate_agents_intial_positions(self):
        for agent in self.all_agents:
            for cell in self.grid.all_cells.values():
                if LandCell(agent.pos_x, agent.pos_y) == cell:
                    agent.land_cells_owned.append(cell)
                    self.set_ownership_of_land_piece(agent, cell)

    def set_ownership_of_land_piece(self, agent, cell):
        cell.set_owned(True)
        cell.set_owner(agent)

    def clear_empty_agents(self):
        for agent in self.all_agents:
            if len(agent.land_cells_owned) == 0:
                self.all_agents.remove(agent)

    def distribute_unoccupied_land(self):
        done = False
        counter = 0
        while not done:
            done = True
            for x in self.grid.all_cells.values():
                if not x.is_owned:
                    done = False

            for agent in self.all_agents:

                agent = agent
                if agent.predefined_number_of_lands > len(agent.land_cells_owned):
                    counters = 0
                    while counters < len(agent.land_cells_owned):

                        curr_land = agent.land_cells_owned[counters]
                        if curr_land.x + 1 < GlobalParamsGame.MAX_CELLS_NUMER:
                            cell = self.grid.all_cells[(curr_land.x + 1, curr_land.y)]
                            if not cell.is_owned:
                                agent.land_cells_owned.append(cell)
                                self.set_ownership_of_land_piece(agent, cell)
                                break
                        if curr_land.x - 1 >= 0:
                            cell = self.grid.all_cells[(curr_land.x - 1, curr_land.y)]
                            if not cell.is_owned:
                                agent.land_cells_owned.append(cell)
                                self.set_ownership_of_land_piece(agent, cell)
                                break
                        if curr_land.y + 1 < GlobalParamsGame.MAX_CELLS_NUMER:
                            cell = self.grid.all_cells[(curr_land.x, curr_land.y + 1)]
                            if not cell.is_owned:
                                agent.land_cells_owned.append(cell)
                                self.set_ownership_of_land_piece(agent, cell)
                                break
                        if curr_land.y - 1 >= 0:
                            cell = self.grid.all_cells[(curr_land.x, curr_land.y - 1)]
                            if not cell.is_owned:
                                agent.land_cells_owned.append(cell)
                                self.set_ownership_of_land_piece(agent, cell)
                                break

                        counters += 1

                counter += 1
        for x in self.grid.all_cells.values():
            if not x.is_owned:
                exit("not all lands where distributed")
