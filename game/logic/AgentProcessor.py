from ai.Agent import Agent
from ai.MADDPG_agent import MADDPGAGENT
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
        self.grid = grid
        self.agents_pos_memory = set()
        # ToDo rewrite this shit
        self.seperate_land()
        # while len(self.all_agents) != GlobalParamsAi.NUMBER_OF_AGENTS:
        #     x, y = self.generate_two_random_numbers_that_does_not_hold_agent()
        #     cell = self.grid.get_cell((x, y))
        #
        #     new_agent = MADDPGAGENT(counter_agent_id, x,
        #                             y, 25, "RuleBasedAgent")
        #     new_agent.land_cells_owned.append(cell)
        #     self.set_ownership_of_land_piece(new_agent, cell)
        #     if new_agent not in self.all_agents:
        #         self.all_agents.append(new_agent)
        #     counter_agent_id += 1
        self.grid = grid

    def generate_two_random_numbers_that_does_not_hold_agent(self):
        while True:

            x = random.randint(0, GlobalParamsGame.MAX_CELLS_NUMER - 1)
            y = random.randint(0, GlobalParamsGame.MAX_CELLS_NUMER - 1)
            if (x, y) not in self.agents_pos_memory:
                self.agents_pos_memory.add((x, y))
                return x, y

    def seperate_land(self):

        # self.generate_agents_intial_positions()
        self.distribute_unoccupied_land()

    # def generate_agents_intial_positions(self):
    #     for agent in self.all_agents:
    #         for cell in self.grid.all_cells.values():
    #             if LandCell(agent.pos_x, agent.pos_y) == cell:
    #                 agent.land_cells_owned.append(cell)
    #                 self.set_ownership_of_land_piece(agent, cell)

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

        for i in range(GlobalParamsAi.NUMBER_OF_AGENTS):

            if i == 0:
                agent =  MADDPGAGENT(i, 0,1, 25, "RuleBasedAgent")
                cells = [self.grid.all_cells[(0, 0)], self.grid.all_cells[(0, 1)], self.grid.all_cells[(0, 2)],
                         self.grid.all_cells[(0, 3)]]
                for land in cells:
                    agent.land_cells_owned.append(land)
                    self.set_ownership_of_land_piece(agent, land)
                self.all_agents.append(agent)
            if i==1:
                agent = MADDPGAGENT(i, 0, 1, 25, "RuleBasedAgent")
                cells = [self.grid.all_cells[(0, 1)], self.grid.all_cells[(1, 1)], self.grid.all_cells[(2, 1)],
                         self.grid.all_cells[(0, 2)],self.grid.all_cells[(0, 3)]]
                for land in cells:
                    agent.land_cells_owned.append(land)
                    self.set_ownership_of_land_piece(agent, land)
                self.all_agents.append(agent)
            if i==2:
                agent = MADDPGAGENT(i, 1, 2, 25, "RuleBasedAgent")
                cells = [self.grid.all_cells[(1, 2)], self.grid.all_cells[(2, 2)], self.grid.all_cells[(1, 3)],
                         self.grid.all_cells[(2, 3)]]
                for land in cells:
                    agent.land_cells_owned.append(land)
                    self.set_ownership_of_land_piece(agent, land)
                self.all_agents.append(agent)
            if i == 3:
                agent = MADDPGAGENT(i, 3, 1, 25, "RuleBasedAgent")
                cells = [self.grid.all_cells[(3, 1)], self.grid.all_cells[(3, 2)], self.grid.all_cells[(3, 3)]]
                for land in cells:
                    agent.land_cells_owned.append(land)
                    self.set_ownership_of_land_piece(agent, land)
                self.all_agents.append(agent)

            counter += 1
        # for x in self.grid.all_cells.values():
        #     if not x.is_owned:
        #         exit("not all lands where distributed")


'''
OLD logic for seperating the land 

        while not done:
            done = True
            for x in self.grid.all_cells.values():
                if not x.is_owned:
                    done = False
            # Fixing the scenario for 4x4 grid and 4 agents
# agent = agent
                # if agent.predefined_number_of_lands > len(agent.land_cells_owned):
                #     counters = 0
                #     while counters < len(agent.land_cells_owned):
                # 
                #         curr_land = agent.land_cells_owned[counters]
                #         if curr_land.x + 1 < GlobalParamsGame.MAX_CELLS_NUMER:
                #             cell = self.grid.all_cells[(curr_land.x + 1, curr_land.y)]
                #             if not cell.is_owned:
                #                 agent.land_cells_owned.append(cell)
                #                 self.set_ownership_of_land_piece(agent, cell)
                #                 break
                #         if curr_land.x - 1 >= 0:
                #             cell = self.grid.all_cells[(curr_land.x - 1, curr_land.y)]
                #             if not cell.is_owned:
                #                 agent.land_cells_owned.append(cell)
                #                 self.set_ownership_of_land_piece(agent, cell)
                #                 break
                #         if curr_land.y + 1 < GlobalParamsGame.MAX_CELLS_NUMER:
                #             cell = self.grid.all_cells[(curr_land.x, curr_land.y + 1)]
                #             if not cell.is_owned:
                #                 agent.land_cells_owned.append(cell)
                #                 self.set_ownership_of_land_piece(agent, cell)
                #                 break
                #         if curr_land.y - 1 >= 0:
                #             cell = self.grid.all_cells[(curr_land.x, curr_land.y - 1)]
                #             if not cell.is_owned:
                #                 agent.land_cells_owned.append(cell)
                #                 self.set_ownership_of_land_piece(agent, cell)
                #                 break
                # 
                #         counters += 1


'''
