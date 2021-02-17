from ai.Agent import Agent
from game.GlobalParamsGame import GlobalParamsAi, GlobalParamsGame
from game.visuals.Grid import Grid
import random

from game.visuals.LandCell import LandCell


class AgentProcessor:
    def __init__(self, grid: Grid):
        # ToDo ensure not overlaping of agents '''{0: 5, 1: 10, 2: 15}[x % 3]'''
        self.all_agents = [Agent(x, random.randint(0, GlobalParamsGame.MAX_CELLS_NUMER),
                                 random.randint(0, GlobalParamsGame.MAX_CELLS_NUMER),15 ) for x in
                           range(GlobalParamsAi.NUMBER_OF_AGENTS)]
        self.grid = grid

    def seperate_land(self):

        self.generate_agents_intial_positions()
        self.distribute_unoccupied_land()

    def generate_agents_intial_positions(self):
        for agent in self.all_agents:
            for cell in self.grid.all_cells.values():
                if LandCell(agent.pos_x, agent.pos_y) == cell:
                    agent.land_cells_owned.append(cell)
                    cell.set_owned(True)
                    cell.set_owner(agent)

    def distribute_unoccupied_land(self):
        done = False
        counter = 0
        while counter < 5000:



            for agent in self.all_agents:
                agent = agent
                if agent.predefined_number_of_lands>len(agent.land_cells_owned):
                    counters = 0
                    while counters< len(agent.land_cells_owned):

                        curr_land = agent.land_cells_owned[counters]
                        if curr_land.x + 1 < GlobalParamsGame.MAX_CELLS_NUMER:
                            cell = self.grid.all_cells[(curr_land.x + 1, curr_land.y)]
                            if not cell.is_owned:
                                agent.land_cells_owned.append(cell)
                                cell.set_owned(True)
                                cell.set_owner(agent)
                                break
                        if curr_land.x - 1 >= 0:
                            cell = self.grid.all_cells[(curr_land.x - 1, curr_land.y)]
                            if not cell.is_owned:
                                agent.land_cells_owned.append(cell)
                                cell.set_owned(True)
                                cell.set_owner(agent)
                                break
                        if curr_land.y + 1 < GlobalParamsGame.MAX_CELLS_NUMER:
                            cell = self.grid.all_cells[(curr_land.x, curr_land.y + 1)]
                            if not cell.is_owned:
                                agent.land_cells_owned.append(cell)
                                cell.set_owned(True)
                                cell.set_owner(agent)
                                break
                        if curr_land.y - 1 >= 0:
                            cell = self.grid.all_cells[(curr_land.x, curr_land.y - 1)]
                            if not cell.is_owned:
                                agent.land_cells_owned.append(cell)
                                cell.set_owned(True)
                                cell.set_owner(agent)
                                break

                        if curr_land.y + 1 < GlobalParamsGame.MAX_CELLS_NUMER and curr_land.x + 1 < GlobalParamsGame.MAX_CELLS_NUMER:
                            cell = self.grid.all_cells[(curr_land.x + 1, curr_land.y + 1)]
                            if not self.grid.all_cells[(curr_land.x + 1, curr_land.y + 1)].is_owned:
                                agent.land_cells_owned.append(cell)
                                cell.set_owned(True)
                                cell.set_owner(agent)
                                break
                        if curr_land.x + 1 < GlobalParamsGame.MAX_CELLS_NUMER and curr_land.y - 1 >= 0:
                            cell = self.grid.all_cells[(curr_land.x + 1, curr_land.y - 1)]
                            if not self.grid.all_cells[(curr_land.x + 1, curr_land.y - 1)].is_owned:
                                agent.land_cells_owned.append(cell)
                                cell.set_owned(True)
                                cell.set_owner(agent)
                                break
                        if curr_land.x - 1 >= 0 and curr_land.y + 1 < GlobalParamsGame.MAX_CELLS_NUMER:
                            cell = self.grid.all_cells[(curr_land.x - 1, curr_land.y + 1)]
                            if not self.grid.all_cells[(curr_land.x - 1, curr_land.y + 1)].is_owned:
                                agent.land_cells_owned.append(cell)
                                cell.set_owned(True)
                                cell.set_owner(agent)
                                break
                        if curr_land.x - 1 >= 0 and curr_land.y - 1 > 0:
                            cell = self.grid.all_cells[(curr_land.x - 1, curr_land.y - 1)]
                            if not self.grid.all_cells[(curr_land.x - 1, curr_land.y - 1)].is_owned:
                                agent.land_cells_owned.append(cell)
                                cell.set_owned(True)
                                cell.set_owner(agent)
                                break

                        counters+=1

                counter += 1
