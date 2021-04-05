import random

from game.visuals.Grid import Grid
from game.GlobalParamsGame import GlobalParamsAi, GlobalParamsGame


class PolinattorsProcessor:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.buffer_lands = []
        self.all_polinattors = set([
            (random.randint(0, GlobalParamsGame.MAX_CELLS_NUMER - 1),
             random.randint(0, GlobalParamsGame.MAX_CELLS_NUMER - 1))
            for _ in range(GlobalParamsAi.NUMBER_OF_RANDOM_POLLINATORS)])
        for polinattors in self.all_polinattors:
            self.grid.all_cells[polinattors].is_pollinator = True
            self.grid.all_cells[polinattors].bag_pointer_actual = 100

    def set_active_pollinator(self, land):
        self.all_polinattors.add((land.x, land.y))

    def get_pollinator(self, cords):

        for x, y in self.grid.all_cells.keys():

            if x == cords[0] and y == cords[1]:
                return self.grid.all_cells[(x,y)]
    def clear_pollinators(self):
        to_delete =[]
        for land in self.all_polinattors:
            if self.grid.all_cells[(land[0],land[1])].bag_pointer_actual==0:
                to_delete.append(land)

        for x in to_delete:
            self.all_polinattors.remove(x)
