import math
import random

from scipy.spatial import distance

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
            self.grid.all_cells[polinattors].bad_pointer_declared = 100

    def set_active_pollinator(self, land):
        self.all_polinattors.add((land.x, land.y))

    def get_pollinator(self, cords):

        for x, y in self.grid.all_cells.keys():

            if x == cords[0] and y == cords[1]:
                return self.grid.all_cells[(x, y)]

    def find_closest_pollinator_to_land(self, current_point , closeness):
        closest_pollinators = list(filter(lambda c: distance.euclidean(c, current_point) < closeness, self.all_polinattors))
        distances = list(map(lambda c: distance.euclidean(c, current_point), closest_pollinators))

        return closest_pollinators, distances

    def clear_pollinators(self):
        to_delete = []
        for land in self.all_polinattors:
            if self.grid.all_cells[(land[0], land[1])].bag_pointer_actual == 0 or self.grid.all_cells[
                (land[0], land[1])].bag_pointer_actual == -1:
                to_delete.append(land)

        for x in to_delete:
            self.all_polinattors.remove(x)

    # I assume that if you have bees you get pollinated
    def get_pollinated(self, land):
        polliator_distance_dict = {c: distance.euclidean(c, (land.x, land.y)) for c in
                                   self.all_polinattors if c != (land.x, land.y)}
        pollinators_within_certain_distance = dict(
            filter(lambda elem: self.distance_less_than(elem[1], 2), polliator_distance_dict.items()))
        for other_pollinator in pollinators_within_certain_distance.keys():
            bag_size_actual = self.get_pollinator(other_pollinator).bag_pointer_actual
            result = self.sample_pollination(bag_size_actual)
            if result:
                return True
        result_from_this_land = self.sample_pollination(land.bag_pointer_actual)

        return  result_from_this_land



        # neighbourhood_actual_pollinators = [self.get_pollinator(k).bag_pointer_actual for k in
        #                                     pollinators_within_certain_distance.keys()]
        # cumulative_neighbour_polinattors = sum(neighbourhood_actual_pollinators)
        # if cumulative_neighbour_polinattors > 100:
        #     cumulative_neighbour_polinattors = 100

    @staticmethod
    def distance_less_than(number, less_than):
        return number <= less_than

    @staticmethod
    def sample_pollination(x):
        probablity = -1.757078e-8 - (-0.02995441/0.5990879)*(1 - math.exp(-0.5990879*x))
        randy_random = random.uniform(0,1)
        return  randy_random < probablity