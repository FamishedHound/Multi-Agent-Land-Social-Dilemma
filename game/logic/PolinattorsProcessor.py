import math
import random

from scipy.spatial import distance

from game.visuals.Grid import Grid
from game.GlobalParamsGame import GlobalParamsAi, GlobalParamsGame

#Utility and pollination range and formuka
class PolinattorsProcessor:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.buffer_lands = []
        self.all_polinattors = set([
            (random.randint(0, GlobalParamsGame.MAX_CELLS_NUMER - 1),
             random.randint(0, GlobalParamsGame.MAX_CELLS_NUMER - 1))
            for _ in range(GlobalParamsAi.NUMBER_OF_RANDOM_POLLINATORS)])
        for polinattors in self.all_polinattors:
            self.get_pollinator(polinattors).is_pollinator = True
            self.get_pollinator(polinattors).bag_pointer_declared = 10
            self.get_pollinator(polinattors).bag_pointer_actual = 10
            self.grid.all_cells[polinattors].is_pollinator = True
            self.grid.all_cells[polinattors].bag_pointer_declared = 10
            self.grid.all_cells[polinattors].bag_pointer_actual = 10


    def set_active_pollinator(self, land):
        self.all_polinattors.add((land.x, land.y))

    def get_pollinator(self, cords):

        for x, y in self.grid.all_cells.keys():

            if x == cords[0] and y == cords[1]:
                return self.grid.all_cells[(x, y)]

    def find_closest_pollinator_to_land(self, current_point, closeness):
        closest_pollinators = list(
            filter(lambda c: distance.euclidean(c, current_point) < closeness, self.all_polinattors))
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
        for l,v in self.grid.all_cells.items():
            if v.bag_pointer_actual>0:
                self.all_polinattors.add(l)
    def check_for_failed_pollinators_during_exploration(self):

        for land in self.all_polinattors:
            if self.grid.all_cells[(land[0], land[1])].bag_pointer_actual != 0:
                return False
        return True


    # I assume that if you have bees you get pollinated
    def get_pollinated(self, land):

        polliator_distance_dict = {c: distance.euclidean(c, (land.x, land.y)) for c in
                                   self.all_polinattors if c != (land.x, land.y)}
        pollinators_within_certain_distance = dict(
            filter(lambda elem: self.distance_less_than(elem[1], 1.9), polliator_distance_dict.items()))
        for other_pollinator in pollinators_within_certain_distance.keys():
            bag_size_actual = self.get_pollinator(other_pollinator).bag_pointer_actual
            result = self.sample_pollination(bag_size_actual)

            if result:
                return True
        result_from_this_land = self.sample_pollination(land.bag_pointer_actual,0) #ToDo for learning purpose

        return result_from_this_land

        # neighbourhood_actual_pollinators = [self.get_pollinator(k).bag_pointer_actual for k in
        #                                     pollinators_within_certain_distance.keys()]
        # cumulative_neighbour_polinattors = sum(neighbourhood_actual_pollinators)
        # if cumulative_neighbour_polinattors > 100:
        #     cumulative_neighbour_polinattors = 100

    @staticmethod
    def distance_less_than(number, less_than):
        return number <= less_than

    @staticmethod
    def sample_pollination(x,mode=0):


        if x ==50:
            return True
        else:
            return False
            # probablity = 0.7627864 + (-1.579016e-7 - 0.7627864)/(1 + (x/84.04566)**13.87343)
        # probablity =  1.52239 + (-0.001725851 - 1.52239)/(1 + (x/98.6658)**7.729825)
        # if probablity <0:
        #      probablity=0
        # randy_random = random.uniform(0, 1)
        # return randy_random < probablity


