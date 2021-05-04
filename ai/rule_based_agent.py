import copy

import numpy as np
from scipy.spatial import distance

from ai.Agent import Agent


class RuleBasedAgent(Agent):
    def __init__(self, id, pos_x: int, pos_y: int, number_of_lands: int, agent_type):
        super().__init__(id, pos_x, pos_y, number_of_lands, agent_type)

        self.all_lands = self.land_cells_owned
        self.new_map = self.divide_land_into_n_smaller_chunks(3)

    def make_a_decision(self):
        pass

    def divide_land_into_n_smaller_chunks(self, n):
        all_lands_copy = copy.deepcopy(self.all_lands)
        new_land_mapping = {}
        for land in self.all_lands:
            current_point = (land.x, land.y)
            if land in all_lands_copy:
                closest_owned_points = sorted(self.all_lands, key=lambda i: distance.euclidean(current_point, i))
                new_land_mapping[land] = closest_owned_points
                all_lands_copy = set(all_lands_copy) - set(closest_owned_points) - set(land)

    def calculate_euclidian_distance(self, a, b):
        return np.linalg.norm(a - b)
