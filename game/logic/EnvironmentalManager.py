from random import uniform

import scipy
import numpy as np
from scipy.spatial import distance
from math import e
from game.logic import PolinattorsProcessor
import math


class EnvironmentalManager:
    def __init__(self, pollinators_processor: PolinattorsProcessor):
        self.pollinators_processor = pollinators_processor

    def calculate_euclidian_distance(self, a, b):
        return np.linalg.norm(a - b)

    def find_closest_pollinator_to_land(self, current_point, all_pollinators):
        closest_pollinator = min(all_pollinators, key=lambda c : distance.euclidean(c, current_point))
        distances = list(map(lambda c : distance.euclidean(c, current_point),all_pollinators))

        return self.pollinators_processor.get_pollinator(closest_pollinator),distances[0]

    def process_declared_lands(self):
        for land in self.pollinators_processor.buffer_lands:
            closest_polinator,distance = self.find_closest_pollinator_to_land((land.x, land.y), self.pollinators_processor.all_polinattors)
            self.calculate_environmental_bag(closest_polinator,self.pollinators_processor.get_pollinator((land.x,land.y)),distance)
        self.pollinators_processor.buffer_lands = []

    def calculate_environmental_bag(self, closest_pollinator,land,x):
        probability = 1.092391 * e ** (-(x - -2.526657) ** 2 / (2 * 6.155006 ** 2))
        #print(f"probability was {probability} distance was {x}")
        random_chance = uniform(0, 1)
        #print("random chance {}".format(random_chance))
        if random_chance <= probability:
            land.bag_pointer_actual = land.bag_pointer_declared

        land.bag_pointer_declared = 0


