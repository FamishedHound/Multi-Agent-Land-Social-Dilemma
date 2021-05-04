from random import uniform, randint, choice

import scipy
import numpy as np
from scipy.spatial import distance
from math import e
from game.logic import PolinattorsProcessor
import math


class EnvironmentalManager:
    def __init__(self, pollinators_processor: PolinattorsProcessor):
        self.pollinators_processor = pollinators_processor
        self.all_polinattors = self.pollinators_processor.all_polinattors

    def calculate_euclidian_distance(self, a, b):
        return np.linalg.norm(a - b)

    def find_closest_pollinator_to_land(self, current_point):
        closest_pollinators = list(filter(lambda c: distance.euclidean(c, current_point) < 3, self.all_polinattors))
        distances = list(map(lambda c: distance.euclidean(c, current_point), closest_pollinators))

        return zip(closest_pollinators, distances)

    def process_declared_lands(self):
        for land in self.pollinators_processor.buffer_lands:

            if land.bag_pointer_declared < land.bag_pointer_actual:
                land.bag_pointer_actual = land.bag_pointer_declared
                continue

            current_point = (land.x, land.y)
            closest_pollinators_with_distance = self.find_closest_pollinator_to_land(current_point)
            if closest_pollinators_with_distance:
                self.calculate_environmental_bag(closest_pollinators_with_distance,
                                                 self.pollinators_processor.get_pollinator(current_point))
        self.pollinators_processor.buffer_lands = []

    def calculate_environmental_bag(self, closest_pollinators_with_distance, land):

        for closests_pollinator, euclidian_distance in closest_pollinators_with_distance:
            probability = math.exp(-3*euclidian_distance)

            self.sample_pollinator_to_create_new_one(land, probability,
                                                     self.pollinators_processor.get_pollinator(closests_pollinator))

    def sample_pollinator_to_create_new_one(self, land, probability, pollinator):
        randy_random = uniform(0, 1)
        if land.bag_pointer_declared > land.bag_pointer_actual and land.bag_pointer_actual < 100:

            if randy_random <= probability:

                probability_how_much_we_get = 10 * (1 + 0.03) ** pollinator.bag_pointer_actual / 100
                randy_random = uniform(0, 1)
                if randy_random < probability_how_much_we_get:
                    land.bag_pointer_actual += 20
                else:
                    land.bag_pointer_actual += 10

        if land.bag_pointer_actual > 0:
            self.pollinators_processor.all_polinattors.add((land.x, land.y))

