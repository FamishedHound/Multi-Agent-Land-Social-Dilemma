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

    def process_declared_lands(self):

        lands_to_process = [land for land in self.pollinators_processor.grid.all_cells.values()  ]
        for land in lands_to_process:

            if land.bag_pointer_actual != -1:
               land.bag_pointer_actual = land.bag_pointer_declared
               if land.bag_pointer_actual > 0:
                   self.pollinators_processor.all_polinattors.add((land.x, land.y))
                #turned off for NOW
                # current_point = (land.x, land.y)
                # closest_pollinators, distances = self.pollinators_processor.find_closest_pollinator_to_land(current_point,
                #                                                                                             3)
                # if closest_pollinators:
                #     self.calculate_environmental_bag(zip(closest_pollinators, distances),
                #                                      self.pollinators_processor.get_pollinator(current_point))



    def calculate_environmental_bag(self, closest_pollinators_with_distance, land):
        #ToDo enable spreading
        for closests_pollinator, euclidian_distance in closest_pollinators_with_distance:
            probability = math.exp(-1 * euclidian_distance)
            #land.bag_pointer_actual = land.bag_pointer_declared



            self.sample_pollinator_to_create_new_one(land, probability,
                                                     self.pollinators_processor.get_pollinator(closests_pollinator))

    def sample_pollinator_to_create_new_one(self, land, probability, pollinator):
        randy_random = uniform(0, 1)
        actual_bag = land.bag_pointer_actual
        declared_bag = land.bag_pointer_declared
        if declared_bag > actual_bag and actual_bag < 100:

            if randy_random <= probability:
                result =0
                probability_how_much_we_get = 10 * (1 + 0.03) ** pollinator.bag_pointer_actual / 100
                randy_random = uniform(0, 1)
                if randy_random >0 and randy_random <0.7:
                    result +=10
                elif randy_random>0.7 and randy_random <0.8 and actual_bag <=80:
                    result += 20
                elif randy_random>0.8 and randy_random <0.85 and actual_bag <=70:
                    result += 30
                else:
                    result += 10
                if actual_bag + result <= declared_bag:
                    land.bag_pointer_actual += result
                else:
                    land.bag_pointer_actual = declared_bag
        elif declared_bag < actual_bag:
            land.bag_pointer_actual = declared_bag
        if actual_bag > 0:
            self.pollinators_processor.all_polinattors.add((land.x, land.y))
