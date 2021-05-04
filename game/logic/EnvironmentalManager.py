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

    def calculate_euclidian_distance(self, a, b):
        return np.linalg.norm(a - b)

    def find_closest_pollinator_to_land(self, current_point, all_pollinators):
        closest_pollinator = min(all_pollinators, key=lambda c: distance.euclidean(c, current_point))
        distances = list(map(lambda c: distance.euclidean(c, current_point), all_pollinators))

        return self.pollinators_processor.get_pollinator(closest_pollinator), distances[0]

    def process_declared_lands(self):
        for land in self.pollinators_processor.buffer_lands:
            closest_polinator, distance = self.find_closest_pollinator_to_land((land.x, land.y),
                                                                               self.pollinators_processor.all_polinattors)
            self.calculate_environmental_bag(closest_polinator,
                                             self.pollinators_processor.get_pollinator((land.x, land.y)), distance)
        self.pollinators_processor.buffer_lands = []

    def calculate_environmental_bag(self, closest_pollinator, land, x):
        probability =  -0.07355621 + (0.8519887 - -0.07355621)/(1 + (x/1.361429)**0.7561517)
        # -0.07355621 + (0.8519887 - -0.07355621)/(1 + (x/1.361429)**0.7561517) #ToDo present meaaning of each coefficent of the equation
        # -0.07355621 + (0.8519887 - -0.07355621)/(1 + (x/1.361429)**0.7561517)
        # 0.000017709 + (0.2850471 - 0.000017709)/(1 + (x/1.695153)**3.722618)
        # -0.2566034 + (2.126455 - -0.2566034)/(1 + (x/1.160579)**0.734929)


        randy_random = uniform(0, 1)
        if land.bag_pointer_declared < land.bag_pointer_actual:
            land.bag_pointer_actual = land.bag_pointer_actual
            land.bag_pointer_declared = 0

        elif randy_random <= probability:
            if closest_pollinator.bag_pointer_actual >=70 :
                roll_strength_from_closest_pollinator = 2
            else:
                roll_strength_from_closest_pollinator = 1

            our_assigned_value = max(
                [choice(range(0, 100, 10)) for _ in range(roll_strength_from_closest_pollinator)])
            final_value = 0
            if our_assigned_value > land.bag_pointer_declared:
                final_value = land.bag_pointer_declared
            else:
                final_value = our_assigned_value
            land.bag_pointer_actual = final_value
        if land.bag_pointer_actual>0:
            self.pollinators_processor.all_polinattors.add((land.x,land.y))
        #ToDo Remove if you aree no longer pollinator
        land.bag_pointer_declared = 0
