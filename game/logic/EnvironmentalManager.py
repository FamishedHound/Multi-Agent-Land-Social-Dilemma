import scipy
import numpy as np
from game.logic import PolinattorsProcessor
import math

class EnvironmentalManager:
    def __init__(self, pollinators_processor: PolinattorsProcessor):
        self.pollinators_processor = pollinators_processor

        self.all_pollinators = []

    def calculate_euclidian_distance(self, a, b):
        return np.linalg.norm(a-b)

    def find_closest_pollinator_to_land(self):

        s = [(1, 4), (4, 2), (6, 3)]
        p = (3, 7)

        p0, p1 = p
        dist = []

        for s0, s1 in s:
            dist_ = math.sqrt((p0 - s0) ** 2 + (p1 - s1) ** 2)
            dist_ = dist_ + 1

            dist.append(dist_)  # S
    def process_declared_lands(self):
        for land in self.pollinators_processor.buffer_lands:
            pass
        self.pollinators_processor.buffer_lands = []