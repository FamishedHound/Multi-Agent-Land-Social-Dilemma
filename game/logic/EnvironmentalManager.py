import scipy
import numpy as np
from game.logic import PolinattorsProcessor


class EnvironmentalManager:
    def __init__(self, pollinators_processor: PolinattorsProcessor):
        self.pollinators_processor = pollinators_processor

        self.all_pollinators = []

    def calculate_euclidian_distance(self, a, b):
        return np.linalg.norm(a-b)
