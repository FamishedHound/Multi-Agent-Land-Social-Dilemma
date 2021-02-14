import random

from typing import List

from game.visuals.LandCell import LandCell


class Agent:
    def __init__(self, id, pos_x : int, pos_y : int):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.id = id
        self.land_cells_owned: List[LandCell] = []
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
