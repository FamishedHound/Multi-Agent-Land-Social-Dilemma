import random

from typing import List

from game.visuals.LandCell import LandCell


class Agent:
    def __init__(self, id, pos_x: int, pos_y: int, number_of_lands: int, agent_type=None):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.agent_type = agent_type
        self.id = id
        self.land_cells_owned: List[LandCell] = []
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.predefined_number_of_lands = number_of_lands
        self.no_already_assigned_lands = 0
        self.money = 0
        self.is_dead = False
        self.last_income = None

        self.money = 7000

    def __eq__(self, other):
        return self.pos_x == other.pos_x and self.pos_y == other.pos_y
    def make_a_decision(self):
        pass
