from scipy.spatial import distance

from ai.Agent import Agent

from copy import deepcopy


class LandAgent(Agent):

    def __init__(self, id, pos_x: int, pos_y: int, number_of_lands: int, agent_type, pollinators_processor):
        super().__init__(id, pos_x, pos_y, number_of_lands)
        self.utility = 0
        self.agent_type = agent_type
        self.pollinators_processor = pollinators_processor
        self.money_past = []

    # If I have pollinator where it can pollinate
    # Look out for pollinators from neighbours and free ride

    def make_a_decision(self):

        lands_to_process = [x for x in self.land_cells_owned]
        my_pollinators = self.find_my_pollinators()
        closest_pols = []
        for pollinator in my_pollinators:
            closest_pols = self.find_closest_lands_in_my_farm((pollinator.x, pollinator.y), 2)
            if pollinator.bag_pointer_actual <= 50:
                for land in closest_pols:
                    if land.bag_pointer_declared < 100:
                        land.bag_pointer_declared += 10

        remaining_lands = [x for x in lands_to_process if x not in closest_pols and x not in my_pollinators]
        for land in remaining_lands:
            land.bag_pointer_declared=40



        self.money_past.append(self.money)

    def find_closest_lands_in_my_farm(self, current_point, closeness):
        closest_lands = list(
            filter(
                lambda c: distance.euclidean((c.x, c.y), current_point) < closeness and distance.euclidean((c.x, c.y),
                                                                                                           current_point) != 0,
                self.land_cells_owned))

        return closest_lands

    def find_my_pollinators(self):
        return [land for land in self.land_cells_owned if land.bag_pointer_actual > 0]

    @staticmethod
    def average(lst):
        return sum(lst) / len(lst)