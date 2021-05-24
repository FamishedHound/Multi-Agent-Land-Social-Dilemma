from scipy.spatial import distance

from ai.Agent import Agent

from copy import deepcopy

from game.GlobalParamsGame import GlobalEconomyParams


class LandAgent(Agent):

    def __init__(self, id, pos_x: int, pos_y: int, number_of_lands: int, agent_type, pollinators_processor):
        super().__init__(id, pos_x, pos_y, number_of_lands)
        self.utility = 0
        self.agent_type = agent_type
        self.pollinators_processor = pollinators_processor
        self.money_past = []
        self.average_past = []
        self.pollination_memory = {}
        self.my_fees = len(self.land_cells_owned) * GlobalEconomyParams.LAND_UPCOST
        self.income = 0
        self.observation_counter = 0
        self.emergency_counter = 0
    # If I have pollinator where it can pollinate
    # Look out for pollinators from neighbours and free ride

    def make_a_decision(self):
        if not self.pollination_memory:
            self.pollination_memory = {(k.x, k.y): [] for k in self.land_cells_owned}
        self.take_observation_of_pollination()
        lands_to_process = [x for x in self.land_cells_owned]
        my_pollinators = self.find_my_pollinators()
        closest_pols = []
        for pollinator in my_pollinators:
            closest_pols = self.find_closest_lands_in_my_farm((pollinator.x, pollinator.y), 2)
            for x in closest_pols:
                if pollinator.bag_pointer_actual >= x.bag_pointer_actual:
                    x.bag_pointer_declared = 0

        remaining_lands = [x for x in lands_to_process if x not in closest_pols and x not in my_pollinators]

        self.analyze_current_situation(remaining_lands)

        self.money_past.append(self.income)

    def analyze_current_situation(self, lands_to_analyze):
        self.emergency_counter_measure()

        if self.emergency_counter<3:
            for k, v in self.pollination_memory.items():
                for land in lands_to_analyze:
                    landCell_pollinator = self.pollinators_processor.get_pollinator(k)
                    if landCell_pollinator.x == land.x and \
                            landCell_pollinator.y == land.y:

                        proportion = sum(v) / len(v)
                        if proportion <= 0.1:
                            landCell_pollinator.bag_pointer_declared = 100
                        elif proportion > 0.1 and proportion <= 0.3:
                            landCell_pollinator.bag_pointer_declared += 20
                        else:
                            landCell_pollinator.bag_pointer_declared = 0
    def emergency_counter_measure(self):

        if self.money_past and self.money_past[-1] < self.my_fees:
            self.emergency_counter+=1
        elif self.emergency_counter>=0:
            self.emergency_counter-=1
        print(self.emergency_counter)
        if self.emergency_counter==3:
            for land in self.land_cells_owned:
                if land.bag_pointer_actual < 100:
                    land.bag_pointer_declared=0


    def take_observation_of_pollination(self):
        for land in self.land_cells_owned:
            if land.was_pollinated:
                self.pollination_memory[(land.x, land.y)].append(1)
            else:
                self.pollination_memory[(land.x, land.y)].append(0)

    def find_closest_lands_in_my_farm(self, current_point, closeness):
        closest_lands = list(
            filter(
                lambda c: distance.euclidean((c.x, c.y), current_point) <= closeness and distance.euclidean((c.x, c.y),
                                                                                                           current_point) != 0,
                self.land_cells_owned))

        return closest_lands

    def find_my_pollinators(self):
        return [land for land in self.land_cells_owned if land.bag_pointer_actual > 0]

    @staticmethod
    def average(lst):
        return sum(lst) / len(lst)
