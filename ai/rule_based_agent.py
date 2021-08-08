import copy

import numpy as np
from scipy.spatial import distance

from ai.Agent import Agent


class RuleBasedAgent(Agent):
    def __init__(self, id, pos_x: int, pos_y: int, number_of_lands: int, agent_type,pollinators_processor):
        super().__init__(id, pos_x, pos_y, number_of_lands, agent_type)

        self.new_map = None
        self.pollinators_processor=pollinators_processor
        self.counter = 0
    def make_a_decision(self):
        if self.new_map == None:
            self.new_map = self.divide_land_into_n_smaller_chunks(2)
        else:
            if self.counter%3==0:
                for key,value in self.new_map.items():
                    clostest_polinattors = self.find_closest_pollinator_to_land(key)
                    pollinators_potential = sum((self.pollinators_processor.get_pollinator(x).bag_pointer_actual for x in clostest_polinattors))
                    if pollinators_potential>=1200:
                        self.apply_to_all_sub_lands_and_key(key, value, 0)
                    elif pollinators_potential>=800 and pollinators_potential<=1200:
                        self.apply_to_all_sub_lands_and_key(key, value, 10)
                    elif pollinators_potential<=200 and pollinators_potential>=100 :
                        self.apply_to_all_sub_lands_and_key(key, value, 100)

                    elif pollinators_potential>=60 and pollinators_potential<=80:
                        self.apply_to_all_sub_lands_and_key(key, value, 40)
                    elif pollinators_potential>=40 and pollinators_potential<=60:
                        self.apply_to_all_sub_lands_and_key(key, value, 60)
                    elif pollinators_potential>=10 and pollinators_potential<=40:
                        self.apply_to_all_sub_lands_and_key(key, value, 80)
                    else:
                        self.apply_to_all_sub_lands_and_key(key, value, 100)

        self.counter+=1
    def apply_to_all_sub_lands_and_key(self,key,sublands,value):
        key_land_cell = self.pollinators_processor.get_pollinator(key)
        key_land_cell.bag_pointer_declared = value
        self.pollinators_processor.buffer_lands.append(key_land_cell)
        for subland in sublands:
            subland_key = self.pollinators_processor.get_pollinator(subland)
            subland_key.bag_pointer_declared = value
            self.pollinators_processor.buffer_lands.append(subland_key)



    def get_polinattors_potential(self,land_cords):
        pass

    def divide_land_into_n_smaller_chunks(self, n):
        all_lands_copy = copy.deepcopy(self.land_cells_owned)
        all_lands_copy = list(map(lambda x: (x.x, x.y), all_lands_copy))
        buffer_all_lands = copy.deepcopy(all_lands_copy)
        new_land_mapping = {}
        for land in all_lands_copy:

            if land in buffer_all_lands:
                buffer_all_lands.remove(land)
                closest_owned_points = sorted( buffer_all_lands,
                                              key=lambda i: distance.euclidean(land, i))[:n]
                new_land_mapping[land] = closest_owned_points
                buffer_all_lands = set(buffer_all_lands) - set(closest_owned_points) - set(land)

        if len(buffer_all_lands) > 1:
            new_land_mapping[buffer_all_lands[0]] = buffer_all_lands[1:0]

        return new_land_mapping
    def calculate_euclidian_distance(self, a, b):
        return np.linalg.norm(a - b)
