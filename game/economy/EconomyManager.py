from typing import List

from scipy.spatial import distance

from ai import Agent
from game.GlobalParamsGame import GlobalEconomyParams


class EconomyManager:
    def __init__(self, all_agents, pollinators_processor):
        self.upcost_price = GlobalEconomyParams.LAND_UPCOST
        self.all_agents = all_agents
        self.starting_gold_multiplier = GlobalEconomyParams.STARTING_GOLD_MULTIPLIER
        self.land_fee = GlobalEconomyParams.LAND_UPCOST
        for a in all_agents:
            a.money = 1000
        self.polinator_processor = pollinators_processor

    def deduce_land_fee(self):
        for a in self.all_agents:
            income = self.add_income(a)
            a.money += income
            fee_to_pay = len(a.land_cells_owned) * self.land_fee
            if fee_to_pay > a.money:
                for land in a.land_cells_owned:
                    land.bag_pointer_actual = -1
                a.is_dead = True

            else:
                a.money -= fee_to_pay

    def add_income(self, agent):
        total_gross_income = 0

        for land in agent.land_cells_owned:

            dict_land_euclidian = {c: distance.euclidean(c, (land.x, land.y)) for c in
                                   self.polinator_processor.all_polinattors if c != (land.x, land.y)}
            newDict = dict(filter(lambda elem: self.distance_less_than(elem[1], 3), dict_land_euclidian.items()))

            neighbourhood_actual_pollinators = [self.polinator_processor.get_pollinator(k).bag_pointer_actual for k in
                                                newDict.keys()]
            cumulative_neighbour_polinattors = sum(neighbourhood_actual_pollinators)
            if cumulative_neighbour_polinattors > 100:
                cumulative_neighbour_polinattors = 100
            total_gross_income+= (cumulative_neighbour_polinattors - land.bag_pointer_actual) /100 * GlobalEconomyParams.MAXIMAL_INCOME
        return total_gross_income

    def distance_less_than(self, number, less_than):
        return number < less_than
