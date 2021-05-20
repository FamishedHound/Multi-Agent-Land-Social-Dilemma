import random
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
            a.money = 6000
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
            print("income {} last income {}".format(income,a.last_income))
            a.last_income = income

    def add_income(self, agent):
        total_gross_income = 0

        for land in agent.land_cells_owned:

            if self.polinator_processor.get_pollinated(land):
                total_gross_income += (100 - land.bag_pointer_declared) / 100 * GlobalEconomyParams.MAXIMAL_INCOME
                land.was_pollinated = True

            else:
                land.was_pollinated = False

        return total_gross_income
