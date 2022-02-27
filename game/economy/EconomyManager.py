import random
from typing import List

from scipy.spatial import distance

from ai import Agent
from game.GlobalParamsGame import GlobalEconomyParams


class EconomyManager:
    def __init__(self, all_agents, pollinators_processor):
        self.upcost_price = GlobalEconomyParams.LAND_UPCOST
        self.all_agents = all_agents
        # self.starting_gold_multiplier = GlobalEconomyParams.STARTING_GOLD_MULTIPLIER
        self.land_fee = GlobalEconomyParams.LAND_UPCOST

        self.polinator_processor = pollinators_processor

    def deduce_land_fee(self):
        for a in self.all_agents:

            income = self.add_income(a)
            a.money += income
            a.income = income
            fee_to_pay = len(a.land_cells_owned) * self.land_fee
            if fee_to_pay > a.money:
                self.handle_dead_situation(a)

            else:
                a.money -= fee_to_pay

            a.last_income = income
    def handle_dead_situation(self,agent):
        agent.money = GlobalEconomyParams.STARTING_GOLD
        agent.is_dead=False
        agent.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        agent.color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    def add_income(self, agent):
        total_gross_income = 0

        for land in agent.land_cells_owned:

            if self.polinator_processor.get_pollinated(land):# ToDo Changed
                #print(f"land {(land.x, land.y)}")
            #if self.get_new_pollinated(land):
                this_land_income = (100 - land.bag_pointer_actual) / 100 * GlobalEconomyParams.MAXIMAL_INCOME
                total_gross_income += this_land_income
                land.was_pollinated = True
                land.last_income = this_land_income
            else:
                land.was_pollinated = False
                land.last_income = 0

        return total_gross_income
    def get_new_pollinated(self,land):
        if land.bag_pointer_actual>0  and land.bag_pointer_actual<100:
            return True
        return False