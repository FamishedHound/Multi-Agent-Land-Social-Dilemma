import sys

import pygame

from game.logic.ActionProcessor import ActionProcessor
from game.logic.AgentProcessor import AgentProcessor
from game.logic.EnvironmentalManager import EnvironmentalManager
from game.logic.PolinattorsProcessor import PolinattorsProcessor
from game.visuals.Grid import Grid
from game import GlobalParamsGame




def process_pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()


def main_loop():
    grid = Grid()
    agent_processor = AgentProcessor(grid=grid)
    agent_processor.seperate_land()
    agent_processor.clear_empty_agents()
    polinattor_processor = PolinattorsProcessor(grid = grid)
    action_processor = ActionProcessor(all_agents=agent_processor.all_agents,pollinator_processor=polinattor_processor)
    clockobject = pygame.time.Clock()

    environmental_manager = EnvironmentalManager(polinattor_processor)
    environmental_manager.process_declared_lands()
    while True:
        clockobject.tick(99)
        process_pygame_events()
        pygame.display.update()


        action_processor.all_agents_make_a_move()
        environmental_manager.process_declared_lands()
        polinattor_processor.clear_pollinators()
        grid.drawGrid()





if __name__ == '__main__':
    main_loop()
