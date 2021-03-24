import sys

import pygame

from game.logic.ActionProcessor import ActionProcessor
from game.logic.AgentProcessor import AgentProcessor
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
    clockobject = pygame.time.Clock()
    polinattor_processor = PolinattorsProcessor(grid = grid)
    action_processor = ActionProcessor(all_agents=agent_processor.all_agents)
    action_processor.all_agents_make_a_move()

    while True:
        clockobject.tick(60)
        grid.drawGrid()


        process_pygame_events()
        pygame.display.update()


if __name__ == '__main__':
    main_loop()
