import sys

import pygame

from game.logic.AgentProcessor import AgentProcessor
from game.visuals.Grid import Grid
from game import GlobalParamsGame

grid = Grid()


def process_pygame_events():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()


def main_loop():
    agent_processor = AgentProcessor(grid=grid)
    agent_processor.seperate_land()
    while True:
        grid.drawGrid()
        process_pygame_events()
        pygame.display.update()


if __name__ == '__main__':
    main_loop()
