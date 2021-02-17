import sys

import pygame

from game.logic.AgentProcessor import AgentProcessor
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
    snakeFace = pygame.image.load("/Users/l.pelcner/PycharmProjects/Multi-Agent-Land-Social-Dilemma/images/bee.jpg").convert_alpha()
    #win = GlobalParamsGame.GlobalParamsGame.SCREEN.blit(snakeFace, (5, 5))
    desired_rec = grid.all_cells[(5,5)].get_rect()
    charImage = pygame.transform.scale(snakeFace, desired_rec.size)
    charImage = charImage.convert()
    while True:
        grid.drawGrid()

        GlobalParamsGame.GlobalParamsGame.SCREEN.blit(charImage, desired_rec )
        process_pygame_events()
        pygame.display.update()


if __name__ == '__main__':
    main_loop()
