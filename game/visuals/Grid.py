import pygame

from game.GlobalParamsGame import GlobalParamsGame
from game.visuals.LandCell import LandCell


class Grid:
    def __init__(self):
        from game.visuals.LandCell import LandCell
        self.all_cells = {(x,y):LandCell(x, y) for x in range(GlobalParamsGame.WINDOW_WIDTH) for y in
                          range(GlobalParamsGame.WINDOW_HEIGHT)}

    def drawGrid(self):

        for cell in self.all_cells.values():
            if cell.is_owned:
                pygame.draw.rect(GlobalParamsGame.SCREEN, cell.owner.color, cell.get_rect())
                pygame.draw.rect(GlobalParamsGame.SCREEN, GlobalParamsGame.WHITE, cell.get_rect(), 2)
            else:
                pygame.draw.rect(GlobalParamsGame.SCREEN, GlobalParamsGame.WHITE, cell.get_rect(),2)
