import os

#import pg as pg
import pygame
from pygame import font

from game.GlobalParamsGame import GlobalParamsGame
from game.visuals.LandCell import LandCell


class Grid:
    def __init__(self):
        from game.visuals.LandCell import LandCell
        self.all_cells = {(x, y): LandCell(x, y) for x in range(GlobalParamsGame.MAX_CELLS_NUMER) for y in
                          range(GlobalParamsGame.MAX_CELLS_NUMER)}
    def get_cell(self,cords):
        for k,v in self.all_cells.items():
            if k == cords:
                return v
    def drawGrid(self):
        pygame.font.init()
        font = pygame.font.SysFont("Grobold", 25,italic=True)


        for cell in self.all_cells.values():
            txt_surf = font.render(str(cell.bag_pointer_actual), True, cell.owner.color2)
            if cell.is_owned:

                pygame.draw.rect(GlobalParamsGame.SCREEN, cell.owner.color, cell.get_rect())
                pygame.draw.rect(GlobalParamsGame.SCREEN, cell.owner.color2, cell.get_rect(), 2)
                GlobalParamsGame.SCREEN.blit(txt_surf, (cell.get_rect().x+4,cell.get_rect().y+10))



    def draw_rect_alpha(self, surface, color, rect):
        shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
        surface.blit(shape_surf, rect)

