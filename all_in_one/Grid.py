#import pg as pg
import pygame
from pygame import font

from GlobalParamsGame import GlobalParamsGame


class Grid:
    def __init__(self):
        from LandCell import LandCell
        self.all_cells = {(x, y): LandCell(x, y) for x in range(GlobalParamsGame.MAX_CELLS_NUMER) for y in
                          range(GlobalParamsGame.MAX_CELLS_NUMER)}
        self.image = pygame.image.load(
            'C:\\Users\\LukePC\PycharmProjects\\polinators_social_dilema\\images\\117480-middle.png')
        pygame.font.init()
        self.font = pygame.font.SysFont("Grobold", 25,italic=True)
    def get_cell(self,cords):
        for k,v in self.all_cells.items():
            if k == cords:
                return v
    def drawGrid(self):







        for cell in self.all_cells.values():



            self.image = pygame.transform.scale(self.image, (40,40)).convert_alpha()
            self.image.set_alpha(40)
            txt_surf = self.font.render(str(cell.bag_pointer_actual), True, cell.owner.color2)
            txt_surf2 = self.font.render(f"agent : {str(cell.owner.id)}" , True, cell.owner.color2)
            txt_surf3 = self.font.render(f"money : {str(cell.owner.money)}", True, cell.owner.color2)
            txt_surf4 = self.font.render(f"alpha : {str(cell.owner.alpha)}", True, cell.owner.color2)
            if cell.is_owned:

                pygame.draw.rect(GlobalParamsGame.SCREEN, cell.owner.color, cell.get_rect())
                pygame.draw.rect(GlobalParamsGame.SCREEN, cell.owner.color2, cell.get_rect(), 2)
                GlobalParamsGame.SCREEN.blit(txt_surf, (cell.get_rect().x+4,cell.get_rect().y+10))
                GlobalParamsGame.SCREEN.blit(txt_surf2, (cell.get_rect().x + 20, cell.get_rect().y + 50))
                GlobalParamsGame.SCREEN.blit(txt_surf3, (cell.get_rect().x + 20, cell.get_rect().y + 80))
                GlobalParamsGame.SCREEN.blit(txt_surf4, (cell.get_rect().x + 20, cell.get_rect().y + 110))
            if cell.was_pollinated:
                GlobalParamsGame.SCREEN.blit(self.image, (cell.get_rect().x+70,cell.get_rect().y))


    def draw_rect_alpha(self, surface, color, rect):
        shape_surf = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
        pygame.draw.rect(shape_surf, color, shape_surf.get_rect())
        surface.blit(shape_surf, rect)

