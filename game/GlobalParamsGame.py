import pygame


class GlobalParamsGame:

    BLACK = (0, 0, 0)
    WHITE = (229, 255, 204)
    WINDOW_HEIGHT = 600
    WINDOW_WIDTH = 600
    BLOCKSIZE = 20
    SCREEN = pygame.display.set_mode((WINDOW_HEIGHT, WINDOW_WIDTH))
    MAX_CELLS_NUMER = WINDOW_HEIGHT / BLOCKSIZE

class GlobalParamsAi:
    NUMBER_OF_AGENTS = 100
