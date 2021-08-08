import pygame


class GlobalParamsGame:

    BLACK = (0, 0, 0)
    WHITE = (229, 255, 204)
    WINDOW_HEIGHT = 900
    WINDOW_WIDTH = 900
    BLOCKSIZE = 300
    SCREEN = pygame.display.set_mode((WINDOW_HEIGHT, WINDOW_WIDTH))
    MAX_CELLS_NUMER = int(WINDOW_HEIGHT / BLOCKSIZE)

class GlobalParamsAi:
    NUMBER_OF_AGENTS = 2
    NUMBER_OF_RANDOM_POLLINATORS = 1

class GlobalEconomyParams:
    LAND_UPCOST = 25
    STARTING_GOLD =10000
    MAXIMAL_INCOME = 100
