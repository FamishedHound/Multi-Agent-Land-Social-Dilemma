import pygame


class GlobalParamsGame:

    BLACK = (0, 0, 0)
    WHITE = (229, 255, 204)
    WINDOW_HEIGHT = 600
    WINDOW_WIDTH = 600
    BLOCKSIZE = 120
    SCREEN = pygame.display.set_mode((WINDOW_HEIGHT, WINDOW_WIDTH))
    MAX_CELLS_NUMER = int(WINDOW_HEIGHT / BLOCKSIZE)

class GlobalParamsAi:
    NUMBER_OF_AGENTS = 9
    NUMBER_OF_RANDOM_POLLINATORS = 1

class GlobalEconomyParams:
    LAND_UPCOST = 25
    STARTING_GOLD =500
    MAXIMAL_INCOME = 100
