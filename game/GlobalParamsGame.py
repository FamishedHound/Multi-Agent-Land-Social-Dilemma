import pygame


class GlobalParamsGame:

    BLACK = (0, 0, 0)
    WHITE = (229, 255, 204)
    WINDOW_HEIGHT = 600
    WINDOW_WIDTH = 600
    BLOCKSIZE = 40
    SCREEN = pygame.display.set_mode((WINDOW_HEIGHT, WINDOW_WIDTH))
    MAX_CELLS_NUMER = int(WINDOW_HEIGHT / BLOCKSIZE)

class GlobalParamsAi:
    NUMBER_OF_AGENTS = 20
    NUMBER_OF_RANDOM_POLLINATORS = 10

class GlobalEconomyParams:
    LAND_UPCOST =10
    STARTING_GOLD_MULTIPLIER =10000
    MAXIMAL_INCOME = 100
