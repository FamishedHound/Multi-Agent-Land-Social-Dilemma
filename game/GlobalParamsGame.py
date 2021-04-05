import pygame


class GlobalParamsGame:

    BLACK = (0, 0, 0)
    WHITE = (229, 255, 204)
    WINDOW_HEIGHT = 800
    WINDOW_WIDTH = 800
    BLOCKSIZE = 40
    SCREEN = pygame.display.set_mode((WINDOW_HEIGHT, WINDOW_WIDTH))
    MAX_CELLS_NUMER = int(WINDOW_HEIGHT / BLOCKSIZE)

class GlobalParamsAi:
    NUMBER_OF_AGENTS = 30
    NUMBER_OF_RANDOM_POLLINATORS = 10

class GlobalEconomyParams:
    LAND_UPCOST = 50
    STARTING_GOLD_MULTIPLIER = 100
