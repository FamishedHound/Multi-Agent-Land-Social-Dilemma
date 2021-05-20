import pygame

from game.GlobalParamsGame import GlobalParamsGame


class LandCell:
    def __init__(self, x, y):
        self.rect = pygame.Rect(x * GlobalParamsGame.BLOCKSIZE, y * GlobalParamsGame.BLOCKSIZE,
                                GlobalParamsGame.BLOCKSIZE, GlobalParamsGame.BLOCKSIZE)
        self.is_owned = False
        self.owner = None
        self.x = x
        self.y = y
        self.is_pollinator = False
        self.bag_pointer_declared = 0
        self.bag_pointer_actual = 0
        self.was_pollinated = False
    def get_rect(self):
        return self.rect

    def set_owner(self, owner):
        self.owner = owner
        owner.no_already_assigned_lands+=1

    def set_owned(self, is_owned: bool):
        self.is_owned = is_owned

    def euclidian_distance(self):
        pass
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
