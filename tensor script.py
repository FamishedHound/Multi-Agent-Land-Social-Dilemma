import numpy as np
import cv
tensor_size = 84
def picture_to_map(tensor):
    height, width = image.shape
    tiles = image.reshape((GRID_HEIGHT, height / GRID_HEIGHT,
                           GRID_WIDTH, width / GRID_WIDTH)).swapaxes(1, 2)