# import osmnx as ox
# ox.plot_graph(ox.graph_from_place('morecambe, Uk'))
import gc
import sys
from typing import Optional, Union, List

from ray.thirdparty_files import psutil
from scipy.spatial import distance
import gym
import numpy as np
import pygame
from PIL import Image
# im = Image.open('2021/data_10m/data/LCM.tif')
# # im.show()
# im = np.array(im)
# print(im)
from gym import spaces
from ray.rllib.algorithms.apex_dqn.apex_dqn import ApexDQNConfig
import matplotlib.pyplot as plt
# import pyvips
# x = pyvips.Image.new_from_file("2021/data_10m/data/LCM.tif")
import tifffile

import ray
from ray.tune import register_env


# print(img)
# plt.imshow(img[0])
# plt.show()

dict_colors = {
    255: (255, 255, 255),
    0: (255, 255, 255),
    1: (255, 0, 0),
    2: (0, 102, 0),
    3: (115, 38, 0),
    4: (0, 255, 0),
    5: (127, 229, 127),
    6: (112, 168, 0),
    7: (153, 129, 0),
    8: (255, 255, 0),
    9: (128, 26, 128),
    10: (230, 140, 166),
    11: (0, 128, 115),
    12: (210, 210, 255),
    13: (0, 0, 128),
    14: (0, 0, 255),
    15: (204, 179, 0),
    16: (204, 179, 0),
    17: (255, 255, 128),
    18: (255, 255, 128),
    19: (128, 128, 255),
    20: (255, 224, 32),
    21: (128, 128, 128)

}
import scipy.stats as stats
from scipy.spatial.distance import cdist
from scipy.spatial.distance import hamming
def env_creator(env_config):
    return PredictionMorecabmreEnv(env_config)
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))
# main()
def pad_with_zeros(A, r, c):
   out = np.zeros((r, c))
   r_, c_ = np.shape(A)
   out[0:r_, 0:c_] = A
   return out
def create_config():
    config = ApexDQNConfig()
    register_env("MORECAMBRE", env_creator)
    config.framework("torch")
    config.resources(num_gpus=1)
    config.num_rollout_workers=5
    config.environment("MORECAMBRE")
    config.replay_buffer_config["capacity"] = 600000
    return config
class PredictionMorecabmreEnv(gym.Env):

    def __init__(self,env_config):

        # self.data = tifffile.imread("2021/north_uk/data/LCM.tif")[0][tifffile.imread("2021/north_uk/data/LCM.tif")[0] != 0].astype(np.float32)
        self.data = tifffile.imread("2021/north_uk/data/LCM.tif")[0]
        print(f"self data {self.data.shape}")
        m, n = 240, 240
        self.data = self.preprocess_land_data()
        print(f"self data {self.data.shape}")
        register_env("MORECAMBRE",env_creator)
        self.observation_space =   spaces.Box(
        low=0, high=21, shape= self.data.shape,dtype=np.float32)
        # self.action_space = spaces.Sequence(spaces.Box(
        # low=0, high=21, shape= self.data.shape,dtype=np.float32))
        self.action_space = spaces.Discrete(21)
        surface = pygame.display.set_mode((800, 800))
        self.dict_colors = {
            255: (255, 255, 255),
            0: (255, 255, 255),
            1: (255, 0, 0),
            2: (0, 102, 0),
            3: (115, 38, 0),
            4: (0, 255, 0),
            5: (127, 229, 127),
            6: (112, 168, 0),
            7: (153, 129, 0),
            8: (255, 255, 0),
            9: (128, 26, 128),
            10: (230, 140, 166),
            11: (0, 128, 115),
            12: (210, 210, 255),
            13: (0, 0, 128),
            14: (0, 0, 255),
            15: (204, 179, 0),
            16: (204, 179, 0),
            17: (255, 255, 128),
            18: (255, 255, 128),
            19: (128, 128, 255),
            20: (255, 224, 32),
            21: (128, 128, 128)
        }
        self.obs = np.ones(shape=self.data.shape).astype(np.float32)

        self.x = 0
        self.y =0
        self.done = False
    def aggregate_types(self,past_type,new_type,new_data):
        for i in range(len(past_type)):
            new_data[new_data==past_type[i]] = new_type
        return new_data
    def preprocess_land_data(self):
        self.data = pad_with_zeros(self.data, 9600, 9600)
        boxes = blockshaped(self.data, 40, 40)
        print(boxes.shape)
        new_data = np.array([])
        for box in range(boxes.shape[0]):
            new_data = np.append(new_data, round(boxes[box].mean()))
        new_data = np.reshape(new_data, (240, 240))
        new_data =self.aggregate_types([15,16,17,12],18,new_data) #Rocks and swamps
        new_data =self.aggregate_types([8],11,new_data) # swamps
        new_data =self.aggregate_types([5,6,7],4,new_data) #grassland
        new_data =self.aggregate_types([2],1,new_data) #woodland
        new_data =self.aggregate_types([9],10,new_data) # heathers
        new_data =self.aggregate_types([21],20,new_data) # Suburan => Urban
        new_data =self.aggregate_types([13,19],14,new_data) # water
        print(np.unique(new_data))
        # new_data[new_data == 13] = 14 # Saltwater into water or freshwater
        # new_data[new_data==5] = 4 # Saltwater into water or freshwater
        # new_data[new_data==6] = 14 # Saltwater into water or freshwater
        # new_data[new_data == 13] = 14 # Saltwater into water or freshwater
        # new_data[np.logical_and(,,new_data==10)] = 4 # All types of grassland into grassland
        # new_data[np.logical_and(new_data==15,new_data==16,new_data==17,new_data==12)] = 18 # rocks into 1 type

        return new_data

    def step(self, action):
        if action ==0:
            action=1
        self.obs[self.x] = action
        self.x += 1
        # done  = False


        if self.x == len(self.data):
            self.done = True
            # self.render()
        # aflat = np.hstack(self.data)
        # bflat = np.hstack(self.obs)
        # reward  = np.linalg.norm(self.data-self.obs)
        # reward  = hamming(self.data, self.obs)

        self.auto_garbage_collect()
        tau, p_value = stats.kendalltau(self.data, self.obs)
        reward = tau
        print(f"reward : {reward}  action : {action} {type(reward)}")
        print(self.x,self.y)
        return self.obs,reward,self.done,{}
    def _get_reward(self):
        return 0
    def render(self, mode="human"):
        self.drawGrid()
        pygame.display.update()

    def reset(self):
        self.obs = np.zeros(shape=(self.data.shape)).astype(np.float32)
        self.done = False
        self.x = 0

        return self.obs
    def render(self):
        pass

    def auto_garbage_collect(self,pct=80.0):
        """
        auto_garbage_collection - Call the garbage collection if memory used is greater than 80% of total available memory.
                                  This is called to deal with an issue in Ray not freeing up used memory.

            pct - Default value of 80%.  Amount of memory in use that triggers the garbage collection call.
        """
        if psutil.virtual_memory().percent >= pct:
            gc.collect()
        return
    def drawGrid(self,mode):
        blockSize = 20  # Set the size of the grid block
        for x in range(0, 196):
            for y in range(0, 229):
                rect = pygame.Rect(x * 3, y * 3, blockSize, blockSize)
                s = pygame.display.get_surface()
                if mode =="prediction":
                    s.fill(tuple(dict_colors[self.obs[x, y]]), rect)
                else:
                    s.fill(tuple(dict_colors[self.data[x, y]]), rect)

# PredictionMorecabmreEnv({}).step(5)
z = tifffile.imread("2021/data/LCM.tif")[0]
import torch
# print()
print(torch.cuda.is_available())
ray.init(num_gpus=1,num_cpus=10)
x = PredictionMorecabmreEnv({})
while True:
    # action = x.action_space.sample()
    # x.step(action)
    x.drawGrid("abc")
    pygame.display.flip()
# algo = create_config().build()
# x.reset()
# for i in range(50000):
#     algo.train()

