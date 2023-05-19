import tifffile
import numpy as np

from eden_project.TDA_category_1 import categoryLosses

data = tifffile.imread("2021/north_uk/data/LCM.tif")[0]

print(data.shape)
dict_colors = {
            255: [],
            0: [],
            1: [],
            2: [],
            3: [],
            4: [],
            5:[],
            6: [],
            7: [],
            8: [],
            9: [],
            10:[],
            11: [],
            12: [],
            13: [],
            14: [],
            15: [],
            16: [],
            17: [],
            18: [],
            19: [],
            20: [],
            21: []
        }
x_temp = 0
y_temp = 0
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
def pad_with_zeros(A, r, c):
   out = np.zeros((r, c))
   r_, c_ = np.shape(A)
   out[0:r_, 0:c_] = A
   return out
def preprocess_land_data(data):
    data = pad_with_zeros(data, 9600, 9600)
    boxes = blockshaped(data, 40, 40)
    print(boxes.shape)
    new_data = np.array([])
    for box in range(boxes.shape[0]):
        new_data = np.append(new_data, round(boxes[box].mean()))
    new_data = np.reshape(new_data, (240, 240))
    new_data = aggregate_types([15, 16, 17, 12], 18, new_data)  # Rocks and swamps
    new_data = aggregate_types([8], 11, new_data)  # swamps
    new_data = aggregate_types([5, 6, 7], 4, new_data)  # grassland
    new_data = aggregate_types([2], 1, new_data)  # woodland
    new_data = aggregate_types([9], 10, new_data)  # heathers
    new_data = aggregate_types([21], 20, new_data)  # Suburan => Urban
    new_data = aggregate_types([13, 19], 14, new_data)  # water
    print(np.unique(new_data))
    # new_data[new_data == 13] = 14 # Saltwater into water or freshwater
    # new_data[new_data==5] = 4 # Saltwater into water or freshwater
    # new_data[new_data==6] = 14 # Saltwater into water or freshwater
    # new_data[new_data == 13] = 14 # Saltwater into water or freshwater
    # new_data[np.logical_and(,,new_data==10)] = 4 # All types of grassland into grassland
    # new_data[np.logical_and(new_data==15,new_data==16,new_data==17,new_data==12)] = 18 # rocks into 1 type

    return new_data

def aggregate_types( past_type, new_type, new_data):
    for i in range(len(past_type)):
        new_data[new_data == past_type[i]] = new_type
    return new_data
# while x_temp != data.shape[0] and y_temp != data.shape[1]:
data = preprocess_land_data(data)
# for x in range(data.shape[0]):
#     for y in range(data.shape[1]):
#
#         for i in range(x-1,x+2):
#             for j in range(y-1,y+2):
#                 if i > 0 and i < data.shape[0] and j >0 and j < data.shape[1]:
#                     dist = np.linalg.norm(np.array((x, y)) - np.array((i, j)))
#                     if data[x,y] != data[i,j] and dist <2 and data[x,y]!=0 and data[i,j] != 0:
#                         print(f"{x} {y} append to {data[x,y]} {data[i,j]} dist {dist}")
#                         dict_colors[data[x,y]].append(data[i,j])


from collections import Counter
#Conditional probability
# for k,v in dict_colors.items():
#     dictionary = {}
#     for item in v:
#         dictionary[item] = dictionary.get(item, 0) + 1
#
#     print(f"for {k} {dictionary}")
loss = categoryLosses(data)
x = loss.find_specific_land_types(4)
print()