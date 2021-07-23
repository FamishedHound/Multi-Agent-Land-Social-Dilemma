import math

# import numpy as np
# arr = np.array([[[1,2,3],[3,5,3]],[[1,2,3],[3,5,3]]])
# stack = np.stack(arr)
# print(arr)

#
import numpy as np
good = [[[1]], [[0.2]], [[0.2]]]
bad = [[[1]], [[0.6]], [[0.6]]]

print(np.array(good).reshape((10,3,1,1)))