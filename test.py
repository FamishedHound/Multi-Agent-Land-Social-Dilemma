import math

# import numpy as np
# arr = np.array([[[1,2,3],[3,5,3]],[[1,2,3],[3,5,3]]])
# stack = np.stack(arr)
# print(arr)

#
import numpy as np
good = [[[1]], [[0.2]], [[0.2]]]


print([100,0,0,0] in [0,0,100,0])
list1 = [100,0,0,0]
list2  = [0,0,100,0]
print(all(elem in list1  for elem in list2))