import math

# import numpy as np
# arr = np.array([[[1,2,3],[3,5,3]],[[1,2,3],[3,5,3]]])
# stack = np.stack(arr)
# print(arr)

#
import numpy as np
arr = np.array([[[1,2,3],[5,6,7]],[[6,7,6],[6,6,6]]])
arr.reshape()
print(arr.shape)
for x in arr:
    print(x)
    for y in x :
        print(y)