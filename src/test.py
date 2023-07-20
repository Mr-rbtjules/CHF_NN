import numpy as np

m = [6,7]
arr = np.array(m)
a = {1:arr}

arr2 = arr
arr2[1] = 0

print(a[1])
a[1][1] = 4
print(a)
print(arr, arr2)
