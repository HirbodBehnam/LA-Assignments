import numpy as np
import sys

a = np.array(list(map(int, input().split())), dtype=np.int64)
b = np.array(list(map(int, input().split())), dtype=np.int64)
if len(b) < len(a):
    a, b = b, a
m = np.zeros((len(a) + len(b) - 1, len(b)), dtype=np.int64)
for i in range(len(b)):
    m[:,i][i:i+len(a)] = a
np.set_printoptions(threshold=sys.maxsize)
print(m @ b)