import numpy as np

args = list(map(int, input().split()))
space = np.vstack([np.transpose(np.array([list(map(float, input().split()))
                  for _ in range(args[1])], dtype=np.float64)), np.ones(args[1])])
# Fast case: There are free variables
if np.shape(space)[0] < np.shape(space)[1]:
    for _ in range(args[2]):
        print("YES")
    exit()
# Inverse the matrix to calculate the results
try:
    cached_inverse = np.array(np.matrix(space[:np.shape(space)[1]]).I)
except: # Non inversable matrix so we are fine
    for _ in range(args[2]):
        print("YES")
    exit()
# Check each point
for _ in range(args[2]):
    points = np.array(list(map(float, input().split())) + [1], dtype=np.float64)
    ans = cached_inverse @ points[:np.shape(space)[1]]
    if np.all(np.abs(np.dot(space, ans) - points) < 10**(-8)):
        print("YES")
    else:
        print("NO")