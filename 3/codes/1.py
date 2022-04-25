import numpy as np


def inverse_matrix(a: np.ndarray) -> np.ndarray:
    a = np.copy(a)  # Make a copy
    n = a.shape[0]
    result = np.identity(n)
    for i in range(n):
        if abs(a[i][i]) <= 10**(-6):
            # Try to swap the rows
            ok = False
            for j in range(i+1, n):
                if abs(a[j][i]) >= 10**(-6):
                    ok = True
                    a[[i, j]] = a[[j, i]]
                    result[[i, j]] = result[[j, i]]
                    break
            if not ok:
                raise ZeroDivisionError()
        for j in range(i+1, n):
            ratio = a[j][i]/a[i][i]
            a[j] -= ratio * a[i]
            result[j] -= ratio * result[i]
    # Inverse to make the top zero
    for i in range(n-1, -1, -1):
        for j in range(i-1, -1, -1):
            ratio = a[j][i]/a[i][i]
            a[j] -= ratio * a[i]
            result[j] -= ratio * result[i]
    # Fix the diag
    for i in range(n):
        result[i] /= a[i][i]
    return result


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
    #cached_inverse = np.array(np.matrix(space[:np.shape(space)[1]]).I)
    cached_inverse = inverse_matrix(space[:np.shape(space)[1]])
except ZeroDivisionError:  # Non inversable matrix so we are fine
    for _ in range(args[2]):
        print("YES")
    exit()
# Check each point
for _ in range(args[2]):
    points = np.array(list(map(float, input().split())) +
                      [1], dtype=np.float64)
    ans = cached_inverse @ points[:np.shape(space)[1]]
    if np.all(np.abs(np.dot(space, ans) - points) < 10**(-8)):
        print("YES")
    else:
        print("NO")
