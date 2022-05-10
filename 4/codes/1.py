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


n = int(input())
matrix = np.array([list(map(int, input().split()))
                  for _ in range(n)], dtype=np.float64)
matrix = np.round(inverse_matrix(matrix))
for i in range(n):
    for j in range(n):
        print(int(matrix[i][j]), end=' ')
    print()