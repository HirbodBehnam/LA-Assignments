import numpy as np

def create_matrix(a: np.ndarray) -> np.ndarray:
    result = np.identity(len(a) + 1)
    result[len(a), len(a)] = 0
    result[len(a), :len(a)] = a
    result[:len(a), len(a)] = a
    return result

source = np.array([1,2,3,4])
m = np.matrix(create_matrix(source))
print(m)
print(m.I)
