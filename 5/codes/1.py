import numpy as np
import numpy.linalg as linalg

def eig_vals(A: np.matrix , iter : int) -> np.array:
    X = A
    for _ in range(iter):
        q, r = linalg.qr(X)
        X = r @ q
    return np.diag(X)

A = list(map(lambda x: x.split(", "), input().split(";")))
eigs = -np.sort(-eig_vals(np.matrix(A, dtype=np.float64), 1000))
for eig in eigs:
    print("{:.2f}".format(eig), end=' ')