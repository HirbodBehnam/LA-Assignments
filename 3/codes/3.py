import numpy as np
import sys

n = int(input())
points = np.array([list(map(float, input().split()))
                  for _ in range(n)], dtype=np.float64)
rotate_angels = list(map(float, input().split()))
a = np.array([
    [np.cos(rotate_angels[2]), -np.sin(rotate_angels[2]), 0],
    [np.sin(rotate_angels[2]), np.cos(rotate_angels[2]), 0],
    [0, 0, 1]
], dtype=np.float64)
b = np.array([
    [np.cos(rotate_angels[1]), 0, np.sin(rotate_angels[1])],
    [0, 1, 0],
    [-np.sin(rotate_angels[1]), 0, np.cos(rotate_angels[1])]
], dtype=np.float64)
c = np.array([
    [1, 0, 0],
    [0, np.cos(rotate_angels[0]), -np.sin(rotate_angels[0])],
    [0, np.sin(rotate_angels[0]), np.cos(rotate_angels[0])]
], dtype=np.float64)
for i in range(n):
    points[i] = a @ (b @ (c @ points[i]))
np.set_printoptions(
    formatter={'float': '{:.1f}'.format}, threshold=sys.maxsize)
print(points)
