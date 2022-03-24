import threading
import numpy as np

line = list(map(int, input().split()))
k = line[0]
m = line[1]
n = line[2]
neighbors = np.array([list(map(float, input().split())) for _ in range(m)], dtype=np.float64)
votes = list(map(int, input().split()))
for _ in range(n):
    sorted_indexes = np.argsort(
        np.sum(np.square(neighbors - np.array(list(map(float, input().split())), dtype=np.float64)), axis=1))
    votes_temp = np.zeros(m, dtype=np.int64)
    for sorted_index in sorted_indexes[:k]:
        votes_temp[votes[sorted_index]] += 1
    print(np.argmax(votes_temp), end=' ')
