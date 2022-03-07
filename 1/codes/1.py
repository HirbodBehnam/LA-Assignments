import numpy as np
from enum import Enum


class Orientation(Enum):
    VERTICAL = 1
    HORIZONTAL = 2

    @staticmethod
    def swap(first):
        if first == Orientation.VERTICAL:
            return Orientation.HORIZONTAL
        return Orientation.VERTICAL


def numbers_to_ints(read_line):
    return list(map(int, read_line.strip().split()))


a = np.ones(int(input()), dtype=np.int64)
q = int(input())
a_orientation = Orientation.VERTICAL

for _ in range(q):
    line = input()
    if line == "T":
        a_orientation = Orientation.swap(a_orientation)
    elif line.startswith("dot"):
        numbers = numbers_to_ints(line[3:])
        if len(numbers) == len(a) and a_orientation == Orientation.HORIZONTAL:
            print(np.dot(a, numbers))
    elif line.startswith("out"):
        if a_orientation == Orientation.HORIZONTAL:
            continue
        split_line = line[3:].split(',')
        numbers = numbers_to_ints(split_line[0])
        k = int(split_line[1].strip())
        if k > len(a):
            continue
        m = int(a[k - 1])
        a = np.multiply(numbers, m)
        a_orientation = Orientation.HORIZONTAL
    elif line.startswith("cross"):
        if len(a) != 3:
            continue
        numbers = numbers_to_ints(line[5:])
        if len(numbers) != 3:
            continue
        res = np.cross(a, numbers)
        res = res / np.linalg.norm(res)
        print(f'{res[0]:.4f} {res[1]:.4f} {res[2]:.4f}')
    elif line.startswith("had"):
        numbers = numbers_to_ints(line[3:])
        if len(numbers) == len(a):
            a = np.multiply(a, numbers)
    elif line == "print":
        if a_orientation == Orientation.HORIZONTAL:
            print(' '.join(map(str, a)))
        else:
            for i in a:
                print(i)
    elif line.startswith("reset"):
        a = np.ones(int(line[5:].strip()), dtype=np.int64)
