import numpy as np


class WordIndexer:
    def __init__(self):
        self.counter = 0
        self.dict_of_words = {}

    def add(self, word: str):
        if word not in self.dict_of_words:
            self.dict_of_words[word] = self.counter
            self.counter += 1


words = WordIndexer()
lines = []
vectors = []
# Get all inputs
for _ in range(int(input())):
    line_words = input().split()
    lines.append(line_words)
    for word in line_words:
        words.add(word)
# Process them all
for line_words in lines:
    vec = np.zeros(len(words.dict_of_words))
    for word in line_words:
        vec[words.dict_of_words[word]] += 1
    vectors.append(vec)
# Now get the result
for i, first in enumerate(vectors):
    best = None
    best_index = None
    for j, second in enumerate(vectors):
        if j == i:  # Bruh
            continue
        c = np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second))
        if best is None or c > best:
            best = c
            best_index = j
    print(best_index + 1)
