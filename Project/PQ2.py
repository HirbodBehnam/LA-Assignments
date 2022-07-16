import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
def show_graph(adjacency_matrix, labels=None, node_size=500):
    color_map = {1: 'blue', 2: 'green', 3: 'red', 4: 'yellow'}
    colors = [color_map[x] for x in labels] if labels is not None else None
        
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=node_size, node_color=np.array(colors)[list(gr.nodes)] if labels is not None else None)
    plt.show()

adjacency_matrix = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0],
                         [1, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 1, 1, 0, 0, 0, 0],
                         [1, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 1, 0, 1, 1],
                         [0, 0, 0, 0, 1, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 1, 1],
                         [0, 0, 0, 0, 1, 0, 1, 0, 1],
                         [0, 0, 0, 0, 1, 0, 1, 1, 0]])
show_graph(adjacency_matrix)

Sum = np.sum(adjacency_matrix, axis=0)
Lapl = np.diag(Sum) - adjacency_matrix

# https://stackoverflow.com/a/8093043
eigenvalues, eigenvectors = np.linalg.eig(Lapl)
eigenvalues[abs(eigenvalues) < 10e-12] = 0
eigenvectors[abs(eigenvectors) < 10e-12] = 0
idx = eigenvalues.argsort()   
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]
del(idx)
print(eigenvalues)
print(eigenvectors)

# Second one is not zero
eigenvector_for_graph = eigenvectors[:,1]
print("Elements in first part:", [(i+1) for i in range(len(eigenvector_for_graph)) if eigenvector_for_graph[i] > 0])
print("Elements in second part:", [(i+1) for i in range(len(eigenvector_for_graph)) if eigenvector_for_graph[i] < 0])

show_graph(adjacency_matrix, labels=list(map(lambda x : 1 if x > 0 else 2, eigenvector_for_graph)))

def two_eigenvector_classify(a: float, b: float) -> int:
    if a > 0 and b > 0:
        return 1
    elif a > 0 and b < 0:
        return 2
    elif a < 0 and b > 0:
        return 3
    else:
        return 4

groups = [two_eigenvector_classify(eigenvectors[i,1], eigenvectors[i,2]) for i in range(len(eigenvector_for_graph))]
show_graph(adjacency_matrix, labels=groups)

# Main graph
adjacency_matrix = np.zeros((100, 100))
file1 = open('data/data.txt', 'r')
lines = file1.readlines()[1:] # not first line
print(len(lines))
for l in lines:
    i, j = l.split()
    adjacency_matrix[int(i) - 1, int(j) - 1] = 1
    adjacency_matrix[int(j) - 1, int(i) - 1] = 1
show_graph(adjacency_matrix)

# Find eigenvalues
Sum = np.sum(adjacency_matrix, axis=0)
Lapl = np.diag(Sum) - adjacency_matrix
eigenvalues, eigenvectors = np.linalg.eig(Lapl)
eigenvalues[abs(eigenvalues) < 10e-12] = 0
eigenvectors[abs(eigenvectors) < 10e-12] = 0
idx = eigenvalues.argsort()   
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:,idx]
del(idx)
show_graph(adjacency_matrix, labels=list(map(lambda x : 1 if x > 0 else 2, eigenvectors[:,1])))
groups = [two_eigenvector_classify(eigenvectors[i,1], eigenvectors[i,2]) for i in range(len(eigenvectors[:,1]))]
show_graph(adjacency_matrix, labels=groups)
