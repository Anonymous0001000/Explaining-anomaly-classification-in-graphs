import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

G = nx.Graph()
with open('toto_0.edgeList') as f:
    for i, line in enumerate(f):
        line = line.split()
        u = int(line[0])
        v = int(line[1])
        G.add_edge(u, v)

communities = [-1 for i in range(G.number_of_nodes())]
with open('toto_0.community') as f:
    for i, line in enumerate(f):
        line = line.split()
        node = int(line[0])
        comm = int(line[1])
        communities[node] = comm

number_of_attributes = 2
attributes = [[0 for i in range(number_of_attributes)] for i in range(G.number_of_nodes())]
with open('toto_0.attributes') as f:
    for i, line in enumerate(f):
        line = line.split()
        node = int(line[0])
        features = line[1].split(',')
        attributes[node] = [float(features[0]), float(features[1])]

x = [attributes[i][0] for i in range(G.number_of_nodes())]
y = [attributes[i][1] for i in range(G.number_of_nodes())]

plt.scatter(x, y, marker = '+', c = communities)
plt.show()
