# -*- coding: utf-8 -*-

# Amir Afridi

from time import time
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from sklearn.neighbors import NearestNeighbors
import math
from sklearn.decomposition import PCA
Axes3D


def GetDistMatrix(nodes):
    i = 0
    arr = np.zeros((len(nodes), len(nodes)))
    for node1 in nodes:
        j = 0
        for node2 in nodes:
            if node1[0] == node2[0] and node1[1] == node2[1] and node1[2] == node2[2]:
                dist = 0
            else:
                dist = ((node1[0] - node2[0]) * (node1[0] - node2[0])) + ((node1[1] - node2[1])
                                                                          * (node1[1] - node2[1])) + ((node1[2] - node2[2]) * (node1[2] - node2[2]))
                dist = math.sqrt(dist)
            arr[i][j] = dist
            j = j + 1
        i = i + 1

    return arr


"""
Implement MDS

"""


def classical_MDS(D, n_components):
    Dsquared = np.zeros((len(D), len(D)))
    i = 0
    for d1 in D:
        j = 0
        for d2 in d1:
            d3 = d2 * d2
            Dsquared[i][j] = d3
            j = j + 1
        i = i + 1
    I = np.zeros((len(D), len(D)))
    for num in range(0, len(D)):
        I[num][num] = 1
    oneOnePrimeMatrix = np.zeros((len(D), len(D)))
    J = np.zeros((len(D), len(D)))
    negHalfJ = np.zeros((len(D), len(D)))
    for num1 in range(len(D)):
        for num2 in range(len(D)):
            oneOnePrimeMatrix[num1][num2] = 1 / len(D)
            J[num1][num2] = I[num1][num2] - oneOnePrimeMatrix[num1][num2]
            negHalfJ[num1][num2] = (-1 * J[num1][num2]) / 2

    '''
        firstDotProduct is the dot product of -.5J * P^2
        B = firstDotProduct * J to give the B matrix (double centering)
    '''
    firstDotProduct = np.dot(negHalfJ, Dsquared)
    B = np.dot(firstDotProduct, J)

    eigenvalues, eigenvectors = np.linalg.eigh(B)

    eigenvectors = eigenvectors.T

    zipped = list(zip(eigenvalues, eigenvectors))
    zipped.sort()
    dLargest = zipped[len(D) - n_components:]

    X = np.zeros((len(D), 2))
    for i in range(0, len(D)):
        X[i][1] = -dLargest[0][1][i]
        X[i][0] = dLargest[1][1][i]
    mult = np.zeros((2, 2))
    mult[1][1] = math.sqrt(dLargest[0][0])
    mult[0][0] = math.sqrt(dLargest[1][0])
    points = X.dot(mult)

    return points


"""
Implement ISOMAP

"""


def ISOMAP(listOfPoints, d, K):
    # DistMatrix is the distance matrix
    nbrs = NearestNeighbors(n_neighbors=K).fit(listOfPoints)
    knnMatrix = nbrs.kneighbors_graph(listOfPoints).toarray()
    tempDistMatrix = GetDistMatrix(listOfPoints)

    G = nx.Graph()
    for i in range(0, len(listOfPoints)):
        G.add_node(i)
    for i in range(0, len(listOfPoints)):
        for j in range(0, len(listOfPoints)):
            if knnMatrix[i][j] == 1:
                G.add_edge(i, j, weight=tempDistMatrix[i][j])

    distMatrix = nx.floyd_warshall_numpy(G)
    distMatrixAsArray = np.squeeze(np.asarray(distMatrix))
    G2 = classical_MDS(distMatrixAsArray, d)

    return G2


n_points = 1000
X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
n_neighbors = 10
n_components = 2

fig = plt.figure(figsize=(15, 8))
plt.suptitle("Manifold Learning with %i points, %i neighbors" %
             (1000, n_neighbors), fontsize=14)
ax = fig.add_subplot(251, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

distMatrix = GetDistMatrix(X)

t0 = time()
G1 = classical_MDS(distMatrix, n_components)
t1 = time()
print("MDS: %.2g sec" % (t1 - t0))

#pos = nx.get_node_attributes(G1, 'pos')

fig = plt.figure()
plt.suptitle("MDS", fontsize=14)
plt.scatter(G1[:, 0], G1[:, 1], c=color, cmap=plt.cm.Spectral)
plt.show()

t0 = time()
G2 = ISOMAP(X, n_components, n_neighbors)
t1 = time()
print("ISOMAP: %.2g sec" % (t1 - t0))
fig = plt.figure()
plt.suptitle("ISOMAP", fontsize=14)
plt.scatter(G2[:, 0], G2[:, 1], c=color, cmap=plt.cm.Spectral)
plt.show()


"""
Use the data "S" with PCA and, project and plot the data.

"""
pca = PCA(n_components=2)
Xpca = pca.fit_transform(X)
plt.suptitle("PCA")
plt.scatter(Xpca[:, 0], Xpca[:, 1], c=color, cmap=plt.cm.Spectral)
plt.show()

