import networkx as nx
#import networkit as nk
from operator import itemgetter
import random

#import matplotlib.pyplot as plt

#N=int(sys.argv[1]) # number of nodes
#k=int(sys.argv[2])
#SEED=int(sys.argv[3])

N = 5000

k = 3.0

SEED = 1099 

p = k / float(N-1)

step_size = 0.01

numNodesToRemove = int(step_size * N)

#random.seed(SEED)


def findArt(G):

	listOfNodes = G.nodes()

	biconnSizeList = [] 

	for i in listOfNodes:

		newG = G.copy()

		newG.remove_node(i)

		newBiconn = list(nx.biconnected_component_subgraphs(newG))

		largestBiconn = 0

		if len(newBiconn) > 0: 

			largestBiconn = len(max(newBiconn, key = len))

		biconnSizeList.append((largestBiconn,i))

	return biconnSizeList




GER = nx.erdos_renyi_graph(N, p, seed = SEED)

biconn = list(nx.biconnected_component_subgraphs(GER))

biconnToCall = max(biconn, key = len)

for i in range(10):

	a = findArt(biconnToCall)

	print(len(biconnToCall))

	a.sort()

	#print(a)

	b = a[:numNodesToRemove]

	nodesToRemove = []

	for (i,j) in b:
		nodesToRemove.append(j)


	#print(nodesToRemove)

	biconnToCall.remove_nodes_from(nodesToRemove)

	biconn = list(nx.biconnected_component_subgraphs(biconnToCall))

	biconnToCall = max(biconn, key = len)

	#print(len(biconnToCall))

