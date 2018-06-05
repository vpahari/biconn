#import random
#import sys
#import math
#from functools import reduce
#import csv
#from operator import itemgetter

import networkx as nx
import networkit as nk

#import matplotlib.pyplot as plt

N = 10

k = 4.0

p = k / float(N)

S = 1099 

G = nx.erdos_renyi_graph(N, p, seed = S)

print(len(G))

Gnk = nk.nxadapter.nx2nk(G)

print(Gnk)

print("Nodes")

listOfNodes = Gnk.nodes()

newGraph = nk.graph.Graph(n = N, weighted = False, directed = True)

lastEdges = Gnk.edges()

print(lastEdges)

for (i,j) in lastEdges:
	newGraph.addEdge(i,j)
	newGraph.addEdge(j,i)

newEdges = newGraph.edges()
print(newEdges)

print(newGraph.isDirected())


#newGraph.addEdge(j,i)

sp = nk.distance.AllSimplePaths(newGraph, 0, 5)

sp.run()

#print(sp)

allSP = sp.getAllSimplePaths()

counter = 0

firstSP = []

newNodes = []

listOfAllPaths = []

if len(allSP) != 0:	
	firstSP = allSP[0]
	counter = counter + 1
	listOfAllPaths.append(firstSP)
	newNodes = newNodes + firstSP[1:len(firstSP) - 1]

for i in range(1, len(allSP)):
	currSP = allSP[i]
	found = True
	for j in range(len(newNodes)):
		if newNodes[j] in currSP:
			found = False
			break
	if found:
		counter = counter + 1
		listOfAllPaths.append(currSP)
		newNodes = newNodes + currSP[1:len(currSP) - 1]


print(counter)
print(newNodes)
print(listOfAllPaths)


#b = sp.numberOfSimplePaths()

#print(a)

#print(b)

#print(a)
#for i in range(10):
#print(Gnk.nk.distance.getDistances())
	

