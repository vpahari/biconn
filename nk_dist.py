import networkx as nx
import networkit as nk
from operator import itemgetter
import random

#import matplotlib.pyplot as plt

#N=int(sys.argv[1]) # number of nodes
#k=int(sys.argv[2])
#SEED=int(sys.argv[3])

N = 500

k = 4.0

SEED = 1099 

p = k / float(N-1)

step_size = 0.01

#random.seed(SEED)


def findAllDisjointPaths(G,s,t):
	distPaths = []

	uniqueNodes = [] 

	newDist = nk.distance.Dijkstra(G, s, True, True)

	newDist.run()

	hasPath = newDist.numberOfPaths(t)

	while hasPath != 0:
		
		shortestPath = newDist.getPath(t)

		for i in shortestPath[1:len(shortestPath) - 1]:
			G.removeNode(i)
			uniqueNodes.append(i)

		
		distPaths.append(shortestPath)
		
		newDist = nk.distance.Dijkstra(G, s, True, True)
		
		newDist.run()
		
		hasPath = newDist.numberOfPaths(t)

	print(distPaths)
		
	print(uniqueNodes)

	return (distPaths, uniqueNodes)


def takeOutNodes(G,sortedList):
	numNodesToRemove = int(step_size * N)
	print(numNodesToRemove)
	listToRemove = sortedList[:numNodesToRemove]
	for (node, num) in listToRemove:
		G.removeNode(node)


def findLargestNodes(uniqueNodes):
	keysValues = uniqueNodes.items()
	sortedKeysValues = sorted(keysValues, key = itemgetter(1), reverse = True)
	return sortedKeysValues



GER = nx.erdos_renyi_graph(N, p, seed = SEED)

biconn = list(nx.biconnected_component_subgraphs(GER))

G = max(biconn, key=len)

sizeOfNewList = G.number_of_nodes()

Gnk = nk.nxadapter.nx2nk(G)

newGraph = nk.graph.Graph(n = sizeOfNewList, weighted = False, directed = True)

lastEdges = Gnk.edges()

for (i,j) in lastEdges:
	newGraph.addEdge(i,j)
	newGraph.addEdge(j,i)

#newEdges = newGraph.edges()

listOfNodes = newGraph.nodes()

print(newGraph.isDirected())


lengthOfNodes = len(listOfNodes)

#print(newGraph.edges())

listOfAllSP = []
uniqueNodesDict = {}

lenBiconnList = []

lenBiconnList.append(lengthOfNodes)

for s in listOfNodes:

	print("0")

	print(s)

	print(listOfNodes)
	
	if s != listOfNodes[0]:
		break

	allSPforS = []

	dijk = nk.distance.Dijkstra(newGraph, s, True, True)
	
	dijk.run()


	for t in listOfNodes:

		if t == listOfNodes[0]:
			continue
		
		isPath = dijk.numberOfPaths(t)
		
		tempG = newGraph.copyNodes()
		
		for (e1,e2) in newGraph.edges():
			tempG.addEdge(e1,e2)

		if isPath != 0:
			
			shortestPath = dijk.getPath(t)
			
			allSPforS.append(shortestPath)

			if len(shortestPath) == 2:
				tempG.removeEdge(s,t)
				tempG.removeEdge(t,s)

			else:
				nodesToRemove = shortestPath[1:len(shortestPath)-1]
				
				for node in nodesToRemove:

					tempG.removeNode(node)

					if node in uniqueNodesDict:
						uniqueNodesDict[node] = uniqueNodesDict[node] + 1
					else:
						uniqueNodesDict[node] = 1
			
			(DPST, uniqueNodes) = findAllDisjointPaths(tempG,s,t)

			for distSP in DPST:
				allSPforS.append(distSP)
			
			for node in uniqueNodes:
				if node in uniqueNodesDict:
					uniqueNodesDict[node] = uniqueNodesDict[node] + 1
				else:
					uniqueNodesDict[node] = 1

		else:
			allSPforS.append([])

	listOfAllSP.append(allSPforS)

	sortedList = findLargestNodes(uniqueNodesDict)

	takeOutNodes(newGraph,sortedList)

	nxGraph = nk.nxadapter.nk2nx(newGraph)

	nxGraphUndirected = nx.to_undirected(nxGraph)

	biconn = list(nx.biconnected_component_subgraphs(nxGraphUndirected))

	G = max(biconn, key=len)

	Gnk = nk.nxadapter.nx2nk(G)

	sizeOfNewList = G.number_of_nodes()

	if sizeOfNewList == 0:
		break

	newGraph = nk.graph.Graph(n = sizeOfNewList, weighted = False, directed = True)

	lastEdges = Gnk.edges()

	for (i,j) in lastEdges:
		newGraph.addEdge(i,j)
		newGraph.addEdge(j,i)

	listOfNodes = newGraph.nodes()

	lengthOfNodes = len(listOfNodes)

	lenBiconnList.append(lengthOfNodes)


print("lenBiconnList")
print(lenBiconnList)

#print(len(listOfAllSP))
#a = findLargestNodes(uniqueNodesDict)
#print(len(a))
#print(a)

"""
for isss in (listOfAllSP):
	print(isss)

print(len(listOfAllSP))

print(uniqueNodesDict)
k = list(uniqueNodesDict.keys())
v = list(uniqueNodesDict.values())

k.sort()
v.sort()

print(k)
print(v)

print(newGraph.edges())
"""


