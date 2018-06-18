import networkx as nx
import networkit as nk
from operator import itemgetter
import random

#import matplotlib.pyplot as plt

#N=int(sys.argv[1]) # number of nodes
#k=int(sys.argv[2])
#SEED=int(sys.argv[3])

N = 120

k = 4.0

SEED = 1099 

p = k / float(N-1)

step_size = 0.01

#random.seed(SEED)


def findAllDisjointPaths(G,s,t,shortestPath):
	distPaths = []

	distPaths.append(shortestPath)

	if len(shortestPath) == 2:
		G.removeEdge(s,t)
		G.removeEdge(t,s)

	else:
		nodesToRemove = shortestPath[1:len(shortestPath)-1]

		for node in nodesToRemove:
			tempG.removeNode(node)


	newDist = nk.distance.Dijkstra(G, s, True, False)

	newDist.run()

	hasPath = newDist.numberOfPaths(t)

	while hasPath != 0:
		
		shortestPath = newDist.getPath(t)

		for i in shortestPath[1:len(shortestPath) - 1]:
			G.removeNode(i)

		
		distPaths.append(shortestPath)
		
		newDist = nk.distance.Dijkstra(G, s, True, True)
		
		newDist.run()
		
		hasPath = newDist.numberOfPaths(t)

	print(distPaths)
		
	return distPaths


def takeOutNodes(G,sortedList):
	numNodesToRemove = int(step_size * N)
	listToRemove = sortedList[:numNodesToRemove]
	print(listToRemove)
	for (node, num) in listToRemove:
		G.removeNode(node)


def findLargestNodes(uniqueNodes):
	keysValues = uniqueNodes.items()
	sortedKeysValues = sorted(keysValues, key = itemgetter(1), reverse = True)
	return sortedKeysValues



#GER = nx.read_gpickle("sageGraph.gpickle")

GER = nx.erdos_renyi_graph(N,p,SEED)

print(type(GER))

#print(GER.edges())
#print(GER.nodes())

biconn = list(nx.biconnected_component_subgraphs(GER))

G = max(biconn, key=len)

print(len(G))

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


lenBiconnList = []

lenBiconnList.append(lengthOfNodes)

percentageSample = 0.01

num_Node_Sample = int(percentageSample * lengthOfNodes)

print(num_Node_Sample)

node_Sample = random.sample(listOfNodes, num_Node_Sample)

print(node_Sample)

number_of_DP_List = []

DP_List = []

for s in node_Sample:

	number_of_DP_List_s = []

	DP_List_s = []

	dijk = nk.distance.Dijkstra(newGraph, s, True, False)
	
	dijk.run()

	for t in listOfNodes:

		if t == s:
			DP_List_s.append([])
			number_of_DP_List_s.append(0)
			continue
		
		isPath = dijk.numberOfPaths(t)
		
		tempG = newGraph.copyNodes()
		
		for (e1,e2) in newGraph.edges():
			tempG.addEdge(e1,e2)

		if isPath != 0:

			shortestPath = dijk.getPath(t)
			
			DP = findAllDisjointPaths(tempG,s,t, shortestPath)
			number_of_DP = len(DP)

			DP_List_s.append(DP)
			number_of_DP_List_s.append(number_of_DP)

	DP_List.append(DP_List_s)
	number_of_DP_List.append(number_of_DP_List_s)

print("DP")
print(DP_List)

print("number_of_DP")
print(number_of_DP_List)

#print(len(listOfAllSP))
#a = findLargestNodes(uniqueNodesDict)
#print(len(a))
#print(a)


