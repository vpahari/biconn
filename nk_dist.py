import networkx as nx
import networkit as nk
from operator import itemgetter

#import matplotlib.pyplot as plt

#N=int(sys.argv[1]) # number of nodes
#k=int(sys.argv[2])
#SEED=int(sys.argv[3])

step_size = 0.01

random.seed(SEED)

def findLargestNodes(uniqueNodes):
	keysValues = uniqueNodes.items()
	sortedKeysValues = sorted(keysValues, key = itemgetter(1), reverse = True)
	return sortedKeysValues


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


N = 10000

k = 4.0

S = 1099 

p = k / float(N-1)

G = nx.erdos_renyi_graph(N, p, seed = SEED)

Gnk = nk.nxadapter.nx2nk(G)

listOfNodes = Gnk.nodes()

newGraph = nk.graph.Graph(n = N, weighted = False, directed = True)

lastEdges = Gnk.edges()

for (i,j) in lastEdges:
	newGraph.addEdge(i,j)
	newGraph.addEdge(j,i)

newEdges = newGraph.edges()

print(newGraph.isDirected())


lengthOfNodes = len(listOfNodes)

#print(newGraph.edges())

listOfAllSP = []
uniqueNodesDict = {}

for s in range(N - 1):

	if s != 0:
		break

	allSPforS = []

	dijk = nk.distance.Dijkstra(newGraph, s, True, True)
	
	dijk.run()


	for t in range(s+1,N):
		
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

print(len(listOfAllSP))
a = findLargestNodes(uniqueNodesDict)
print(len(a))
print(a)

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


