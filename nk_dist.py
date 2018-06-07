import networkx as nx
import networkit as nk

#import matplotlib.pyplot as plt

def findAllDisjointPaths(G,s,t):
	distPaths = []

	uniqueNodes = [] 

	newDist = nk.distance.Dijkstra(G, s, True, True)

	newDist.run()

	hasPath = newDist.numberOfPaths(t)

	while hasPath != 0:

		shortestPath = dijk.getPath(t)

		for i in shortestPath[1:len(shortestPath) - 1]:
			G.removeNode(i)
			uniqueNodes.append(i)

		distPaths.append(shortestPath)

		newDist = nk.distance.Dijkstra(G, s, True, True)

		hasPath = newDist.numberOfPaths(t)

	return (distPaths, uniqueNodes)
	


N = 30

k = 2.0

p = k / float(N-1)

S = 1099 

G = nx.erdos_renyi_graph(N, p, seed = S)

Gnk = nk.nxadapter.nx2nk(G)

print("Nodes")

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

	allSPforS = []

	dijk = nk.distance.Dijkstra(newGraph, 0, True, True)
	print(0)
	dijk.run()
	print(1)

	for t in range(s+1,N):
		print(2)
		isPath = dijk.numberOfPaths(t)
		print(3)
		tempG = newGraph.copyNodes()
		print(4)
		for (e1,e2) in newGraph.edges():
			tempG.addEdge(e1,e2)

		if isPath != 0:
			print(5)
			shortestPath = dijk.getPath(t)
			allSPforS.append(shortestPath)
			print(6)
			nodesToRemove = shortestPath[1:len(shortestPath)-1]
			print(7)
			for node in nodesToRemove:
				tempG.removeNode(node)
				if node in uniqueNodesDict:
					uniqueNodesDict[node] = uniqueNodesDict[node] + 1
				else:
					uniqueNodesDict[node] = 1
			print(8)
			(DPST, uniqueNodes) = findAllDisjointPaths(tempG,s,t)
			print(9)
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


print(listOfAllSP)

"""

for source in range(lengthOfNodes-1):

	for target in range(source + 1, lengthOfNodes):

		newG = newGraph.copy()

		(DPST, uniqueNodes) = findAllDisjointPaths(newG,source,target)

		allDP.append(DPST)

		print(DPST)
		print(uniqueNodes)

		for node in uniqueNodes:
			if node in nodesDict:
				nodesDict[node] = nodesDict[node] + 1

			else:
				nodesDict[node] = 1


		
"""

