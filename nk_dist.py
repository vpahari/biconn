import networkx as nx
import networkit as nk

#import matplotlib.pyplot as plt

def findAllDisjointPaths(G,s,t):
	distPaths = []

	uniqueNodes = [] 

	newDist = nk.distance.Dijkstra(G, s, True, True)

	newDist.run()

	hasPath = newDist.numberOfPaths(t)

	print(hasPath)

	while hasPath != 0:
		
		shortestPath = newDist.getPath(t)

		print(shortestPath)

		if len(shortestPath) == 2:
			G.removeEdge(s,t)
			G.removeEdge(t,s)

		else:
			for i in shortestPath[1:len(shortestPath) - 1]:
				G.removeNode(i)
				uniqueNodes.append(i)

		
		distPaths.append(shortestPath)
		
		newDist = nk.distance.Dijkstra(G, s, True, True)
		
		newDist.run()
		
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
	
	dijk.run()


	for t in range(s+1,N):
		print("currNode")
		print(t)
		
		isPath = dijk.numberOfPaths(t)
		
		tempG = newGraph.copyNodes()
		
		for (e1,e2) in newGraph.edges():
			tempG.addEdge(e1,e2)

		if isPath != 0:
			
			shortestPath = dijk.getPath(t)
			allSPforS.append(shortestPath)
			
			nodesToRemove = shortestPath[1:len(shortestPath)-1]
			
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

		print(allSPforS)

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

