import networkx as nx
import networkit as nk

#import matplotlib.pyplot as plt

def findAllDisjointPaths(G,s,t):
	distPaths = []

	uniqueNodes = [] 

	while True:

		dijk = nk.distance.Dijkstra(G, s, True, False, t)

		try:
			dijk.run()
		except:
			return (distPaths, uniqueNodes)

		shortestPath = dijk.getPath(t)

		for i in range(1,len(shortestPath) - 1):
			G.removeNode(shortestPath[i])
			uniqueNodes.append(shortestPath[i])

		distPaths.append(shortestPath)


N = 10000

k = 4.0

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

allDP = []

nodesDict = {}

lengthOfNodes = len(listOfNodes)

s = 0

t = 5

print(newGraph.edges())

dijk = nk.distance.Dijkstra(newGraph, 0, True, True)
dijk.run()

for i in range(1,N):
	v = dijk.numberOfPaths(i)
	print(v)


"""
for source in range(N-1):
	print(source)
	dijk = nk.distance.Dijkstra(newGraph, source, True, True)
	print(source)
	dijk.run()
	for i in range(source+1,N):
		#v = dijk.getNodesSortedByDistance()
		v = dijk.numberOfPaths(i)
		print(v)
"""

		"""
		try:
			print("this")
			shortestPath = dijk.getPath(i)
			print(shortestPath)
		except:
			print([])
		"""

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

