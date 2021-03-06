import networkx as nx
import networkit as nk

GER = nx.erdos_renyi_graph(1000,2/999,232)

biconn = list(nx.connected_component_subgraphs(GER))

print(len(max(biconn, key=len)))

Gnk = nk.nxadapter.nx2nk(GER)

newGraph = nk.graph.Graph(n = 1000, weighted = True, directed = True)

lastEdges = Gnk.edges()

for (i,j) in lastEdges:
	newGraph.addEdge(i,j)
	newGraph.addEdge(j,i)


dijk = nk.distance.Dijkstra(newGraph, 0, True, True)

dijk.run()

dijk.numberOfPaths(1)

wt = dijk.distance(630)

stack = dijk.getStack()

print(wt)

print(len(stack))