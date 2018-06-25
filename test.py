import networkx as nx
import networkit as nk

biconn = list(nx.biconnected_component_subgraphs(GER))

print(len(max(biconn, key=len)))

GER = nx.erdos_renyi_graph(1000,2/999,232)

Gnk = nk.nxadapter.nx2nk(GER)

newGraph = nk.graph.Graph(n = 1000, weighted = True, directed = True)

lastEdges = Gnk.edges()

for (i,j) in lastEdges:
	newGraph.addEdge(i,j)
	newGraph.addEdge(j,i)


dijk = nk.distance.Dijkstra(newGraph, 0, True, False)

dijk.run()

dijk.numberOfPaths(1)

wt = dijk.getDistance(0,1)

print(wt)