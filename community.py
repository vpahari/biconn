import networkx as nx
import networkit as nk

N = 1000

k = 4.0

p = k / float(N-1)

S = 2351

Graph1 = nx.erdos_renyi_graph(N, p, seed = S)

Graph2 = nx.erdos_renyi_graph(N, p, seed = S)

G1 = nk.nxadapter.nx2nk(Graph1)

G2 = nk.nxadapter.nx2nk(Graph2)

print(list(G1.nodes()))

print(list(G2.nodes()))

for i in range(1000,2000):
	G1.addNode()

print(list(G1.nodes()))

for i,j in list(G2.edges()):
	G1.addEdge(1000 + i, 1000 + j)

print(list(G1.edges()))

print(list(G2.edges()))

#(1993, 1957), (1994, 1914), (1995, 1035), (1995, 1043), (1995, 1262), (1995, 1265), (1995, 1359), (1995, 1407), (1995, 1409), (1995, 1535), (1995, 1661), (1996, 1664), (1996, 1955), (1996, 1981), (1997, 1433), (1997, 1595), (1997, 1931), (1998, 1307), (1998, 1854), (1998, 1920), (1999, 1097), (1999, 1337), (1999, 1445), (1999, 1751)]



