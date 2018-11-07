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

