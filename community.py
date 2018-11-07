import networkx as nx
import networkit as nk
import random

def connect(G,nTC1,nTC2,M_out):
	newSet = set([])
	counter = 0
	while counter < M_out:
		i = random.choice(nTC1)
		j = random.choice(nTC2)
		if (i,j) not in newSet:
			G.addEdge(i,j)
			newSet.add((i,j))
			counter += 1
		



N = 1000

k = 4.0

p = k / float(N-1)

S = 2351

Graph1 = nx.erdos_renyi_graph(N, p, seed = S)

Graph2 = nx.erdos_renyi_graph(N, p, seed = S)

G1 = nk.nxadapter.nx2nk(Graph1)

G2 = nk.nxadapter.nx2nk(Graph2)

#print(list(G1.nodes()))

#print(list(G2.nodes()))

for i in range(N,2*N):
	G1.addNode()

#print(list(G1.nodes()))

for i,j in list(G2.edges()):
	G1.addEdge(N + i, N + j)

#print(list(G1.edges()))

#print(list(G2.edges()))

r = 10 ** (-2)

numNodesToConnect = int(r * N)

nodesInGraph = list(G1.nodes())

N1 = nodesInGraph[:N]

N2 = nodesInGraph[N:]

print(N1)

print(N2)

print(numNodesToConnect)
#nodes to connect
nTC1 = random.sample(N1, numNodesToConnect)
nTC2 = random.sample(N2, numNodesToConnect)

print(nTC1)
print(nTC2)

M_out = int(N / 10)


connect(G1,nTC1,nTC2,M_out)





