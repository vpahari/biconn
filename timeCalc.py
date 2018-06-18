import time
import networkx as nx
import pickle

N = 5000

k = 4.0

SEED = 9875

G = nx.erdos_renyi_graph(N, k/(N-1), SEED)

startTime = time.time()

Gdict = nx.betweenness_centrality(G, N, normalized = True)

endTime = time.time()

print(Gdict)
print(startTime)
print(endTime)

output_file_name = "time_N_%d_k_%d.txt"%(N,k)

fh = open(output_file_name, 'w')

fh.write(str(startTime) + "\n")
fh.write(str(endTime) + "\n")

fh.close()