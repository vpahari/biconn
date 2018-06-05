#import random
#import sys
#import math
#from functools import reduce
#import csv
#from operator import itemgetter

import networkx as nx
import networkit as nk

#import matplotlib.pyplot as plt

N = 100

k = 4.0

p = k / float(N)

S = 1099 

G = nx.erdos_renyi_graph(N, p, seed = S)

print(len(G))

Gnk = nk.nxadapter.nx2nk(G)

print(Gnk)

print("Nodes")

listOfNodes = Gnk.nodes()

print(nk.distance.getDistance)

print(type(listOfNodes[0]))

#for i in range(10):
print(Gnk.nk.distance.getDistances())
	

