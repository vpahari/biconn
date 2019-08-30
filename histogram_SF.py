import networkx as nx
import networkit as nk
import random
import sys
import math
from functools import reduce
import csv
from operator import itemgetter
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle
import igraph as ig
import numpy as np
import os
import itertools




def make_SF_Graph(N,k,exp_out,SEED):

	random.seed(SEED)

	num_edges = int((N * k) / 2)

	igG = ig.Graph.Static_Power_Law(N,num_edges,exp_out)

	allEdges = igG.get_edgelist()

	fixed_G = nx.Graph()

	listOfNodes = [i for i in range(N)]

	fixed_G.add_nodes_from(listOfNodes)

	fixed_G.add_edges_from(allEdges)

	G_nk = nk.nxadapter.nx2nk(fixed_G)

	return G_nk



def make_dict(l):

	d = {}

	for i in l:

		if i not in d:
			d[i] = 0

		d[i] += 1

	return d



N=int(sys.argv[1]) 

k=float(sys.argv[2])

exp_out = float(sys.argv[3])

SEED=int(sys.argv[4])


G = make_SF_Graph(N,k,exp_out,SEED)

degree_seq = []

for i in range(N):

	print(G.degree(i))

	degree_seq.append(G.degree(i))

print(degree_seq)

d = make_dict(degree_seq)

print(d)






