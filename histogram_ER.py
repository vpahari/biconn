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




def get_name_ER(initial_name, N, k, SEED,radius):

	return initial_name + "_N_" + str(N) + "_k_" + str(k) + "_SEED_" + str(SEED) + "_radius_" + str(radius) + "_" + ".pickle"



def make_dict(l):

	d = {}

	for i in l:

		if i not in d:
			d[i] = 0

		d[i] += 1

	return d



N=int(sys.argv[1]) 

k=float(sys.argv[2])

SEED=int(sys.argv[3])

p = k / (N-1)


G = make_ER_Graph(N,p,SEED)

degree_seq = []

for i in range(N):

	#print(G.degree(i))

	degree_seq.append(G.degree(i))

#print(degree_seq)

d = make_dict(degree_seq)

keys = list(d.keys())

keys.sort()

#print(keys)

for k in keys:

	print(k, d[k])



