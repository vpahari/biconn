import csv
import glob
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle
import math
from functools import reduce
import networkx as nx
import igraph as igG
import sys
import random


def make_WS_graph(dim,size,nei,p,SEED):

	N = size ** dim

	random.seed(SEED)

	igG = ig.Graph.Watts_Strogatz(dim,size,nei,p)

	allEdges = igG.get_edgelist()

	fixed_G = nx.Graph()

	listOfNodes = [i for i in range(N)]

	fixed_G.add_nodes_from(listOfNodes)

	fixed_G.add_edges_from(allEdges)

	return G_nx



def make_ER_Graph(N,k,SEED):

	G_nx = nx.erdos_renyi_graph(N, k/(N-1), seed = SEED) 

	return G_nx


def make_BA_Graph(N,k,SEED):

	k = int(k)

	G_nx = nx.barabasi_albert_graph(N, k, SEED)

	return G_nx

def get_avg_list(big_list):

	counter = 0

	size_of_list = len(big_list[0])

	avg_list = []

	while counter < size_of_list:

		index_list = list(map(lambda x : x[counter], big_list))

		avg = sum(index_list) / len(index_list)

		avg_list.append(avg)

		counter += 1

	return avg_list


def make_list(counter_list,GC_list):

	final_list = []

	i = 0

	while i < (len(counter_list) - 1):

		delta = counter_list[i+1] - counter_list[i]

		for j in range(delta):

			final_list.append(GC_list[i])

		i += 1


	#print(len(final_list))

	return final_list


def get_max_len(l):

	return len(max(l, key = len))

def make_lists_bigger(l,size):

	delta = size - len(l)

	for i in range(delta):

		l.append(0)




def make_list_same_size(l):

	max_size = get_max_len(l)

	for i in l:

		make_lists_bigger(i, max_size)


def get_n_k(filename):
	split_list = filename.split("_")
	N_position = split_list.index("N") + 1
	k_position = split_list.index("k") + 1
	SEED_position = split_list.index("SEED") + 1
	return (int(split_list[N_position]), float(split_list[k_position]),int(split_list[SEED_position]))


def get_deg_seq(N, k, p, SEED_list):

	num_graphs = 0

	degree_dict = {}

	for SEED in SEED_list:

		curr_ER = make_WS_graph(1,N,k,p,SEED)

		degree_seq = list(map(lambda x : x[1],list(curr_ER.degree())))

		for i in degree_seq:

			if i not in degree_dict:
				degree_dict[i] = 0

			degree_dict[i] = degree_dict[i] + 1


		num_graphs += 1


	num_to_divide = N * num_graphs

	for i in list(degree_dict.keys()):

		degree_dict[i] = degree_dict[i] / num_to_divide

	print(degree_dict)

	x_list = sorted(list(degree_dict.keys()))

	y_list = []

	for i in x_list:
		y_list.append(degree_dict[i])

	print(x_list)
	print(y_list)

	return (x_list, y_list)

	


N = 5000

SEED = 4123

p = int(sys.argv[1])

SEED_list = []

for i in range(10):

	SEED = (SEED * (i+1)) + 1

	SEED_list.append(SEED)




(x_list_2, k_2_deg) = get_deg_seq(N, 2, p, SEED_list)
(x_list_3, k_3_deg) = get_deg_seq(N, 3, p, SEED_list)
(x_list_4, k_4_deg) = get_deg_seq(N, 4, p, SEED_list)
(x_list_5, k_5_deg) = get_deg_seq(N, 5, p, SEED_list)
(x_list_6, k_6_deg) = get_deg_seq(N, 6, p, SEED_list)






plt.xlabel('degree', fontsize=10)
plt.ylabel('p_k', fontsize=10, rotation=0, labelpad=20)

plt.plot(x_list_2, k_2_deg,".-", label = "k=2")
plt.plot(x_list_3, k_3_deg,".-", label = "k=3")
plt.plot(x_list_4, k_4_deg,".-", label = "k=4")
plt.plot(x_list_5, k_5_deg,".-", label = "k=5")
plt.plot(x_list_6, k_6_deg,".-", label = "k=6")


filename = "WS_P_K_seq" + "_N_" + str(N) + "_p_" + str(p) + ".png"

#plt.xlim(0,100)

plt.legend(loc='best')
plt.tight_layout() 
plt.savefig(filename)
plt.clf()





