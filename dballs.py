import networkx as nx
#import networkit as nk
import random
import sys
import math
from functools import reduce
import csv
from operator import itemgetter
import matplotlib.pyplot as plt
import pickle


def add_into_set(s,new_s):
	for i in new_s:
		s.add(i)

	return s

def get_dBN(G,node,radius):

	dBall = set([node])
	ball = set([node])

	for i in range(radius):

		neighbor = []

		for j in dBall:

			neighbor += G.neighbors(j)

		ball = add_into_set(ball,neighbor)

		dBall = set(neighbor.copy())

	return (dBall,ball)



def get_all_dBN(G,radius):

	all_nodes = G.nodes()

	dict_nodes_dBall = {}
	dict_nodes_ball = {}
	dict_nodes_x_i = {}

	for n in all_nodes:

		(dBall,ball) = get_dBN(G,n,radius)

		dict_nodes_dBall[n] = len(dBall)
		dict_nodes_ball[n] = len(ball)
		dict_nodes_x_i[n] = len(dBall) / len(ball)

	return (dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i)


		 
def make_partitions(dict_nodes_x_i, step_size):

	counter = 0

	values_list = list(dict_nodes_x_i.values())

	num_partitions = int(1 / step_size)

	all_values = [0 for i in range(num_partitions)]

	for i in values_list:

		box_to_put = int(i / step_size)

		if box_to_put == num_partitions:
			all_values[-1] = all_values[-1] + 1
			continue

		all_values[box_to_put] = all_values[box_to_put] + 1

	return all_values


def merge_boxes(boxes,big_list):

	for i in range(len(boxes)):

		big_list[i] = big_list[i] + boxes[i]




def make_partitions_multiple_graphs(N,k,SEED,radius,step_size,num_sims):

	num_partitions = int(1 / step_size)

	all_values = [0 for i in range(num_partitions)]

	for i in range(num_sims):

		G = nx.erdos_renyi_graph(N, k/(N-1), seed = SEED * (i+1) + 1) 

		(dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i) = get_all_dBN(G,radius)

		boxes = make_partitions(dict_nodes_x_i,step_size)

		merge_boxes(boxes,all_values)

	normalized_values = list(map(lambda x : x / (N * num_sims), all_values))

	print(sum(all_values))
	print(sum(normalized_values))

	return normalized_values





N=int(sys.argv[1]) # number of nodes
k=int(sys.argv[2]) # average degree
SEED = int(sys.argv[3])
radius = int(sys.argv[4])
num_sims = int(sys.argv[5])
step_size = float(sys.argv[6])

norm_vals = make_partitions_multiple_graphs(N,k,SEED,radius,step_size,num_sims)

print(norm_vals)

filename = 'dballs_N_' + str(N) + '_k_' + str(k) + '_SEED_' + str(SEED) + '_radius_xw' + str(radius) + "_numsims_" + str(num_sims) + '_stepsize_' + str(step_size) '.pickle'

with open(filename,'wb') as handle:
	pickle.dump(norm_vals, handle, protocol=pickle.HIGHEST_PROTOCOL)

