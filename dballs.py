import networkx as nx
import networkit as nk
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


def take_out_list(dBall, ball):
	new_list = []

	for i in dBall:

		if i in ball:
			continue

		new_list.append(i)

	return new_list








#change this such that the neighbors are diff
def get_dBN(G,node,radius):

	dBall = set([node])
	ball = set([node])

	for i in range(radius):

		neighbor = []

		for j in dBall:

			for n in G.neighbors(j):

				if n in ball:
					continue

				neighbor.append(n)

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


def dict_to_sorted_list(d):

	new_list = list(d.items())

	final_list = sorted(new_list, key = itemgetter(1))

	final_list_no_0 = list(filter(lambda x : x[1] != 0, final_list))

	return final_list_no_0


def get_GC_nodes(G):
	comp = nk.components.DynConnectedComponents(G)
	comp.run()

	all_comp = comp.getComponents()

	all_comp.sort(key = len)

	return all_comp[-1]





def get_GC(G):
	comp = nk.components.DynConnectedComponents(G)
	comp.run()

	all_comp_sizes = comp.getComponentSizes()

	all_values = list(all_comp_sizes.values())
	all_values.sort()

	return all_values[-1]



def copy_graph(G):
	G_copy = G.copyNodes()

	edges = G.edges()

	for (i,j) in edges:
		G_copy.addEdge(i,j)

	return G_copy



def perc_process_dBalls(G_copy,radius,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []
	num_nodes_removed = [] 
	nodes_removed = []

	counter = 0

	while counter < num_nodes_to_remove:

		(dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i) = get_all_dBN(G,radius)

		list_to_remove = dict_to_sorted_list(dict_nodes_x_i)

		if len(list_to_remove) == 0:
			break

		node = list_to_remove[0][0]

		print(counter)

		(dBall,ball) = get_dBN(G,node,radius) 


		print(dBall)
		print(ball)

		for i in dBall:
			G.removeNode(i)
			nodes_removed.append(i)
			counter += 1

		GC_List.append(get_GC(G))

		num_nodes_removed.append(counter)

	print(G.numberOfNodes())

	return (GC_List,num_nodes_removed,nodes_removed)



def perc_random(G_copy,num_nodes_to_remove):

	G = copy_graph(G_copy)

	all_nodes = random.sample(list(G.nodes()),num_nodes_to_remove)

	for i in all_nodes:
		G.removeNode(i)

	return get_GC(G)


def ADA_attack(G_copy,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []

	for i in range(num_nodes_to_remove):

		degree = nk.centrality.DegreeCentrality(G)

		degree.run()

		degree_sequence = degree.ranking()

		random.shuffle(degree_sequence)

		degree_sequence.sort(key = itemgetter(1), reverse = True)

		node_to_remove = degree_sequence[0][0]

		G.removeNode(node_to_remove)

		GC_List.append(get_GC(G))

	return GC_List


def ABA_attack(G_copy,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []

	for i in range(num_nodes_to_remove):

		between = nk.centrality.DynBetweenness(G)
		between.run()

		between_sequence = between.ranking()

		between_sequence.sort(key = itemgetter(1), reverse = True)

		node_to_remove = between_sequence[0][0]

		G.removeNode(node_to_remove)

		GC_List.append(get_GC(G))

	return GC_List




def turn_lists_together(GC_List,num_nodes_removed):

	final_list = []
	pointer = 0
	counter = 0

	for i in num_nodes_removed:

		diff = i - counter

		for j in range(diff):

			final_list.append(GC_List[pointer]) 
			counter += 1


		pointer += 1

	return final_list





N = 1000
k = 4
SEED = 321

radius = 2

G_nx = nx.erdos_renyi_graph(N, k/(N-1), seed = SEED) 
G_nx_copy = G_nx.copy()

G = nk.nxadapter.nx2nk(G_nx)
G_copy = nk.nxadapter.nx2nk(G_nx_copy)


get_GC_nodes(G)


"""
(dBalls_GC,nodes_remaining,nodes_removed) = perc_process_dBalls(G,radius,int(0.1*N))

ADA_GC = ADA_attack(G,int(0.1*N))

ABA_GC = ABA_attack(G,int(0.1*N))



print(dBalls_GC)
print(ADA_GC)
print(nodes_remaining)
print(ABA_GC)

print(len(dBalls_GC))
print(len(ADA_GC))


filename_ADA = "ADA.png"
filename_dball = "dBall.png"

final_list = turn_lists_together(dBalls_GC,nodes_remaining)

print(dBalls_GC)
print(nodes_remaining)
print(nodes_removed)
print(len(nodes_removed))


print(get_dBN(G,145,radius))
print(get_dBN(G,324,radius))
print(get_dBN(G,551,radius))

print(G.neighbors(551))
print(G.neighbors(145))
print(G.neighbors(324))
"""

#plt.plot(x_axis,dBalls_GC)
#plt.plot(x_axis,ADA_GC)
#plt.show()


"""

N=int(sys.argv[1]) # number of nodes
k=int(sys.argv[2]) # average degree
SEED = int(sys.argv[3])
radius = int(sys.argv[4])
num_sims = int(sys.argv[5])
step_size = float(sys.argv[6])

norm_vals = make_partitions_multiple_graphs(N,k,SEED,radius,step_size,num_sims)

print(norm_vals)

filename = 'dballs_N_' + str(N) + '_k_' + str(k) + '_SEED_' + str(SEED) + '_radius_' + str(radius) + "_numsims_" + str(num_sims) + '_stepsize_' + str(step_size) + '.pickle'

with open(filename,'wb') as handle:
	pickle.dump(norm_vals, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""
