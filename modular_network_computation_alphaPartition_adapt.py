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








def get_name_WS(initial_name, dim, size, nei, p, SEED,radius):

	return initial_name + "_dim_" + str(dim) + "_size_" + str(size) + "_nei_" + str(nei) + "_p_" + str(p) + "_SEED_" + str(SEED) + "_radius_" + str(radius) + "_" + ".pickle"


def get_name_ER(initial_name, N, k, SEED,radius):

	return initial_name + "_N_" + str(N) + "_k_" + str(k) + "_SEED_" + str(SEED) + "_radius_" + str(radius) + "_" + ".pickle"

def get_name_SF(initial_name,N,k,exp_out,SEED,radius):

	return initial_name + "_N_" + str(N) + "_k_" + str(k) + "_expout_" + str(exp_out) + "_SEED_" + str(SEED) + "_radius_" + str(radius) + "_" + ".pickle"


def get_name_ModularNetwork(initial_name,N,k_intra,k_inter,SEED1,SEED2,radius):

	return initial_name + "_N_" + str(N) + "_kintra_" + str(k_intra) + "_kinter_" + str(k_inter) + "_SEED1_" + str(SEED1) + "_SEED2_" + str(SEED2) + "_radius_" + str(radius) + "_" + ".pickle"


def get_name_ModularNetworks(initial_name,N,k_intra,k_inter,SEED,num_modules,radius):

	return initial_name + "_N_" + str(N) + "_kintra_" + str(k_intra) + "_kinter_" + str(k_inter) + "_SEED_" + str(SEED) + "_numModules_" + str(num_modules) +  "_radius_" + str(radius) + "_" + ".pickle"



def get_name_ModularNetworks_alpha(initial_name,N,k_intra,k_inter,alpha,SEED,num_modules,radius):

	return initial_name + "_N_" + str(N) + "_kintra_" + str(k_intra) + "_kinter_" + str(k_inter) + "_alpha_" + str(alpha) + "_SEED_" + str(SEED) + "_numModules_" + str(num_modules) +  "_radius_" + str(radius) + "_" + ".pickle"



def make_graphs_into_one_multiple_graphs_alpha(G_list,num_edges_to_connect,alpha):

	counter = 0

	G = nk.Graph()

	#print(G.nodes())
	#print(G.edges())

	nodes_list = []

	alpha_nodes_list = []

	set_of_connected_nodes = set([])

	size_G_nodes = 0

	for G_mod in G_list:

		G_nodes = list(G_mod.nodes())
		G_edges = list(G_mod.edges())

		G_nodes = list(map(lambda x : x + size_G_nodes, G_nodes))
		#size_G_nodes += len(G_nodes)

		nodes_list.append(G_nodes)

		num_nodes_to_sample = int(len(G_nodes) * alpha)

		print(num_nodes_to_sample)

		curr_alpha = random.sample(G_nodes, num_nodes_to_sample)

		alpha_nodes_list.append(curr_alpha)


		for n in G_nodes:
			G.addNode()


		for (a,b) in G_edges:
			u = size_G_nodes + a
			v = size_G_nodes + b
			G.addEdge(u,v)


		size_G_nodes += len(G_nodes)


	for i in range(num_edges_to_connect):
		
		(u,v) = get_random_u_v(alpha_nodes_list)
		G.addEdge(u,v)

		set_of_connected_nodes.add(u)
		set_of_connected_nodes.add(v)

	print(alpha_nodes_list)
	print(set_of_connected_nodes)

	for i in alpha_nodes_list:
		print(len(i))

	print(len(set_of_connected_nodes))

	return (G, set_of_connected_nodes)


def make_graphs_into_one(G1,G2,num_edges_to_connect,alpha):

	counter = 0

	G = nk.Graph()

	print(G.nodes())
	print(G.edges())

	G1_edges = list(G1.edges())
	G2_edges = list(G2.edges())

	G1_nodes = list(G1.nodes())
	G2_nodes = list(G2.nodes())

	size_of_G1 = len(G1_nodes)

	G2_nodes = list(map(lambda x : size_of_G1 + x, G2_nodes))

	num_nodes_to_connect = alpha * len(G1_nodes)

	nodes_to_connect_G1 = random.choice(G1_nodes, num_nodes_to_connect)
	nodes_to_connect_G2 = random.choice(G2_nodes, num_nodes_to_connect)

	for i in G1_nodes:
		G.addNode()

	for i in G2_nodes:
		G.addNode()

	#print(G.nodes())
	print(len(G.nodes()))

	for i,j in G1_edges:
		G.addEdge(i,j)

	print(G.edges())
	#print(len(G.edges()))

	for i,j in G2_edges:
		u = size_of_G1 + i
		v = size_of_G1 + j
		G.addEdge(u,v)

	print(G.edges())
	print(len(G1.edges()))
	print(len(G2.edges()))
	print(len(G.edges()))

	for i in range(num_edges_to_connect):

		u = random.choice(nodes_to_connect_G1)
		v = random.choice(nodes_to_connect_G2)

		G.addEdge(u,v)

	print(num_edges_to_connect)
	print(len(G.edges()))

	print(G.edges())

	return G



def get_random_u_v(nodes_list):

	l = [i for i in range(len(nodes_list))]

	u_index = random.choice(l)

	u = random.choice(nodes_list[u_index])

	l.remove(u_index)

	v_index = random.choice(l)

	v = random.choice(nodes_list[v_index])

	return (u,v)




def make_graphs_into_one_multiple_graphs(G_list,num_edges_to_connect):

	counter = 0

	G = nk.Graph()

	#print(G.nodes())
	#print(G.edges())

	nodes_list = []

	size_G_nodes = 0

	for G_mod in G_list:

		G_nodes = list(G_mod.nodes())
		G_edges = list(G_mod.edges())

		G_nodes = list(map(lambda x : x + size_G_nodes, G_nodes))

		nodes_list.append(G_nodes)

		for n in G_nodes:
			G.addNode()

		print(G_edges)
		print(list(G.nodes()))

		for (a,b) in G_edges:
			u = size_G_nodes + a
			v = size_G_nodes + b
			G.addEdge(u,v)

		size_G_nodes += len(G_nodes)

	set_of_connected_nodes = set([])

	for i in range(num_edges_to_connect):
		
		(u,v) = get_random_u_v(nodes_list)
		G.addEdge(u,v)

		set_of_connected_nodes.add(u)
		set_of_connected_nodes.add(v)


	return (G, set_of_connected_nodes)


def make_modular_network_ER(N,k_intra,k_inter,num_modules,SEED,alpha):

	size_of_one_module = int(N / num_modules)

	list_Graphs = [make_ER_Graph(size_of_one_module, k_intra, SEED * (i+2) + 1) for i in range(num_modules)]

	#G = nk.Graph()

	beta_i = size_of_one_module / N

	num_edges = int(k_inter * N / 2)

	for i in list_Graphs:

		print(i.numberOfNodes())
		print(i.numberOfEdges())
	

	(G, set_of_connected_nodes) = make_graphs_into_one_multiple_graphs_alpha(list_Graphs,num_edges,alpha)

	print(set_of_connected_nodes)

	return (G, set_of_connected_nodes)


	"""
	nodes_list = []

	size_G_nodes = 0

	for count in range(num_modules):

		SEED = (SEED * (count+1)) + 1

		G_mod = make_ER_Graph(size_of_one_module, k_intra, SEED)

		G_nodes = list(G_mod.nodes())
		G_edges = list(G_mod.edges())

		G_nodes = list(map(lambda x : x + size_G_nodes, G_nodes))

		nodes_list.append(G_nodes)

		for n in G_nodes:
			G.addNode()

		print(G_edges)
		print(list(G.nodes()))

		for (a,b) in G_edges:

			u = size_G_nodes + a
			v = size_G_nodes + b

			G.addEdge(u,v)


		size_G_nodes += len(G_nodes)

	return G
	"""

















	


def get_num_k_inter_edges(N, num_modules, prob):

	size_of_one_module = int(N / num_modules)







def make_WS_graph(dim,size,nei,p,SEED):

	N = size ** dim

	random.seed(SEED)

	igG = ig.Graph.Watts_Strogatz(dim,size,nei,p)

	allEdges = igG.get_edgelist()

	fixed_G = nx.Graph()

	listOfNodes = [i for i in range(N)]

	fixed_G.add_nodes_from(listOfNodes)

	fixed_G.add_edges_from(allEdges)

	G_nk = nk.nxadapter.nx2nk(fixed_G)

	return G_nk


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



def make_ER_Graph(N,k,SEED):

	G_nx = nx.erdos_renyi_graph(N, k/(N-1), seed = SEED) 

	G_nk = nk.nxadapter.nx2nk(G_nx)

	return G_nk



def DA_attack(G_copy,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []

	original_degree_list = []

	adaptive_degree_list = []

	GC_List.append(get_GC(G))

	degree = nk.centrality.DegreeCentrality(G)

	degree.run()

	degree_sequence = degree.ranking()

	random.shuffle(degree_sequence)

	degree_sequence.sort(key = itemgetter(1), reverse = True)

	for i in range(num_nodes_to_remove):

		node_to_remove = degree_sequence[i][0]

		original_degree = degree_sequence[i][1]

		adaptive_degree_list.append(G.degree(node_to_remove))

		original_degree_list.append(original_degree)

		G.removeNode(node_to_remove)

		GC_List.append(get_GC(G))

	return (GC_List, original_degree_list,adaptive_degree_list)


def ADA_attack(G_copy,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []

	SGC_List = []

	num_comp_List = []

	(GC,SGC,num_comp) = get_GC_SGC_number_of_components(G)

	GC_List.append(GC)

	SGC_List.append(SGC)

	num_comp_List.append(num_comp)

	degree_list = []

	for i in range(num_nodes_to_remove):

		degree = nk.centrality.DegreeCentrality(G)

		degree.run()

		degree_sequence = degree.ranking()

		random.shuffle(degree_sequence)

		degree_sequence.sort(key = itemgetter(1), reverse = True)

		node_to_remove = degree_sequence[0][0]

		degree_list.append(G.degree(node_to_remove))

		G.removeNode(node_to_remove)

		(GC,SGC,num_comp) = get_GC_SGC_number_of_components(G)

		GC_List.append(GC)

		SGC_List.append(SGC)

		num_comp_List.append(num_comp)

	return (GC_List, SGC_List, num_comp_List, degree_list)


def BA_attack(G_copy,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []

	GC_List.append(get_GC(G))

	between = nk.centrality.DynBetweenness(G)
	between.run()

	between_sequence = between.ranking()

	random.shuffle(between_sequence)

	between_sequence.sort(key = itemgetter(1), reverse = True)

	for i in range(num_nodes_to_remove):
		
		node_to_remove = between_sequence[i][0]

		G.removeNode(node_to_remove)

		GC_List.append(get_GC(G))

	return GC_List


def ABA_attack(G_copy,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []

	SGC_List = []

	num_comp_List = []

	(GC,SGC,num_comp) = get_GC_SGC_number_of_components(G)

	GC_List.append(GC)

	SGC_List.append(SGC)

	num_comp_List.append(num_comp)

	for i in range(num_nodes_to_remove):

		between = nk.centrality.DynBetweenness(G)
		between.run()

		between_sequence = between.ranking()

		between_sequence.sort(key = itemgetter(1), reverse = True)

		node_to_remove = between_sequence[0][0]

		G.removeNode(node_to_remove)

		(GC,SGC,num_comp) = get_GC_SGC_number_of_components(G)

		GC_List.append(GC)

		SGC_List.append(SGC)

		num_comp_List.append(num_comp)

	return (GC_List, SGC_List, num_comp_List)



def RA_attack(G_copy,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []

	SGC_List = []

	num_comp_List = []

	(GC,SGC,num_comp) = get_GC_SGC_number_of_components(G)

	GC_List.append(GC)

	SGC_List.append(SGC)

	num_comp_List.append(num_comp)

	all_nodes = random.sample(list(G.nodes()),num_nodes_to_remove)

	for i in all_nodes:

		G.removeNode(i)

		(GC,SGC,num_comp) = get_GC_SGC_number_of_components(G)

		GC_List.append(GC)

		SGC_List.append(SGC)

		num_comp_List.append(num_comp)

	return (GC_List, SGC_List, num_comp_List)



def big_RA_attack(G_copy,num_nodes_to_remove,num_sims):

	big_GC_List = []
	big_SGC_List = []
	big_numComp_List = []

	for i in range(num_sims):

		(GC_List, SGC_List, num_comp_List) = RA_attack(G_copy,num_nodes_to_remove)

		big_GC_List.append(GC_List)
		big_SGC_List.append(SGC_List)
		big_numComp_List.append(num_comp_List)

	avg_list_GC = get_avg_list(big_GC_List)
	avg_list_SGC = get_avg_list(big_SGC_List)
	avg_list_numComp = get_avg_list(big_numComp_List)

	return (avg_list_GC, avg_list_SGC, avg_list_numComp) 



def get_betweenness_score(G, node):

	between = nk.centrality.DynBetweenness(G)
	between.run()

	return between.score(node)


def get_degree_score(G,node):

	return G.degree(node)


def get_coreness_score(G,node):

	coreness = nk.centrality.CoreDecomposition(G) 
	coreness.run()

	partition = coreness.getPartition()
	core_number = partition.subsetOf(node)

	return core_number


def get_betweenness_score_list(G, node_list):

	between = nk.centrality.DynBetweenness(G)
	between.run()

	final_list = []

	for node in node_list:

		final_list.append(between.score(node))

	return final_list


def get_degree_score_list(G,node_list):

	final_list = []

	for node in node_list:

		final_list.append(G.degree(node))

	return final_list


def get_coreness_score_list(G,node_list):

	coreness = nk.centrality.CoreDecomposition(G) 
	coreness.run()

	final_list = []

	partition = coreness.getPartition()

	for node in node_list:

		final_list.append(partition.subsetOf(node))

	return final_list



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

	return (list(dBall),list(ball))



def get_all_dBN(G,radius):

	all_nodes = get_GC_nodes(G)

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



def get_all_same_x_i(sorted_list,x_i_value):

	node_list = []

	for i in sorted_list:

		if i[1] == x_i_value:

			node_list.append(i[0])

	return node_list





def get_largest_dball(dball_dict,node_list):

	largest_dball = 0
	largest_node = 0

	for i in node_list:

		print(dball_dict[i])

		if dball_dict[i] > largest_dball:

			largest_dball = dball_dict[i]
			largest_node = i

	return largest_node



def get_random_dball(node_list):

	return random.choice(node_list)



def dict_to_sorted_list(d):

	new_list = list(d.items())

	final_list = sorted(new_list, key = itemgetter(1))

	final_list_no_0 = list(filter(lambda x : x[1] != 0, final_list))

	if len(final_list_no_0) != 0:

		x_i_value = final_list_no_0[0][1]

		nodes_list = get_all_same_x_i(final_list_no_0, x_i_value)

		return nodes_list 

	else:

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


def get_avg_comp_size(all_val):

	avg = sum(all_val) / len(all_val)

	return avg



def get_GC_SGC_number_of_components(G):

	comp = nk.components.DynConnectedComponents(G)
	comp.run()

	all_comp_sizes = comp.getComponentSizes()

	all_values = list(all_comp_sizes.values())
	all_values.sort()

	if len(all_values) == 1:
		return (all_values[-1], 0,1, all_values[-1])

	else:
		avg_comp_size = get_avg_comp_size(all_values)
		return (all_values[-1],all_values[-2],len(all_values),avg_comp_size)



def copy_graph(G):
	G_copy = G.copyNodes()

	edges = G.edges()

	for (i,j) in edges:
		G_copy.addEdge(i,j)

	return G_copy




#dball, vball, degree, betweenness, coreness
def dBalls_attack(G_copy,radius):

	G = copy_graph(G_copy)

	GC_List = []
	SGC_List = []
	num_comp_List = []


	size_dball = [] 
	size_ball = []

	degree_list_mainNode = []
	betweenness_list_mainNode = []
	coreness_list_mainNode = []

	degree_list_removedNode = []
	betweenness_list_removedNode = []
	coreness_list_removedNode = []

	counter = 0

	counter_list = []

	(GC,SGC,num_comp) = get_GC_SGC_number_of_components(G)

	GC_List.append(GC)
	SGC_List.append(SGC)
	num_comp_List.append(num_comp)

	counter_list.append(counter)

	num_nodes_to_remove = G.numberOfNodes()

	while counter < num_nodes_to_remove:

		print(counter)

		(dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i) = get_all_dBN(G,radius)

		list_to_remove = dict_to_sorted_list(dict_nodes_x_i)

		if len(list_to_remove) == 0:
			break
		

		node = get_random_dball(list_to_remove)
		(dBall,ball) = get_dBN(G,node,radius) 

		combined_list = [node] + dBall

		between_list = get_betweenness_score_list(G,combined_list)
		degree_list = get_degree_score_list(G,combined_list)
		coreness_list = get_coreness_score_list(G,combined_list)



		degree_list_mainNode.append(degree_list[0])
		betweenness_list_mainNode.append(between_list[0])
		coreness_list_mainNode.append(coreness_list[0])

		degree_list_removedNode += degree_list[1:]
		betweenness_list_removedNode += between_list[1:]
		coreness_list_removedNode += coreness_list[1:]
		

		size_dball.append(len(dBall))
		size_ball.append(len(ball))


		#print(dBall)
		#print(ball)

		for i in dBall:
			G.removeNode(i)
			counter += 1

		(GC,SGC,num_comp) = get_GC_SGC_number_of_components(G)

		GC_List.append(GC)
		SGC_List.append(SGC)
		num_comp_List.append(num_comp)

		counter_list.append(counter)


	return (GC_List,SGC_List,num_comp_List,counter_list,size_dball,size_ball,degree_list_mainNode,betweenness_list_mainNode,coreness_list_mainNode,degree_list_removedNode,betweenness_list_removedNode,coreness_list_removedNode)


def get_degree_dict(G):

	all_nodes = list(G.nodes())

	final_dict = {}

	for i in all_nodes:

		final_dict[i] = G.degree(i)

	return final_dict



def dBalls_attack_NA(G_copy,radius):

	G = copy_graph(G_copy)

	GC_List = []
	SGC_List = []
	num_comp_List = []
	avg_comp_size_List = []

	size_dball = [] 
	size_ball = []

	degree_list_mainNode = []
	betweenness_list_mainNode = []
	coreness_list_mainNode = []

	degree_list_removedNode = []
	betweenness_list_removedNode = []
	coreness_list_removedNode = []

	counter = 0

	counter_list = []

	(GC,SGC,num_comp,avg_comp_size) = get_GC_SGC_number_of_components(G)

	GC_List.append(GC)
	SGC_List.append(SGC)
	num_comp_List.append(num_comp)
	avg_comp_size_List.append(avg_comp_size)

	counter_list.append(counter)

	original_degree_dict = get_degree_dict(G)

	original_degree_main_node = []

	original_degree_removed_node = []

	num_nodes_to_remove = G.numberOfNodes()

	(dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i) = get_all_dBN(G,radius)

	list_to_remove = dict_to_sorted_list_NA(dict_nodes_x_i)

	counter_for_nodes = 0

	original_xi_values = []



	print(dict_nodes_x_i)

	print(list_to_remove)

	while counter_for_nodes < len(list_to_remove):

		curr_nodes_set = set(list(G.nodes()))

		node = list_to_remove[counter_for_nodes][0]

		print(node,dict_nodes_dBall[node])


		if node not in curr_nodes_set:
			counter_for_nodes += 1
			continue


		(dBall,ball) = get_dBN(G,node,radius) 

		original_xi_values.append(list_to_remove[counter_for_nodes][1])


		if len(dBall) == 0:
			counter_for_nodes += 1
			continue


		size_dball.append(len(dBall))
		size_ball.append(len(ball))

		combined_list = [node] + dBall

		original_degree_main_node.append(original_degree_dict[node])

		for i in dBall:
			original_degree_removed_node.append(original_degree_dict[i])

		#between_list = get_betweenness_score_list(G,combined_list)
		degree_list = get_degree_score_list(G,combined_list)
		#coreness_list = get_coreness_score_list(G,combined_list)

		degree_list_mainNode.append(degree_list[0])
		#betweenness_list_mainNode.append(between_list[0])
		#coreness_list_mainNode.append(coreness_list[0])

		degree_list_removedNode += degree_list[1:]
		#betweenness_list_removedNode += between_list[1:]
		#coreness_list_removedNode += coreness_list[1:]

		for i in dBall:
			G.removeNode(i)
			counter += 1

		(GC,SGC,num_comp,avg_comp_size) = get_GC_SGC_number_of_components(G)

		GC_List.append(GC)
		SGC_List.append(SGC)
		num_comp_List.append(num_comp)
		avg_comp_size_List.append(avg_comp_size)

		counter_list.append(counter)

		counter_for_nodes += 1

	return (GC_List, SGC_List, num_comp_List, avg_comp_size_List, counter_list,size_dball,size_ball,degree_list_mainNode,degree_list_removedNode,original_degree_main_node,original_degree_removed_node, original_xi_values)



def change_connected_node_mainNode(node, nodes_list,set_of_connected_nodes):

	if node in set_of_connected_nodes:
		nodes_list.append(nodes_list[-1] + 1)

	else:
		nodes_list.append(nodes_list[-1])


def dBalls_attack_NA_MOD(G_copy,radius,set_of_connected_nodes):

	G = copy_graph(G_copy)

	GC_List = []
	SGC_List = []
	num_comp_List = []
	avg_comp_size_List = []

	size_dball = [] 
	size_ball = []

	degree_list_mainNode = []
	betweenness_list_mainNode = []
	coreness_list_mainNode = []

	degree_list_removedNode = []
	betweenness_list_removedNode = []
	coreness_list_removedNode = []

	connected_nodes_mainNode = []
	connected_nodes_removedNode = []

	counter = 0

	counter_list = []

	(GC,SGC,num_comp,avg_comp_size) = get_GC_SGC_number_of_components(G)

	GC_List.append(GC)
	SGC_List.append(SGC)
	num_comp_List.append(num_comp)
	avg_comp_size_List.append(avg_comp_size)
	connected_nodes_mainNode.append(0)
	connected_nodes_removedNode.append(0)

	counter_list.append(counter)

	original_degree_dict = get_degree_dict(G)

	original_degree_main_node = []

	original_degree_removed_node = []

	num_nodes_to_remove = G.numberOfNodes()

	(dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i) = get_all_dBN(G,radius)

	list_to_remove = dict_to_sorted_list_NA(dict_nodes_x_i)

	counter_for_nodes = 0

	original_xi_values = []

	print(dict_nodes_x_i)

	print(list_to_remove)

	while counter_for_nodes < len(list_to_remove):

		curr_nodes_set = set(list(G.nodes()))

		node = list_to_remove[counter_for_nodes][0]

		print(node,dict_nodes_dBall[node])


		if node not in curr_nodes_set:
			counter_for_nodes += 1
			continue


		(dBall,ball) = get_dBN(G,node,radius) 

		original_xi_values.append(list_to_remove[counter_for_nodes][1])


		if len(dBall) == 0:
			counter_for_nodes += 1
			continue


		size_dball.append(len(dBall))
		size_ball.append(len(ball))

		combined_list = [node] + dBall

		original_degree_main_node.append(original_degree_dict[node])

		for i in dBall:
			original_degree_removed_node.append(original_degree_dict[i])

		#between_list = get_betweenness_score_list(G,combined_list)
		degree_list = get_degree_score_list(G,combined_list)
		#coreness_list = get_coreness_score_list(G,combined_list)

		degree_list_mainNode.append(degree_list[0])
		#betweenness_list_mainNode.append(between_list[0])
		#coreness_list_mainNode.append(coreness_list[0])

		degree_list_removedNode += degree_list[1:]
		#betweenness_list_removedNode += between_list[1:]
		#coreness_list_removedNode += coreness_list[1:]

		for i in dBall:
			G.removeNode(i)
			counter += 1
			
			if i in set_of_connected_nodes:
				connected_nodes_removedNode.append(connected_nodes_removedNode[-1] + 1)

			else:
				connected_nodes_removedNode.append(connected_nodes_removedNode[-1])


		(GC,SGC,num_comp,avg_comp_size) = get_GC_SGC_number_of_components(G)

		GC_List.append(GC)
		SGC_List.append(SGC)
		num_comp_List.append(num_comp)
		avg_comp_size_List.append(avg_comp_size)

		if node in set_of_connected_nodes:
			connected_nodes_mainNode.append(connected_nodes_mainNode[-1] + 1)

		else:

			connected_nodes_mainNode.append(connected_nodes_mainNode[-1])

		counter_list.append(counter)

		counter_for_nodes += 1

	return (GC_List, SGC_List, num_comp_List, avg_comp_size_List, counter_list,size_dball,size_ball,degree_list_mainNode,degree_list_removedNode,original_degree_main_node,original_degree_removed_node, original_xi_values, connected_nodes_mainNode, connected_nodes_removedNode)






def dBalls_attack_adapt(G_copy,radius,set_of_connected_nodes):

	G = copy_graph(G_copy)

	GC_List = []
	SGC_List = []
	num_comp_List = []
	avg_comp_size_List = []

	size_dball = [] 
	size_ball = []

	degree_list_mainNode = []
	betweenness_list_mainNode = []
	coreness_list_mainNode = []

	degree_list_removedNode = []
	betweenness_list_removedNode = []
	coreness_list_removedNode = []

	connected_nodes_mainNode = []
	connected_nodes_removedNode = []

	counter = 0

	counter_list = []

	(GC,SGC,num_comp,avg_comp_size) = get_GC_SGC_number_of_components(G)

	GC_List.append(GC)
	SGC_List.append(SGC)
	num_comp_List.append(num_comp)
	avg_comp_size_List.append(avg_comp_size)
	connected_nodes_mainNode.append(0)
	connected_nodes_removedNode.append(0)

	counter_list.append(counter)

	original_degree_dict = get_degree_dict(G)

	original_degree_main_node = []

	original_degree_removed_node = []

	num_nodes_to_remove = G.numberOfNodes()

	(dict_nodes_dBall,dict_nodes_ball,original_dict_nodes_x_i) = get_all_dBN(G,radius)

	original_xi_values = []

	num_nodes_to_remove = G.numberOfNodes()

	while counter < num_nodes_to_remove:

		print(counter)

		(dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i) = get_all_dBN(G,radius)

		list_to_remove = dict_to_sorted_list(dict_nodes_x_i)

		if len(list_to_remove) == 0:
			break

		node = get_random_dball(list_to_remove)
		(dBall,ball) = get_dBN(G,node,radius) 

		original_xi_values.append(original_dict_nodes_x_i[node])

		size_dball.append(len(dBall))
		size_ball.append(len(ball))

		combined_list = [node] + dBall

		original_degree_main_node.append(original_degree_dict[node])

		for i in dBall:
			original_degree_removed_node.append(original_degree_dict[i])

		#between_list = get_betweenness_score_list(G,combined_list)
		degree_list = get_degree_score_list(G,combined_list)
		#coreness_list = get_coreness_score_list(G,combined_list)

		degree_list_mainNode.append(degree_list[0])
		#betweenness_list_mainNode.append(between_list[0])
		#coreness_list_mainNode.append(coreness_list[0])

		degree_list_removedNode += degree_list[1:]
		#betweenness_list_removedNode += between_list[1:]
		#coreness_list_removedNode += coreness_list[1:]

		for i in dBall:
			G.removeNode(i)
			counter += 1

			if i in set_of_connected_nodes:
				connected_nodes_removedNode.append(connected_nodes_removedNode[-1] + 1)
			else:
				connected_nodes_removedNode.append(connected_nodes_removedNode[-1])

		print(GC)
		
		(GC,SGC,num_comp,avg_comp_size) = get_GC_SGC_number_of_components(G)

		GC_List.append(GC)
		SGC_List.append(SGC)
		num_comp_List.append(num_comp)
		avg_comp_size_List.append(avg_comp_size)

		if node in set_of_connected_nodes:
			connected_nodes_mainNode.append(connected_nodes_mainNode[-1] + 1)

		else:
			connected_nodes_mainNode.append(connected_nodes_mainNode[-1])

		counter_list.append(counter)

	return (GC_List, SGC_List, num_comp_List, avg_comp_size_List, counter_list,size_dball,size_ball,degree_list_mainNode,degree_list_removedNode,original_degree_main_node,original_degree_removed_node, original_xi_values, connected_nodes_mainNode, connected_nodes_removedNode)




def dict_to_sorted_list_NA(d):

	new_list = list(d.items())

	random.shuffle(new_list)

	final_list = sorted(new_list, key = itemgetter(1))

	return final_list



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


def random_ball_removal(G_copy,radius,num_nodes_to_remove):

	G = copy_graph(G_copy)

	counter = 0

	GC_list = []

	size_dball = [] 

	size_ball = []

	continue_counter = 0

	N = G.numberOfNodes()

	while counter < num_nodes_to_remove:

		if continue_counter > (0.1 * N):
			all_nodes = list(G.nodes())
			node_sample = random.sample(all_nodes,(num_nodes_to_remove - counter))
			for i in node_sample:
				G.removeNode(i)
				counter += 1
				GC_list.append(get_GC(G))

			break

		print(counter)

		all_nodes = get_GC_nodes(G)

		node = random.choice(all_nodes)

		(dBall,ball) = get_dBN(G,node,radius)

		if len(dBall) == 0:
			continue_counter += 1
			continue

		size_dball.append(len(dBall))
		size_ball.append(len(ball))

		for i in dBall:
			G.removeNode(i)
			counter += 1
			GC_list.append(get_GC(G))
			continue_counter = 0
		

	return (GC_list,size_dball,size_ball)



def big_sim(N,k,SEED,radius,perc_to_remove,num_sims):

	big_GC_List = []
	big_size_dball = []
	big_size_ball = []
	big_dg_list = []

	for i in range(num_sims):

		G_nx = nx.erdos_renyi_graph(N, k/(N-1), seed = SEED * (i+1)) 

		G_nk = nk.nxadapter.nx2nk(G_nx)

		num_nodes_to_remove = int(perc_to_remove * N)

		(GC_List,size_dball,size_ball,dg_list) = perc_process_dBalls(G_nk,radius,num_nodes_to_remove)

		GC_List_to_append = GC_List[:num_nodes_to_remove]

		big_GC_List.append(GC_List_to_append)

		big_size_dball.append(size_dball)

		big_size_ball.append(size_ball)

		big_dg_list.append(dg_list)

	return (big_GC_List,big_size_dball,big_size_ball,big_dg_list)




def big_sim_dball(N,k,SEED,radius,perc_to_remove,num_sims):

	big_GC_List = []
	big_size_dball = []
	big_size_ball = []
	big_dg_list = []

	for i in range(num_sims):

		G_nx = nx.erdos_renyi_graph(N, k/(N-1), seed = SEED * (i+1)) 

		G_nk = nk.nxadapter.nx2nk(G_nx)

		num_nodes_to_remove = int(perc_to_remove * N)

		(GC_List,size_dball,size_ball,dg_list) = perc_process_dBalls_bigDBalls(G_nk,radius,num_nodes_to_remove)

		GC_List_to_append = GC_List[:num_nodes_to_remove]

		big_GC_List.append(GC_List_to_append)

		big_size_dball.append(size_dball)

		big_size_ball.append(size_ball)

		big_dg_list.append(dg_list)

	return (big_GC_List,big_size_dball,big_size_ball,big_dg_list)



def big_sim_SF(N,k,exp_out,radius,perc_to_remove,num_sims):

	big_GC_List = []

	big_size_ball = []

	big_size_dball = []

	big_dg_list = []

	for i in range(num_sims):

		G_nk = make_SF_Graph(N,k,exp_out)

		num_nodes_to_remove = int(perc_to_remove * N)

		(GC_List,size_dball,size_ball,degree_list) = perc_process_dBalls(G_nk,radius,num_nodes_to_remove)

		GC_List_to_append = GC_List[:num_nodes_to_remove]

		big_GC_List.append(GC_List_to_append)

		big_size_ball.append(size_ball)

		big_size_dball.append(size_dball)

		big_dg_list.append(degree_list)

	return (big_GC_List,big_size_dball,big_size_ball,big_dg_list)




def big_sim_changing_radius(G,start_radius,end_radius):

	big_GC_List = []
	big_counter_list = []

	curr_radius = start_radius 

	while curr_radius <= end_radius:

		(GC_List,size_dball,size_ball,degree_list,counter_list) = perc_process_dBalls_track_balls(G,curr_radius)

		big_GC_List.append(GC_List)

		big_counter_list.append(counter_list)

		curr_radius += 1

	return (big_GC_List,big_counter_list)



def get_results_NA(G, radius):

	N = G.numberOfNodes()

	GC_list_DA = DA_attack(G, int(N * 0.99))

	GC_list_BA = BA_attack(G, int(N * 0.99))
 
	GC_list_RAN = big_RA_attack(G,int(N * 0.99),20)

	(GC_List_DB,size_dball,size_ball,degree_list,counter_list) = dBalls_attack_NA(G_copy,radius)

	return (GC_list_BA, GC_list_DA, GC_list_RAN, GC_List_DB)



def get_result(G, radius):

	N = G.numberOfNodes()

	(GC_List_DB, SGC_List_DB,num_comp_List_DB,counter_list,size_dball,size_ball,degree_list_mainNode,betweenness_list_mainNode,coreness_list_mainNode,degree_list_removedNode,betweenness_list_removedNode,coreness_list_removedNode) = dBalls_attack(G,radius)

	return (GC_List_DB, SGC_List_DB,num_comp_List_DB,counter_list,size_dball,size_ball,degree_list_mainNode,betweenness_list_mainNode,coreness_list_mainNode,degree_list_removedNode,betweenness_list_removedNode,coreness_list_removedNode)




N=int(sys.argv[1]) 

k_intra=float(sys.argv[2])

k_inter=float(sys.argv[3])

SEED=int(sys.argv[4])

num_modules = int(sys.argv[5])

alpha = float(sys.argv[6])

radius = int(sys.argv[7])

num_times = int(sys.argv[8])


type_graph = "MOD"

adaptive_type = "ADAPT"



for i in range(num_times):

	SEED = (SEED * (i+1)) +1

	(G, set_connected_nodes) = make_modular_network_ER(N,k_intra,k_inter,num_modules,SEED,alpha)

	(GC_List, SGC_List, num_comp_List, avg_comp_size_List, counter_list,size_dball,size_ball,degree_list_mainNode,degree_list_removedNode,original_degree_main_node,original_degree_removed_node, original_xi_values, connected_nodes_mainNode, connected_nodes_removedNode) = dBalls_attack_adapt(G,radius,set_connected_nodes)

	init_name_GC_DB = adaptive_type + "SGCattackDB_" + type_graph +"_GC"

	init_name_dball = adaptive_type +  "SGCattackDB_" + type_graph +"_DBALL"
	init_name_ball = adaptive_type +  "SGCattackDB_" + type_graph +"_BALL"

	init_name_CL = adaptive_type +  "SGCattackDB_" + type_graph +"_CL"

	init_name_deg_mainNode = adaptive_type +  "SGCattackDB_" + type_graph +"_degMainNode"
	init_name_deg_removedNode = adaptive_type + "SGCattackDB_" + type_graph +"_degRemovedNode"

	init_name_SGC_DB = adaptive_type + "SGCattackDB_" + type_graph +"_SGC"

	init_name_numComp_DB = adaptive_type + "SGCattackDB_" + type_graph +"_numberOfComponents"

	init_name_avgSize_DB = adaptive_type + "SGCattackDB_" + type_graph +"_avgComponents"

	init_name_original_degree_main_node = adaptive_type + "SGCattackDB_" + type_graph + "_originalDegreeMainNode"
	init_name_original_degree_removed_node = adaptive_type + "SGCattackDB_" + type_graph + "_originalDegreeRemovedNode"

	init_name_original_xi_values = adaptive_type + "SGCattackDB_" + type_graph + "_originalXIValues"

	init_name_connectedNode_MainNode = adaptive_type + "SGCattackDB_" + type_graph + "_connectedNodesMainNode"

	init_name_connectedNode_RemovedNode = adaptive_type + "SGCattackDB_" + type_graph + "_connectedNodesRemovedNode"


	GC_List_DB_name = get_name_ModularNetworks_alpha(init_name_GC_DB, N,k_intra,k_inter,alpha,SEED,num_modules,radius)

	CL_name = get_name_ModularNetworks_alpha(init_name_CL, N,k_intra,k_inter,alpha,SEED,num_modules,radius)

	dBall_name = get_name_ModularNetworks_alpha(init_name_dball, N,k_intra,k_inter,alpha,SEED,num_modules,radius)
	ball_name = get_name_ModularNetworks_alpha(init_name_ball, N,k_intra,k_inter,alpha,SEED,num_modules,radius)

	deg_mainNode_name = get_name_ModularNetworks_alpha(init_name_deg_mainNode, N,k_intra,k_inter,alpha,SEED,num_modules,radius)
	deg_removedNode_name = get_name_ModularNetworks_alpha(init_name_deg_removedNode, N,k_intra,k_inter,alpha,SEED,num_modules,radius)

	SGC_DB_name = get_name_ModularNetworks_alpha(init_name_SGC_DB, N,k_intra,k_inter,alpha,SEED,num_modules,radius)

	numComp_DB_name = get_name_ModularNetworks_alpha(init_name_numComp_DB, N,k_intra,k_inter,alpha,SEED,num_modules,radius)

	avgComp_DB_name = get_name_ModularNetworks_alpha(init_name_avgSize_DB, N,k_intra,k_inter,alpha,SEED,num_modules,radius)

	original_degree_main_node_name = get_name_ModularNetworks_alpha(init_name_original_degree_main_node, N,k_intra,k_inter,alpha,SEED,num_modules,radius)
	original_degree_removed_node_name = get_name_ModularNetworks_alpha(init_name_original_degree_removed_node, N,k_intra,k_inter,alpha,SEED,num_modules,radius)

	original_xi_values_name = get_name_ModularNetworks_alpha(init_name_original_xi_values, N,k_intra,k_inter,alpha,SEED,num_modules,radius)

	connectedNode_MainNode_name = get_name_ModularNetworks_alpha(init_name_connectedNode_MainNode, N,k_intra,k_inter,alpha,SEED,num_modules,radius)
	connectedNode_RemovedNode_name = get_name_ModularNetworks_alpha(init_name_connectedNode_RemovedNode, N,k_intra,k_inter,alpha,SEED,num_modules,radius)


	with open(GC_List_DB_name,'wb') as handle:
		pickle.dump(GC_List, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(CL_name,'wb') as handle:
		pickle.dump(counter_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(dBall_name,'wb') as handle:
		pickle.dump(size_dball, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(ball_name,'wb') as handle:
		pickle.dump(size_ball, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(deg_mainNode_name,'wb') as handle:
		pickle.dump(degree_list_mainNode, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(deg_removedNode_name,'wb') as handle:
		pickle.dump(degree_list_removedNode, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(SGC_DB_name,'wb') as handle:
		pickle.dump(SGC_List, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(numComp_DB_name,'wb') as handle:
		pickle.dump(num_comp_List, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(avgComp_DB_name,'wb') as handle:
		pickle.dump(avg_comp_size_List, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(original_degree_main_node_name,'wb') as handle:
		pickle.dump(original_degree_main_node, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(original_degree_removed_node_name,'wb') as handle:
		pickle.dump(original_degree_removed_node, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(original_xi_values_name,'wb') as handle:
		pickle.dump(original_xi_values, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(connectedNode_MainNode_name,'wb') as handle:
		pickle.dump(connected_nodes_mainNode, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(connectedNode_RemovedNode_name,'wb') as handle:
		pickle.dump(connected_nodes_removedNode, handle, protocol=pickle.HIGHEST_PROTOCOL)


