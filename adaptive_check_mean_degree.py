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

	print("DA_attack")

	G = copy_graph(G_copy)

	GC_List = []

	SGC_List = []

	num_comp_List = []

	original_degree_list = []

	adaptive_degree_list = []

	(GC,SGC,num_comp) = get_GC_SGC_number_of_components(G)

	GC_List.append(GC)

	SGC_List.append(SGC)

	num_comp_List.append(num_comp)

	degree = nk.centrality.DegreeCentrality(G)

	degree.run()

	degree_sequence = degree.ranking()

	random.shuffle(degree_sequence)

	degree_sequence.sort(key = itemgetter(1), reverse = True)

	mean_degree_list = []
	mean_degree_list_GC = []

	removed_nodes = []


	for i in range(num_nodes_to_remove):

		mean_deg = calculate_mean_degree(G)

		mean_deg_GC = calculate_mean_degree_GC(G)

		print("mean degree : " + str(i))

		print(mean_deg)

		print("mean degree GC : " + str(i))

		print(mean_deg_GC)

		mean_degree_list.append(mean_deg)
		mean_degree_list_GC.append(mean_deg_GC)


		node_to_remove = degree_sequence[i][0]

		original_degree = degree_sequence[i][1]

		adaptive_degree_list.append(G.degree(node_to_remove))

		original_degree_list.append(original_degree)

		G.removeNode(node_to_remove)

		(GC,SGC,num_comp) = get_GC_SGC_number_of_components(G)

		GC_List.append(GC)

		SGC_List.append(SGC)

		num_comp_List.append(num_comp)

		removed_nodes.append(node_to_remove)


	

	return (GC_List, SGC_List, num_comp_List, original_degree_list,adaptive_degree_list,mean_degree_list,mean_degree_list_GC,removed_nodes)


def ADA_attack(G_copy,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []

	SGC_List = []

	num_comp_List = []

	avg_comp_size_List = []

	(GC,SGC,num_comp,avg_comp_size) = get_GC_SGC_number_of_components(G)

	GC_List.append(GC)

	SGC_List.append(SGC)

	num_comp_List.append(num_comp)

	avg_comp_size_List.append(avg_comp_size)

	degree_list = []

	mean_degree_list = []
	mean_degree_list_GC = []

	removed_nodes = []

	for i in range(num_nodes_to_remove):

		mean_deg = calculate_mean_degree(G)

		mean_deg_GC = calculate_mean_degree_GC(G)

		print("mean degree : " + str(i))

		print(mean_deg)

		print("mean degree GC : " + str(i))

		print(mean_deg_GC)

		mean_degree_list.append(mean_deg)
		mean_degree_list_GC.append(mean_deg_GC)

		degree = nk.centrality.DegreeCentrality(G)

		degree.run()

		degree_sequence = degree.ranking()

		random.shuffle(degree_sequence)

		degree_sequence.sort(key = itemgetter(1), reverse = True)

		node_to_remove = degree_sequence[0][0]

		degree_list.append(G.degree(node_to_remove))

		G.removeNode(node_to_remove)

		#(GC,SGC,num_comp,avg_comp_size) = get_GC_SGC_number_of_components(G)

		#GC_List.append(GC)

		#SGC_List.append(SGC)

		#num_comp_List.append(num_comp)

		#avg_comp_size_List.append(avg_comp_size)

	return (GC_List, SGC_List, num_comp_List,avg_comp_size_List, degree_list, mean_degree_list, mean_degree_list_GC)


def BA_attack(G_copy,num_nodes_to_remove):

	print("BA_attack")

	G = copy_graph(G_copy)

	GC_List = []

	SGC_List = []

	num_comp_List = []

	(GC,SGC,num_comp) = get_GC_SGC_number_of_components(G)

	GC_List.append(GC)

	SGC_List.append(SGC)

	num_comp_List.append(num_comp)

	between = nk.centrality.DynBetweenness(G)
	between.run()

	between_sequence = between.ranking()

	random.shuffle(between_sequence)

	between_sequence.sort(key = itemgetter(1), reverse = True)

	between_list = []

	for i in range(num_nodes_to_remove):

		print("BA_attack")
		print(i)
		
		node_to_remove = between_sequence[i][0]

		between_score =  between_sequence[i][1]

		between_list.append(between_score)

		G.removeNode(node_to_remove)

		(GC,SGC,num_comp) = get_GC_SGC_number_of_components(G)

		GC_List.append(GC)

		SGC_List.append(SGC)

		num_comp_List.append(num_comp)

	return (GC_List, SGC_List, num_comp_List, between_list)


def turn_nk_to_igraph(G):

	G_i = ig.Graph()

	nodes_list = list(G.nodes())
	edges_list = list(G.edges())

	G_i.add_vertices(nodes_list)
	G_i.add_edges(edges_list)

	return G_i


def betweenness_igraph(G):

	between_list = G.betweenness(directed = False)

	between_dict = []

	print(between_list)

	for i in range(len(between_list)):

		between_dict.append((i,between_list[i]))

	return between_dict



def BA_attack_igraph(G_copy,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []

	SGC_List = []

	num_comp_List = []

	avg_comp_size_List = []

	(GC,SGC,num_comp,avg_comp_size) = get_GC_SGC_number_of_components(G)

	GC_List.append(GC)

	SGC_List.append(SGC)

	num_comp_List.append(num_comp)

	avg_comp_size_List.append(avg_comp_size)

	G_i = turn_nk_to_igraph(G)

	between_sequence = betweenness_igraph(G_i)

	random.shuffle(between_sequence)

	between_sequence.sort(key = itemgetter(1), reverse = True)

	print(between_sequence)
	
	for i in range(num_nodes_to_remove):
		
		node_to_remove = between_sequence[i][0]

		between_score =  between_sequence[i][1]

		print(i, node_to_remove, between_score)

		G.removeNode(node_to_remove)

		#(GC,SGC,num_comp,avg_comp_size) = get_GC_SGC_number_of_components(G)

		GC_List.append(GC)

		SGC_List.append(SGC)

		num_comp_List.append(num_comp)

		avg_comp_size_List.append(avg_comp_size)

	return (GC_List, SGC_List, num_comp_List, avg_comp_size_List)


def ABA_attack(G_copy,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []

	SGC_List = []

	num_comp_List = []

	avg_comp_size_List = []

	(GC,SGC,num_comp, avg_Comp) = get_GC_SGC_number_of_components(G)

	avg_comp_size_List.append(avg_Comp)

	GC_List.append(GC)

	SGC_List.append(SGC)

	num_comp_List.append(num_comp)

	mean_degree_list = []
	mean_degree_list_GC = []

	removed_nodes = []

	G_i = turn_nk_to_igraph(G)

	for i in range(num_nodes_to_remove):

		mean_deg = calculate_mean_degree(G)

		mean_deg_GC = calculate_mean_degree_GC(G)

		mean_degree_list.append(mean_deg)
		mean_degree_list_GC.append(mean_deg_GC)

		between_sequence = betweenness_igraph(G_i)

		random.shuffle(between_sequence)

		between_sequence.sort(key = itemgetter(1), reverse = True)

		node_to_remove = between_sequence[0][0]

		G.removeNode(node_to_remove)

		G_i.delete_vertices(node_to_remove)

		(GC,SGC,num_comp,avg_comp_size) = get_GC_SGC_number_of_components(G)

		GC_List.append(GC)

		SGC_List.append(SGC)

		num_comp_List.append(num_comp)

		avg_comp_size_List.append(avg_comp_size)

	return (GC_List, SGC_List, num_comp_List, avg_comp_size_List, mean_degree_list, mean_degree_list_GC)



def ABA_attack_igraph(G, G_i, num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []

	SGC_List = []

	num_comp_List = []

	avg_comp_size_List = []

	(GC,SGC,num_comp, avg_Comp) = get_GC_SGC_number_of_components(G)

	avg_comp_size_List.append(avg_Comp)

	GC_List.append(GC)

	SGC_List.append(SGC)

	num_comp_List.append(num_comp)

	mean_degree_list = []
	mean_degree_list_GC = []

	removed_nodes = []

	for i in range(num_nodes_to_remove):

		mean_deg = calculate_mean_degree(G)

		mean_deg_GC = calculate_mean_degree_GC(G)

		mean_degree_list.append(mean_deg)
		mean_degree_list_GC.append(mean_deg_GC)

		between_sequence = betweenness_igraph(G_i)

		random.shuffle(between_sequence)

		between_sequence.sort(key = itemgetter(1), reverse = True)

		node_to_remove = between_sequence[0][0]

		G.removeNode(node_to_remove)

		G_i.delete_vertices(node_to_remove)

		(GC,SGC,num_comp,avg_comp_size) = get_GC_SGC_number_of_components(G)

		GC_List.append(GC)

		SGC_List.append(SGC)

		num_comp_List.append(num_comp)

		avg_comp_size_List.append(avg_comp_size)

	return (GC_List, SGC_List, num_comp_List, avg_comp_size_List, mean_degree_list, mean_degree_list_GC)



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

		print("RA")
		print(i)

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



def get_degree_dict(G):

	all_nodes = list(G.nodes())

	final_dict = {}

	for i in all_nodes:

		final_dict[i] = G.degree(i)

	return final_dict




#dball, vball, degree, betweenness, coreness
def dBalls_attack(G_copy,radius):

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

	num_nodes_to_remove = G.numberOfNodes()

	mean_degree_list = []

	mean_degree_list_GC = []

	mean_deg = calculate_mean_degree(G)

	mean_deg_GC = calculate_mean_degree_GC(G)

	mean_degree_list.append(mean_deg)
	mean_degree_list_GC.append(mean_deg_GC)


	while counter < num_nodes_to_remove:

		#print(counter)

		(dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i) = get_all_dBN(G,radius)

		list_to_remove = dict_to_sorted_list(dict_nodes_x_i)

		if len(list_to_remove) == 0:
			break
		

		node = get_random_dball(list_to_remove)

		(dBall,ball) = get_dBN(G,node,radius) 

		combined_list = [node] + dBall

		#between_list = get_betweenness_score_list(G,combined_list)
		#degree_list = get_degree_score_list(G,combined_list)
		#coreness_list = get_coreness_score_list(G,combined_list)



		#degree_list_mainNode.append(degree_list[0])
		#betweenness_list_mainNode.append(between_list[0])
		#coreness_list_mainNode.append(coreness_list[0])

		#degree_list_removedNode += degree_list[1:]
		#betweenness_list_removedNode += between_list[1:]
		#coreness_list_removedNode += coreness_list[1:]
		

		size_dball.append(len(dBall))
		size_ball.append(len(ball))


		#print(dBall)
		#print(ball)

		for i in dBall:
			G.removeNode(i)
			counter += 1

		#(GC,SGC,num_comp,avg_comp_size) = get_GC_SGC_number_of_components(G)

		#GC_List.append(GC)
		#SGC_List.append(SGC)
		#num_comp_List.append(num_comp)
		#avg_comp_size_List.append(avg_comp_size)

		counter_list.append(counter)

		mean_deg = calculate_mean_degree(G)

		mean_deg_GC = calculate_mean_degree_GC(G)

		print("mean degree : " + str(counter))

		print(mean_deg)

		print("mean degree GC : " + str(counter))

		print(mean_deg_GC)

		mean_degree_list.append(mean_deg)
		mean_degree_list_GC.append(mean_deg_GC)


	return (GC_List,SGC_List,num_comp_List,avg_comp_size_List,counter_list,size_dball,size_ball,degree_list_mainNode,betweenness_list_mainNode,coreness_list_mainNode,degree_list_removedNode,betweenness_list_removedNode,coreness_list_removedNode, mean_degree_list, mean_degree_list_GC)





def dBalls_attack_NA(G_copy,radius):

	print("dball attack")

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

	(GC,SGC,num_comp,avg_comp_size) = get_GC_SGC_number_of_components(G)

	GC_List.append(GC)
	SGC_List.append(SGC)
	num_comp_List.append(num_comp)

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

	mean_degree_list = []
	mean_degree_list_GC = []

	mean_deg = calculate_mean_degree(G)
	mean_deg_GC = calculate_mean_degree_GC(G)

	mean_degree_list.append(mean_deg)
	mean_degree_list_GC.append(mean_deg_GC)

	removed_nodes = []

	while counter_for_nodes < len(list_to_remove):

		
		#print(counter_for_nodes)
		
		curr_nodes_set = set(list(G.nodes()))

		node = list_to_remove[counter_for_nodes][0]

		#print(node,dict_nodes_dBall[node])

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

		removed_nodes += dBall

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


		counter_list.append(counter)

		counter_for_nodes += 1

		mean_deg = calculate_mean_degree(G)

		mean_deg_GC = calculate_mean_degree_GC(G)

		print("mean degree : " + str(counter))

		print(mean_deg)

		print("mean degree GC : " + str(counter))

		print(mean_deg_GC)

		mean_degree_list.append(mean_deg)
		mean_degree_list_GC.append(mean_deg_GC)


	return (GC_List, SGC_List, num_comp_List, counter_list,size_dball,size_ball,degree_list_mainNode,degree_list_removedNode,original_degree_main_node,original_degree_removed_node, original_xi_values, mean_degree_list, mean_degree_list_GC,removed_nodes)





def new_optimal_attack(G_copy,radius,mean_deg_threshold):

	print("dball attack")

	G = copy_graph(G_copy)

	avg_comp_size_List = []

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

	(GC,SGC,num_comp,avg_comp_size) = get_GC_SGC_number_of_components(G)

	GC_List.append(GC)
	SGC_List.append(SGC)
	num_comp_List.append(num_comp)

	counter_list.append(counter)

	original_degree_dict = get_degree_dict(G)

	original_degree_main_node = []

	original_degree_removed_node = []

	num_nodes_to_remove = G.numberOfNodes()

	counter_for_nodes = 0

	original_xi_values = []

	mean_degree_list = []
	mean_degree_list_GC = []

	mean_deg = calculate_mean_degree(G)
	mean_deg_GC = calculate_mean_degree_GC(G)

	mean_degree_list.append(mean_deg)
	mean_degree_list_GC.append(mean_deg_GC)

	removed_nodes = []

	G_i = turn_nk_to_igraph(G)

	vertexSequence = ig.VertexSeq(G_i)

	print(vertexSequence)

	G_i.degree(993)


	while counter < num_nodes_to_remove:

		#print(counter)

		(dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i) = get_all_dBN(G,radius)

		list_to_remove = dict_to_sorted_list(dict_nodes_x_i)

		if len(list_to_remove) == 0:
			break
		

		node = get_random_dball(list_to_remove)
		(dBall,ball) = get_dBN(G,node,radius) 

		combined_list = [node] + dBall

		#between_list = get_betweenness_score_list(G,combined_list)
		#degree_list = get_degree_score_list(G,combined_list)
		#coreness_list = get_coreness_score_list(G,combined_list)



		#degree_list_mainNode.append(degree_list[0])
		#betweenness_list_mainNode.append(between_list[0])
		#coreness_list_mainNode.append(coreness_list[0])

		#degree_list_removedNode += degree_list[1:]
		#betweenness_list_removedNode += between_list[1:]
		#coreness_list_removedNode += coreness_list[1:]
		

		size_dball.append(len(dBall))
		size_ball.append(len(ball))


		#print(dBall)
		#print(ball)

		

		for i in dBall:
			#print(i)
			G.removeNode(i)
			#G_i.delete_vertices(i)
			counter += 1

		print("counter:" + str(counter))
		print(G_i.vcount())

		print(dBall)

		G_i.delete_vertices(dBall)

		(GC,SGC,num_comp,avg_comp_size) = get_GC_SGC_number_of_components(G)

		GC_List.append(GC)
		SGC_List.append(SGC)
		num_comp_List.append(num_comp)
		avg_comp_size_List.append(avg_comp_size)

		counter_list.append(counter)

		mean_deg = calculate_mean_degree(G)

		mean_deg_GC = calculate_mean_degree_GC(G)

		mean_degree_list.append(mean_deg)
		mean_degree_list_GC.append(mean_deg_GC)

		if mean_deg <= mean_deg_threshold:
			 
			(GC_List_DA, SGC_List_DA, num_comp_List_DA, avg_comp_size_List_DA, mean_degree_list_DA, mean_degree_list_GC_DA) = ABA_attack_igraph(G, G_i, (int(G.numberOfNodes() * 0.9)))
			break

	GC_List_DA_Length = len(GC_List_DA[1:])

	for i in range(GC_List_DA_Length):

		counter_list.append(counter_list[-1] + 1)



	GC_List += GC_List_DA[1:]
	SGC_List += SGC_List_DA[1:]
	num_comp_List += num_comp_List_DA[1:]


	return (GC_List, SGC_List, num_comp_List, counter_list,size_dball,size_ball,degree_list_mainNode,degree_list_removedNode,original_degree_main_node,original_degree_removed_node, original_xi_values, mean_degree_list, mean_degree_list_GC,removed_nodes)





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

	(GC_List_DA, SGC_List_DA, num_comp_List_DA, original_degree_list,adaptive_degree_list) = DA_attack(G, int(N * 0.8))

	(GC_List_BA, SGC_List_BA, num_comp_List_BA, between_list) = BA_attack(G, int(N * 0.8))
 
	(avg_list_GC_RAN, avg_list_SGC_RAN, avg_list_numComp_RAN)  = big_RA_attack(G,int(N * 0.99),1)

	(GC_List_DB, SGC_List_DB, num_comp_List_DB, counter_list,size_dball,size_ball,degree_list_mainNode,degree_list_removedNode,original_degree_main_node,original_degree_removed_node,original_xi_values) = dBalls_attack_NA(G, radius)

	return (GC_List_DA, SGC_List_DA, num_comp_List_DA, original_degree_list,adaptive_degree_list, GC_List_BA, SGC_List_BA, num_comp_List_BA, between_list, avg_list_GC_RAN, avg_list_SGC_RAN, avg_list_numComp_RAN,GC_List_DB, SGC_List_DB, num_comp_List_DB, counter_list,size_dball,size_ball,degree_list_mainNode,degree_list_removedNode,original_degree_main_node,original_degree_removed_node,original_xi_values)




def get_result(G, radius):

	N = G.numberOfNodes()

	(GC_List_ADA, SGC_List_ADA, num_comp_List_ADA, degree_list) = ADA_attack(G, int(N * 0.99))

	(GC_List_ABA, SGC_List_ABA, num_comp_List_ABA) = ABA_attack(G, int(N * 0.99))
 
	(GC_List_RAN, SGC_List_RAN, num_comp_List_RAN) = big_RA_attack(G,int(N * 0.99),20)

	(GC_List_DB, SGC_List_DB,num_comp_List_DB,counter_list,size_dball,size_ball,degree_list_mainNode,betweenness_list_mainNode,coreness_list_mainNode,degree_list_removedNode,betweenness_list_removedNode,coreness_list_removedNode) = dBalls_attack(G,radius)

	return (GC_List_ADA, SGC_List_ADA, num_comp_List_ADA, degree_list, GC_List_ABA, SGC_List_ABA, num_comp_List_ABA, GC_List_RAN, SGC_List_RAN, num_comp_List_RAN, GC_List_DB,SGC_List_DB,num_comp_List_DB,counter_list,size_dball,size_ball,degree_list_mainNode,betweenness_list_mainNode,coreness_list_mainNode,degree_list_removedNode,betweenness_list_removedNode,coreness_list_removedNode)


def calculate_mean_degree(G):

	num_nodes = G.numberOfNodes()
	num_edges = G.numberOfEdges()

	return (2 * num_edges) / num_nodes

def calculate_mean_degree_GC(G):

	GC_nodes = get_GC_nodes(G)

	num_edges = 0

	for i in GC_nodes:

		num_edges += G.degree(i)

	num_nodes = len(GC_nodes)

	num_edges = int(num_edges / 2)

	return (2 * num_edges) / num_nodes


def intersection(l1,l2):

	l3 = [value for value in l1 if value in l2]

	return l3

def get_increasing_intersect(delta):

	counter = 0
	delta = 100

	intersection_list = []

	intersection_list_nodes = []

	while counter < len(nodes_removed_DB):

		curr_list_DB = nodes_removed_DB[: (counter + delta)]
		curr_list_DA = nodes_removed_DA[: (counter + delta)]

		counter += delta

		intersect = intersection(curr_list_DB,curr_list_DA)

		intersection_list_nodes.append(intersect)

		intersection_list.append(len(intersect))



def get_intersect_given_fs(fs):

	index = fs / 100








N=int(sys.argv[1]) 

k=float(sys.argv[2])

SEED=int(sys.argv[3])

radius = int(sys.argv[4])

num_times = int(sys.argv[5])

threshold = float(sys.argv[6])

adaptive_type = "ADAPT_"

for i in range(num_times):

	SEED = (SEED * (i+1)) + 1

	G_copy = make_ER_Graph(N,k,SEED)

	(GC_List_DB, SGC_List, num_comp_List, counter_list_DB,size_dball,size_ball,degree_list_mainNode,degree_list_removedNode,original_degree_main_node,original_degree_removed_node, original_xi_values, mean_degree_list, mean_degree_list_GC,removed_nodes) = new_optimal_attack(G_copy,radius,threshold)
	
	init_CL_DB_name = adaptive_type + "SGCattackDB_ER_CL_threshold_" + str(threshold)
	init_GC_DB_name = adaptive_type + "SGCattackDB_ER_GC_threshold_" + str(threshold)

	CL_name = get_name_ER(init_CL_DB_name, N, k, SEED,radius)

	GC_name = get_name_ER(init_GC_DB_name, N, k, SEED,radius)

	with open(CL_name,'wb') as handle:
		pickle.dump(counter_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(GC_name,'wb') as handle:
		pickle.dump(GC_List, handle, protocol=pickle.HIGHEST_PROTOCOL)


"""

G = make_ER_Graph(N, k ,SEED)

(GC_List, SGC_List, num_comp_List, avg_comp_size_List, mean_degree_list_BA, mean_degree_list_GC_BA) = ABA_attack(G,int(N*0.8))
(GC_List, SGC_List, num_comp_List,avg_comp_size_List, degree_list, mean_degree_list_DA, mean_degree_list_GC_DA) = ADA_attack(G,int(N*0.8))
(GC_List,SGC_List,num_comp_List,avg_comp_size_List,counter_list,size_dball,size_ball,degree_list_mainNode,betweenness_list_mainNode,coreness_list_mainNode,degree_list_removedNode,betweenness_list_removedNode,coreness_list_removedNode, mean_degree_list_DB, mean_degree_list_GC_DB) = dBalls_attack(G,radius)


figname = "ADAPTchangingMeanDeg" + "_N_" + str(N) + "_k_" + str(k) + "_SEED_" + str(SEED) + "_radius_" + str(radius) + ".png" 

x_list = [i for i in range(len(mean_degree_list_DA))]

print(len(mean_degree_list_DB))
print(len(mean_degree_list_GC_DB))

print(len(mean_degree_list_DA))
print(len(mean_degree_list_GC_DA))

print(len(mean_degree_list_BA))
print(len(mean_degree_list_GC_BA))

plt.xlabel('nodes_removed', fontsize=20)
plt.ylabel('mean_degree', fontsize=20, rotation=0, labelpad=20)
plt.title("chaning mean deg", fontsize=20)

plt.plot(counter_list, mean_degree_list_DB, label = "dball full Graph")
plt.plot(counter_list, mean_degree_list_GC_DB, label = "dball GC")

plt.plot(x_list, mean_degree_list_DA, label = "DA full Graph")
plt.plot(x_list, mean_degree_list_GC_DA, label = "DA GC")

plt.plot(x_list, mean_degree_list_BA, label = "BA full Graph")
plt.plot(x_list, mean_degree_list_GC_BA, label = "BA GC")

#plt.xlim(0,0.2)
plt.legend(loc='best')
plt.tight_layout() 
plt.savefig(figname)
plt.clf()

"""

