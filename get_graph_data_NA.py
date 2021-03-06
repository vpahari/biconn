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

	G = copy_graph(G_copy)

	GC_List = []

	GC_List.append(get_GC(G))

	degree = nk.centrality.DegreeCentrality(G)

	degree.run()

	degree_sequence = degree.ranking()

	random.shuffle(degree_sequence)

	degree_sequence.sort(key = itemgetter(1), reverse = True)

	for i in range(num_nodes_to_remove):

		node_to_remove = degree_sequence[i][0]

		G.removeNode(node_to_remove)

		GC_List.append(get_GC(G))

	return GC_List


def ADA_attack(G_copy,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []

	GC_List.append(get_GC(G))

	for i in range(num_nodes_to_remove):

		print(i)

		degree = nk.centrality.DegreeCentrality(G)

		degree.run()

		degree_sequence = degree.ranking()

		random.shuffle(degree_sequence)

		degree_sequence.sort(key = itemgetter(1), reverse = True)

		node_to_remove = degree_sequence[0][0]

		G.removeNode(node_to_remove)

		GC_List.append(get_GC(G))

	return GC_List


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

	GC_List.append(get_GC(G))

	for i in range(num_nodes_to_remove):

		print(i)

		between = nk.centrality.DynBetweenness(G)
		between.run()

		between_sequence = between.ranking()

		between_sequence.sort(key = itemgetter(1), reverse = True)

		node_to_remove = between_sequence[0][0]

		G.removeNode(node_to_remove)

		GC_List.append(get_GC(G))

	return GC_List



def RA_attack(G_copy,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []

	GC_List.append(get_GC(G))

	all_nodes = random.sample(list(G.nodes()),num_nodes_to_remove)

	for i in all_nodes:
		G.removeNode(i)
		GC_List.append(get_GC(G))

	return GC_List



def big_RA_attack(G_copy,num_nodes_to_remove,num_sims):

	big_GC_List = []

	for i in range(num_sims):

		GC_list = RA_attack(G_copy,num_nodes_to_remove)

		big_GC_List.append(GC_list)

	avg_list = get_avg_list(big_GC_List)

	return avg_list 



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

	GC_List.append(get_GC(G))
	counter_list.append(counter)

	num_nodes_to_remove = G.numberOfNodes()

	list_for_nodes_dball = []
	list_for_nodes_ball = []

	main_node_track = []

	while counter < num_nodes_to_remove:

		print(counter)

		(dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i) = get_all_dBN(G,radius)

		list_to_remove = dict_to_sorted_list(dict_nodes_x_i)

		if len(list_to_remove) == 0:
			break
		

		node = get_random_dball(list_to_remove)
		(dBall,ball) = get_dBN(G,node,radius) 
		

		size_dball.append(len(dBall))
		size_ball.append(len(ball))


		list_for_nodes_dball.append(dBall)
		list_for_nodes_ball.append(ball)

		main_node_track.append(node)

		#print(dBall)
		#print(ball)

		for i in dBall:
			G.removeNode(i)
			counter += 1

		GC_List.append(get_GC(G))

		counter_list.append(counter)


	return (GC_List,counter_list,size_dball,size_ball,list_for_nodes_dball,list_for_nodes_ball,main_node_track)



def dBalls_attack_NA(G_copy,radius):

	G = copy_graph(G_copy)

	GC_List = []
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

	GC_List.append(get_GC(G))
	counter_list.append(counter)

	num_nodes_to_remove = G.numberOfNodes()

	(dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i) = get_all_dBN(G,radius)

	list_to_remove = dict_to_sorted_list_NA(dict_nodes_x_i)

	counter_for_nodes = 0

	list_for_nodes_dball = []
	list_for_nodes_ball = []

	print(dict_nodes_x_i)

	print(list_to_remove)

	main_node_track = []

	while counter_for_nodes < len(list_to_remove):

		curr_nodes_set = set(list(G.nodes()))

		node = list_to_remove[counter_for_nodes][0]

		print(node,dict_nodes_dBall[node])


		if node not in curr_nodes_set:
			counter_for_nodes += 1
			continue


		(dBall,ball) = get_dBN(G,node,radius) 


		if len(dBall) == 0:
			counter_for_nodes += 1
			continue


		size_dball.append(len(dBall))
		size_ball.append(len(ball))


		main_node_track.append(node)

		for i in dBall:
			G.removeNode(i)
			counter += 1

		GC_List.append(get_GC(G))

		counter_list.append(counter)

		counter_for_nodes += 1

		list_for_nodes_dball.append(dBall)
		list_for_nodes_ball.append(ball)


	return (GC_List,counter_list,size_dball,size_ball,list_for_nodes_dball,list_for_nodes_ball,main_node_track)



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

	(GC_List_DB,counter_list,size_dball,size_ball,degree_list_mainNode,betweenness_list_mainNode,coreness_list_mainNode,degree_list_removedNode,betweenness_list_removedNode,coreness_list_removedNode) = dBalls_attack_NA(G,radius)

	return (GC_list_DA, GC_list_BA, GC_list_RAN, GC_List_DB, counter_list, size_dball, size_ball, degree_list_mainNode, betweenness_list_mainNode, coreness_list_mainNode, degree_list_removedNode, betweenness_list_removedNode, coreness_list_removedNode)



def get_result(G, radius):

	N = G.numberOfNodes()

	GC_list_ADA = ADA_attack(G, int(N * 0.99))

	GC_list_ABA = ABA_attack(G, int(N * 0.99))
 
	GC_list_RAN = big_RA_attack(G,int(N * 0.99),20)

	(GC_List_DB,counter_list,size_dball,size_ball,degree_list_mainNode,betweenness_list_mainNode,coreness_list_mainNode,degree_list_removedNode,betweenness_list_removedNode,coreness_list_removedNode) = dBalls_attack(G,radius)

	return (GC_list_ADA, GC_list_ABA, GC_list_RAN, GC_List_DB, counter_list, size_dball, size_ball, degree_list_mainNode, betweenness_list_mainNode, coreness_list_mainNode, degree_list_removedNode, betweenness_list_removedNode, coreness_list_removedNode)





N=int(sys.argv[1]) 

k=int(sys.argv[2])

SEED=int(sys.argv[3])

radius = int(sys.argv[4])

G = make_ER_Graph(N,k,SEED)



(GC_List,counter_list,size_dball,size_ball,list_for_nodes_dball,list_for_nodes_ball,main_node_track) = dBalls_attack(G,radius)

(GC_List_NA,counter_list_NA,size_dball_NA,size_ball_NA,list_for_nodes_dball_NA,list_for_nodes_ball_NA,main_node_track_NA) = dBalls_attack_NA(G,radius)






init_name_GC_NA = "Fig_NA_attackDEG_ER_GC"
init_name_CL_NA = "Fig_NA_attackDB_ER_CL"
init_name_dball_NA = "Fig_NA_attackDB_ER_DBALL"
init_name_ball_NA = "Fig_NA_attackDB_ER_BALL"
init_name_dball_NODES_NA = "Fig_NA_attackDB_ER_DBALLNODES"
init_name_ball_NODES_NA = "Fig_NA_attackDB_ER_BALLNODES"
init_name_mainNodeTrack_NA = "Fig_NA_attackDB_ER_MAINNODETRACK"


init_name_GC = "Fig_attackDEG_ER_GC"
init_name_CL = "Fig_attackDB_ER_CL"
init_name_dball = "Fig_attackDB_ER_DBALL"
init_name_ball = "Fig_attackDB_ER_BALL"
init_name_dball_NODES = "Fig_attackDB_ER_DBALLNODES"
init_name_ball_NODES = "Fig_attackDB_ER_BALLNODES"
init_name_mainNodeTrack = "Fig_attackDB_ER_MAINNODETRACK"



GC_NA_name = get_name_ER(init_name_GC_NA, N, k, SEED,radius)
CL_NA_name = get_name_ER(init_name_CL_NA, N, k, SEED,radius)
dball_NA_name = get_name_ER(init_name_dball_NA, N, k, SEED,radius)
ball_NA_name = get_name_ER(init_name_ball_NA, N, k, SEED,radius)
dball_NODES_NA_name = get_name_ER(init_name_dball_NODES_NA, N, k, SEED,radius)
ball_NODES_NA_name = get_name_ER(init_name_ball_NODES_NA, N, k, SEED,radius)
mainNodeTrack_NA_name = get_name_ER(init_name_mainNodeTrack_NA, N, k, SEED,radius)


GC_name = get_name_ER(init_name_GC, N, k, SEED,radius)
CL_name = get_name_ER(init_name_CL, N, k, SEED,radius)
dball_name = get_name_ER(init_name_dball, N, k, SEED,radius)
ball_name = get_name_ER(init_name_ball, N, k, SEED,radius)
dball_NODES_name = get_name_ER(init_name_dball_NODES, N, k, SEED,radius)
ball_NODES_name = get_name_ER(init_name_ball_NODES, N, k, SEED,radius)
mainNodeTrack_name = get_name_ER(init_name_mainNodeTrack, N, k, SEED,radius)







with open(GC_name,'wb') as handle:
	pickle.dump(GC_List, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(CL_name,'wb') as handle:
	pickle.dump(counter_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(dball_name,'wb') as handle:
	pickle.dump(size_dball, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(ball_name,'wb') as handle:
	pickle.dump(size_ball, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(dball_NODES_name,'wb') as handle:
	pickle.dump(list_for_nodes_dball, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(ball_NODES_name,'wb') as handle:
	pickle.dump(list_for_nodes_ball, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(mainNodeTrack_name,'wb') as handle:
	pickle.dump(main_node_track, handle, protocol=pickle.HIGHEST_PROTOCOL)





with open(GC_NA_name,'wb') as handle:
	pickle.dump(GC_List_NA, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(CL_NA_name,'wb') as handle:
	pickle.dump(counter_list_NA, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(dball_NA_name,'wb') as handle:
	pickle.dump(size_dball_NA, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(ball_NA_name,'wb') as handle:
	pickle.dump(size_ball_NA, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(dball_NODES_NA_name,'wb') as handle:
	pickle.dump(list_for_nodes_dball_NA, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(ball_NODES_NA_name,'wb') as handle:
	pickle.dump(list_for_nodes_ball_NA, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(mainNodeTrack_NA_name,'wb') as handle:
	pickle.dump(main_node_track_NA, handle, protocol=pickle.HIGHEST_PROTOCOL)







