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



def perc_process_dBalls_track_balls(G_copy,radius):

	G = copy_graph(G_copy)

	GC_List = []
	size_dball = [] 
	size_ball = []

	degree_list = []

	counter = 0

	counter_list = []

	GC_List.append(get_GC(G))
	counter_list.append(counter)

	num_nodes_to_remove = G.numberOfNodes()

	while counter < num_nodes_to_remove:

		print(counter)

		(dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i) = get_all_dBN(G,radius)

		list_to_remove = dict_to_sorted_list(dict_nodes_x_i)

		if len(list_to_remove) == 0:
			break
		
		print(list_to_remove)

		node = get_largest_dball(dict_nodes_dBall,list_to_remove)

		print(node,dict_nodes_dBall[node])

		degree_list.append((node, G.degree(node)))

		#print(counter)

		(dBall,ball) = get_dBN(G,node,radius) 

		size_dball.append(len(dBall))
		size_ball.append(len(ball))


		#print(dBall)
		#print(ball)

		for i in dBall:
			G.removeNode(i)
			counter += 1

		GC_List.append(get_GC(G))

		counter_list.append(counter)


	return (GC_List,size_dball,size_ball,degree_list,counter_list)




def perc_process_dBalls(G_copy,radius,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []
	size_dball = [] 
	size_ball = []

	degree_list = []

	counter = 0

	GC_List.append(get_GC(G))

	while counter < num_nodes_to_remove:

		print(counter)

		(dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i) = get_all_dBN(G,radius)

		list_to_remove = dict_to_sorted_list(dict_nodes_x_i)

		if len(list_to_remove) == 0:
			i = random.sample(list(G.nodes()),1)
			G.removeNode(i[0])
			size_dball.append(0)
			size_ball.append(0)
			counter += 1
			GC_List.append(get_GC(G))
			continue
		
		print(list_to_remove)

		node = get_largest_dball(dict_nodes_dBall,list_to_remove)

		print(node,dict_nodes_dBall[node])

		degree_list.append((node, G.degree(node)))

		#print(counter)

		(dBall,ball) = get_dBN(G,node,radius) 

		size_dball.append(len(dBall))
		size_ball.append(len(ball))


		#print(dBall)
		#print(ball)

		for i in dBall:
			G.removeNode(i)
			counter += 1
			GC_List.append(get_GC(G))


	return (GC_List,size_dball,size_ball,degree_list)


def perc_process_dBalls_nodes_removed(G_copy,radius,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []
	size_dball = [] 
	size_ball = []

	degree_list = []

	counter = 0

	GC_List.append(get_GC(G))

	nodes_removed = []

	while counter < num_nodes_to_remove:

		print(counter)

		(dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i) = get_all_dBN(G,radius)

		list_to_remove = dict_to_sorted_list(dict_nodes_x_i)

		if len(list_to_remove) == 0:
			i = random.sample(list(G.nodes()),1)
			G.removeNode(i[0])
			size_dball.append(0)
			size_ball.append(0)
			counter += 1
			GC_List.append(get_GC(G))
			continue
		
		print(list_to_remove)

		node = get_largest_dball(dict_nodes_dBall,list_to_remove)

		print(node,dict_nodes_dBall[node])

		degree_list.append((node, G.degree(node)))

		#print(counter)

		(dBall,ball) = get_dBN(G,node,radius) 

		size_dball.append(len(dBall))
		size_ball.append(len(ball))


		#print(dBall)
		#print(ball)

		nodes_removed += dBall

		for i in dBall:
			G.removeNode(i)
			counter += 1
			GC_List.append(get_GC(G))


	return (GC_List,size_dball,size_ball,degree_list,nodes_removed)


def dict_to_sorted_list_dball(d):

	new_list = list(d.items())

	final_list = sorted(new_list, key = itemgetter(1), reverse = True)

	return final_list[0][0]


def perc_process_dBalls_bigBalls(G_copy,radius,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []
	size_dball = [] 
	size_ball = []

	degree_list = []

	counter = 0

	GC_List.append(get_GC(G))

	while counter < num_nodes_to_remove:

		print(counter)

		(dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i) = get_all_dBN(G,radius)

		node_to_remove = dict_to_sorted_list_dball(dict_nodes_ball)

		(dBall,ball) = get_dBN(G,node_to_remove,radius)

		print(len(dBall), len(ball))

		if len(dBall) == 0:
			i = random.sample(list(G.nodes()),1)
			G.removeNode(i[0])
			size_dball.append(0)
			size_ball.append(0)
			counter += 1
			GC_List.append(get_GC(G))
			degree_list.append((i[0], G.degree(i[0])))
			continue

		size_dball.append(len(dBall))
		size_ball.append(len(ball))

		degree_list.append((node_to_remove, G.degree(node_to_remove)))

		for i in dBall:
			G.removeNode(i)
			counter += 1
			GC_List.append(get_GC(G))


	return (GC_List,size_dball,size_ball,degree_list)


def perc_process_dBalls_bigDBalls(G_copy,radius,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []
	size_dball = [] 
	size_ball = []

	degree_list = []

	counter = 0

	GC_List.append(get_GC(G))

	while counter < num_nodes_to_remove:

		print(counter)

		(dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i) = get_all_dBN(G,radius)

		node_to_remove = dict_to_sorted_list_dball(dict_nodes_dBall)

		degree_list.append((node_to_remove, G.degree(node_to_remove)))

		(dBall,ball) = get_dBN(G,node_to_remove,radius)

		print(len(dBall), len(ball))

		if len(dBall) == 0:
			i = random.sample(list(G.nodes()),1)
			G.removeNode(i[0])
			size_dball.append(0)
			size_ball.append(0)
			counter += 1
			GC_List.append(get_GC(G))
			degree_list.append((i[0], G.degree(i[0])))
			continue

		size_dball.append(len(dBall))
		size_ball.append(len(ball))

		degree_list.append((node_to_remove, G.degree(node_to_remove)))

		for i in dBall:
			G.removeNode(i)
			counter += 1
			GC_List.append(get_GC(G))


	return (GC_List,size_dball,size_ball,degree_list)




def perc_random(G_copy,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []

	GC_List.append(get_GC(G))

	all_nodes = random.sample(list(G.nodes()),num_nodes_to_remove)

	for i in all_nodes:
		G.removeNode(i)
		GC_List.append(get_GC(G))

	return GC_List




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


def big_random_attack(G_copy,num_nodes_to_remove,num_sims):

	big_GC_List = []

	for i in range(num_sims):

		GC_list = perc_random(G_copy,num_nodes_to_remove)

		big_GC_List.append(GC_list)

	avg_list = get_avg_list(big_GC_List)

	return avg_list



def ADA_attack(G_copy,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []

	GC_List.append(get_GC(G))

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

	GC_List.append(get_GC(G))

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



def large_sims(N,k,SEED,type_of_attack,radius,num_nodes_to_remove,num_sims):

	GC_big_list = []

	size_ball_list = []
	size_dball_list = []

	degree_big_list = []

	if type_of_attack == "ABA":
		attack = ABA_attack

	elif type_of_attack == "ADA":
		attack = ADA_attack

	elif type_of_attack == "RAN":
		attack = perc_random

	elif type_of_attack == "DBA":
		attack = perc_process_dBalls

	for i in range(num_sims):

		G_nx = nx.erdos_renyi_graph(N, k/(N-1), seed = (SEED * i)) 
		G = nk.nxadapter.nx2nk(G_nx)

		if type_of_attack == "DBA":
			(GC_List,size_dball,size_ball,degree_list) = attack(G,radius,num_nodes_to_remove)

			size_dball_list.append(size_dball)
			size_ball_list.append(size_ball)
			degree_big_list.append(degree_list)


		else:
			GC_List = attack(G,num_nodes_to_remove)

		GC_List = GC_List[:(num_nodes_to_remove + 1)]

		GC_big_list.append(GC_List)

	avg_GC_list = get_avg_list(GC_big_list)

	if type_of_attack == "DBA":

		return (avg_GC_list,size_dball_list,size_ball_list,degree_big_list)

	else:
		return avg_GC_list




def get_graphs(G,radius_list,num_nodes_to_remove,filename_plt, filename_pickle_dball,filename_pickle_ball):

	size_dball_list = []
	size_ball_list = []
	dBalls_GC_list = []

	for radius in radius_list:

		(dBalls_GC,size_dball,size_ball) = perc_process_dBalls(G,radius,num_nodes_to_remove)

		dBalls_GC_list.append(dBalls_GC[:(num_nodes_to_remove + 1)])
		size_dball_list.append(size_dball)
		size_ball_list.append(size_ball)

	ADA_GC = ADA_attack(G,num_nodes_to_remove)
	RAN_GC = perc_random(G,num_nodes_to_remove)

	x_axis = [i for i in range(num_nodes_to_remove + 1)]

	counter = 0

	for dB in dBalls_GC_list:

		plt.plot(x_axis,dB, label = "dball_" + str(radius_list[counter]))

		counter += 1

	plt.plot(x_axis,ADA_GC, label = "ADA")
	plt.plot(x_axis,RAN_GC, label = "Random")

	(GC_List_opt,size_dball_opt,size_ball_opt,radius_track_opt) = big_attack(G, radius_list,num_nodes_to_remove)

	plt.plot(x_axis,GC_List_opt, label = "dballs_opt")

	plt.legend(loc='best')

	plt.savefig(filename_plt)

	plt.clf()

	with open(filename_pickle_dball,'wb') as handle:
		pickle.dump(size_dball_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(filename_pickle_ball,'wb') as handle:
		pickle.dump(size_ball_list, handle, protocol=pickle.HIGHEST_PROTOCOL)






def big_attack(G_copy,radius_list, num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []
	size_dball = [] 
	size_ball = []
	radius_track = []

	counter = 0

	GC_List.append(get_GC(G))

	while counter < num_nodes_to_remove:

		x_i_value = 2

		curr_radius = 0

		for radius in radius_list:

			(dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i) = get_all_dBN(G,radius)
			
			list_to_remove = dict_to_sorted_list(dict_nodes_x_i)
			
			if len(list_to_remove) == 0:
				continue

			node = list_to_remove[0][0]

			curr_x_i_value = list_to_remove[0][1]

			if curr_x_i_value < x_i_value:

				x_i_value = curr_x_i_value

				(dBall,ball) = get_dBN(G,node,radius) 

				curr_radius = radius

		if x_i_value == 2:
			break

		size_dball.append(len(dBall))

		size_ball.append(len(ball))

		radius_track.append(curr_radius)



		for i in dBall:
			G.removeNode(i)
			counter += 1
			GC_List.append(get_GC(G))


	return (GC_List,size_dball,size_ball,radius_track)



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

		node_sample = random.sample(all_nodes,1)

		node = node_sample[0]

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


def big_sim_ball(N,k,SEED,radius,perc_to_remove,num_sims):

	big_GC_List = []
	big_size_dball = []
	big_size_ball = []
	big_dg_list = []

	for i in range(num_sims):

		G_nx = nx.erdos_renyi_graph(N, k/(N-1), seed = SEED * (i+1)) 

		G_nk = nk.nxadapter.nx2nk(G_nx)

		num_nodes_to_remove = int(perc_to_remove * N)

		(GC_List,size_dball,size_ball,dg_list) = perc_process_dBalls_bigBalls(G_nk,radius,num_nodes_to_remove)

		GC_List_to_append = GC_List[:num_nodes_to_remove]

		big_GC_List.append(GC_List_to_append)

		big_size_dball.append(size_dball)

		big_size_ball.append(size_ball)

		big_dg_list.append(dg_list)

	return (big_GC_List,big_size_dball,big_size_ball,big_dg_list)



def big_sim_random_ball_removal(N,k,SEED,radius,perc_to_remove,num_sims):

	big_GC_List = []

	big_size_ball = []

	big_size_dball = []

	for i in range(num_sims):

		G_nx = nx.erdos_renyi_graph(N, k/(N-1), seed = SEED * (i+1)) 

		G_nk = nk.nxadapter.nx2nk(G_nx)

		num_nodes_to_remove = int(perc_to_remove * N)

		(GC_List,size_dball,size_ball) = random_ball_removal(G_nk,radius,num_nodes_to_remove)

		GC_List_to_append = GC_List[:num_nodes_to_remove]

		big_GC_List.append(GC_List_to_append)

		big_size_ball.append(size_ball)

		big_size_dball.append(size_dball)

	return (big_GC_List,big_size_dball,big_size_ball)



def make_SF_Graph(N,k,exp_out):

	num_edges = int((N * k) / 2)

	igG = ig.Graph.Static_Power_Law(N,num_edges,exp_out)

	allEdges = igG.get_edgelist()

	fixed_G = nx.Graph()

	listOfNodes = [i for i in range(N)]

	fixed_G.add_nodes_from(listOfNodes)

	fixed_G.add_edges_from(allEdges)

	G_nk = nk.nxadapter.nx2nk(fixed_G)

	return G_nk


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


def big_sims_WS(dim,size,nei,p,SEED,start_radius,end_radius):

	N = size ** dim

	G = make_WS_graph(dim,size,nei,p,SEED)

	(big_GC_List_dball,big_counter_list_dball) = big_sim_changing_radius(G,start_radius,end_radius)

	GC_list_ADA = ADA_attack(G, int(N * 0.99))

	GC_list_RAN = big_random_attack(G,int(N * 0.99),20)

	return (big_GC_List_dball,big_counter_list_dball,GC_list_ADA,GC_list_RAN)


def big_sims_ER(G,start_radius,end_radius):

	N = G.numberOfNodes()

	(big_GC_List_dball,big_counter_list_dball) = big_sim_changing_radius(G,start_radius,end_radius)

	GC_list_ADA = ADA_attack(G, int(N * 0.99))

	GC_list_RAN = big_random_attack(G,int(N * 0.99),20)

	return (big_GC_List_dball,big_counter_list_dball,GC_list_ADA,GC_list_RAN)


def big_sims_SF(G,start_radius,end_radius):

	N = G.numberOfNodes()

	(big_GC_List_dball,big_counter_list_dball) = big_sim_changing_radius(G,start_radius,end_radius)

	GC_list_ADA = ADA_attack(G, int(N * 0.99))

	GC_list_RAN = big_random_attack(G,int(N * 0.99),20)

	return (big_GC_List_dball,big_counter_list_dball,GC_list_ADA,GC_list_RAN)






def get_Area(GC_list):

	return sum(GC_list)

def get_nodes_not_in_list(l1,l2):

	return list(np.setdiff1d(l1,l2))


def get_optimized_fstar(G,fstar):

	all_nodes = get_GC_nodes(G)

	nodes_to_remove = random.sample(all_nodes,fstar)

	counter = 0

	GmodN = get_nodes_not_in_list(all_nodes, nodes_to_remove)

	GC_List = get_GC_list(G,nodes_to_remove)

	min_area = get_Area(GC_List)

	curr_GC_list = []

	min_removal_list = []

	GmodN_len = len(GmodN)

	nodes_to_remove_len = len(nodes_to_remove)

	while counter < 10000:

		print(counter)

		node1 = random.choice(nodes_to_remove)

		node2 = random.choice(GmodN)

		nodes_to_remove.remove(node1)

		nodes_to_remove.append(node2)

		GmodN.remove(node2)

		GmodN.append(node1)

		assert(len(nodes_to_remove) == nodes_to_remove_len)
		assert(len(GmodN) == GmodN_len)

		GC_List = get_GC_list(G,nodes_to_remove)
		
		curr_area = get_Area(GC_List)

		if curr_area < min_area:

			min_area = curr_area

			min_GC_list = GC_List.copy()

			min_removal_list = nodes_to_remove.copy()

			print(curr_area)

			counter = 0

		else:

			nodes_to_remove.remove(node2)

			nodes_to_remove.append(node1)

			GmodN.remove(node1)

			GmodN.append(node2)

			counter += 1

	return (min_area, min_GC_list, min_removal_list)





def get_optimized_fstar_dball(G,fstar,nodes_to_remove):

	all_nodes = get_GC_nodes(G)

	#nodes_to_remove = random.sample(all_nodes,fstar)

	counter = 0

	GmodN = get_nodes_not_in_list(all_nodes, nodes_to_remove)

	GC_List = get_GC_list(G,nodes_to_remove)

	min_area = get_Area(GC_List)

	curr_GC_list = get_GC_list(G,nodes_to_remove)

	min_GC_list = curr_GC_list.copy()

	min_removal_list = nodes_to_remove.copy()

	GmodN_len = len(GmodN)

	nodes_to_remove_len = len(nodes_to_remove)



	while counter < 10000:

		print(counter)

		node1 = random.choice(nodes_to_remove)

		node2 = random.choice(GmodN)

		nodes_to_remove.remove(node1)

		nodes_to_remove.append(node2)

		GmodN.remove(node2)

		GmodN.append(node1)

		assert(len(nodes_to_remove) == nodes_to_remove_len)
		assert(len(GmodN) == GmodN_len)

		GC_List = get_GC_list(G,nodes_to_remove)
		
		curr_area = get_Area(GC_List)

		if curr_area < min_area:

			min_area = curr_area

			min_GC_list = GC_List.copy()

			min_removal_list = nodes_to_remove.copy()

			print(curr_area)

			counter = 0

		else:

			nodes_to_remove.remove(node2)

			nodes_to_remove.append(node1)

			GmodN.remove(node1)

			GmodN.append(node2)

			counter += 1

	return (min_area, min_GC_list, min_removal_list)





#first list is the new one 
def get_diff(GC_list1, GC_list2):

	diff = 0

	counter = 0

	#print(GC_list1)
	#print(GC_list2)

	while counter < len(GC_list1):

		diff += GC_list2[counter] - GC_list1[counter] 

		#print(diff)

		counter += 1

	return diff
		


def swap_element(l,c1,c2):

	t = l[c1]
	l[c1] = l[c2]
	l[c2] = t


def get_GC_list(G_copy,removal_list):

	G = copy_graph(G_copy)

	GC_list = [get_GC(G)]

	for i in removal_list:

		G.removeNode(i)

		GC_list.append(get_GC(G))

	return GC_list




def swap_fun(G,removal_list, GC_list):

	counter = 0

	accumulation = []

	while counter < 1000:

		print(counter)

		#print(len(removal_list))

		#print(len(GC_list))

		l = [i for i in range(len(removal_list))]

		el_list = random.sample(l,2)

		el1 = el_list[0]
		el2 = el_list[1]

		swap_element(removal_list,el1,el2)

		#print(GC_list)

		new_GC_list = get_GC_list(G,removal_list)

		#print(new_GC_list)

		diff = get_diff(new_GC_list, GC_list)

		if diff > 0:

			print(diff)

			counter = 0

			GC_list = new_GC_list.copy()

			accumulation.append(diff)

			print(accumulation)

		else:

			swap_element(removal_list,el1,el2)

			counter += 1

	return accumulation



def get_fStar(G,radius,num_nodes_to_remove):

	ABA_list = ABA_attack(G, num_nodes_to_remove)

	(dball_list,size_dball,size_ball,degree_list,dball_nodes_removed) = perc_process_dBalls_nodes_removed(G,radius,num_nodes_to_remove)

	counter = 0

	big_counter = 0

	while big_counter < 10:

		if dball_list[counter] <= ABA_list[counter]:

			counter += 1

			big_counter = 0

		else:

			counter += 1

			big_counter += 1

	print("dball_fstar")

	print(dball_list[counter - big_counter])

	print(counter - big_counter)

	return (counter - big_counter, ABA_list, dball_list, dball_nodes_removed)


def get_fStar_balls(G,radius,num_nodes_to_remove):

	ABA_list = ABA_attack(G, num_nodes_to_remove)

	(dball_list,size_dball,size_ball,degree_list,dball_nodes_removed) = perc_process_dBalls_nodes_removed(G,radius,num_nodes_to_remove)

	counter = 0

	big_counter = 0

	while big_counter < 10:

		if dball_list[counter] <= ABA_list[counter]:

			counter += 1

			big_counter = 0

		else:

			counter += 1

			big_counter += 1

	print("dball_fstar")

	print(dball_list[counter - big_counter])

	print(counter - big_counter)

	return (counter - big_counter, ABA_list, dball_list, dball_nodes_removed,size_dball,size_ball)



def full_function(G,radius,perc_to_remove):

	N = G.numberOfNodes()

	(fs,ABA_list,dball_list,dball_nodes_removed) = get_fStar(G,radius,int(perc_to_remove * N))

	(min_area, min_GC_list, min_removal_list) = get_optimized_fstar(G,fs)

	original_removal_list = min_removal_list.copy()

	accumulation = swap_fun(G,min_removal_list, min_GC_list)

	original_GC_list = get_GC_list(G,original_removal_list)

	optimal_GC_list = get_GC_list(G,min_removal_list)

	return (fs,ABA_list,dball_list,original_GC_list,optimal_GC_list)


def timestamp_track(G,radius,perc_to_remove):

	N = G.numberOfNodes() 

	(fs,ABA_list,dball_list,dball_nodes_removed) = get_fStar(G,radius,int(perc_to_remove * N))

	get_optimized_fstar_dball(G,fs,nodes_to_remove)


def get_index_ball(ball_list, fstar):

	counter = 0

	ball_counter = 0

	while counter <= fstar:

		counter += ball_list[ball_counter]

		ball_counter += 1

	return ball_counter



def get_all_combinations(G, fs):

	all_nodes = get_GC_nodes(G)

	all_combos = []

	for s in itertools.combinations(all_nodes, fs):

		all_combos.append(list(s))

	return all_combos


def get_dball_fs(G,radius,num_nodes_to_remove):

	(fstar, ABA_list, dball_list, dball_nodes_removed,size_dball,size_ball) = get_fStar_balls(G,radius,num_nodes_to_remove)

	print(fstar)

	print(len(size_ball))

	print(len(size_dball))

	index_list = get_index_ball(size_dball, fstar)

	final_size_dball = size_dball[:]

	final_size_ball = size_ball[:]

	return (final_size_dball, final_size_ball)








N = 1000
k = 10
radius = 2
perc_to_remove = 0.5
SEED = 12456


G_nx = nx.erdos_renyi_graph(N, k/(N-1), seed = SEED)

G = nk.nxadapter.nx2nk(G_nx)


(final_size_dball, final_size_ball) = get_dball_fs(G,radius,int(N * perc_to_remove))

print(final_size_dball)
print(final_size_ball)

print(len(final_size_dball))
print(len(final_size_ball))

filename_dball = "fs_dball_" + str(N) + "_" + str(k) + "_" + str(radius) + "_" + str(SEED) + ".pickle"
filename_ball = "fs_ball_" + str(N) + "_" + str(k) + "_" + str(radius) + "_" + str(SEED) + ".pickle"

with open(filename_dball,'wb') as handle:
	pickle.dump(final_size_dball, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(filename_ball,'wb') as handle:
	pickle.dump(final_size_ball, handle, protocol=pickle.HIGHEST_PROTOCOL)



"""

#all_combos = get_all_combinations(G,200)

(fs, ABA_list, dball_list, dball_nodes_removed) = get_fStar(G,radius,int(N * perc_to_remove))

list_to_use =  dball_nodes_removed[:fs]

GC_list_to_use = dball_list[:(fs + 1)]

print(len(list_to_use))
print(len(GC_list_to_use))

accumulation = swap_fun(G,list_to_use, GC_list_to_use)

print(accumulation)

#print(sum(dball_list[:fs]))

#(min_area, min_GC_list, min_removal_list) = get_optimized_fstar(G,fs)

#print(min_area)
#print(min_GC_list)
#print(min_removal_list)


#filename_fs = "minArea.txt"

#file1 = open(filename_fs, "w") 
#file1.write(str(min_area))
#file1.close()

"""

"""
num_nodes_to_remove = int(perc_to_remove * N)



(GC_List,size_dball,size_ball,degree_list,nodes_removed) = perc_process_dBalls_nodes_removed(G_nk,radius,num_nodes_to_remove)

(fs,ABA_list,dball_list) = get_fStar(G_nk,radius,int(perc_to_remove * N))

print(fs)

nodes_to_remove = nodes_removed[:fs]

(min_area, min_GC_list, min_removal_list) = get_optimized_fstar_dball(G_nk,fs,nodes_to_remove)


print(min_area)

"""




"""
(fs,ABA_list,dball_list) = get_fStar(G_nk,radius,int(perc_to_remove * N))

print("fs")
print(fs)

(min_area, min_GC_list, min_removal_list) = get_optimized_fstar(G_nk,fs)


print(min_area)
print(min_GC_list)
print(min_removal_list)

original_removal_list = min_removal_list.copy()

accumulation = swap_fun(G_nk,min_removal_list, min_GC_list)

print(accumulation)

print(get_GC_list(G_nk,original_removal_list))
print(get_GC_list(G_nk,min_removal_list))
print(get_Area(dball_list[:fs]))
"""









"""
(fs,ABA_list,dball_list,original_GC_list,optimal_GC_list) = full_function(G_nk,radius,perc_to_remove)

filename_ABA = "minArea_ABA.pickle"
filename_dball = "minArea_dball.pickle"
filename_original = "minArea_originalGC.pickle"
filename_optimal = "minArea_optimalGC.pickle"
filename_fs = "minArea_fs.txt"

with open(filename_ABA,'wb') as handle:
	pickle.dump(ABA_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(filename_dball,'wb') as handle:
	pickle.dump(dball_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(filename_original,'wb') as handle:
	pickle.dump(original_GC_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(filename_optimal,'wb') as handle:
	pickle.dump(optimal_GC_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

file1 = open(filename_fs, "w") 
file1.write(str(fs))
file1.close()

"""

