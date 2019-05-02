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



def perc_process_dBalls_removalOrder(G_copy,radius,num_nodes_to_remove):

	G = copy_graph(G_copy)

	GC_List = []
	size_dball = [] 
	size_ball = []

	degree_list = []

	counter = 0

	removal_order = []

	GC_List.append(get_GC(G))

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

		removal_order += dBall

		#print(dBall)
		#print(ball)

		for i in dBall:
			G.removeNode(i)
			counter += 1
			GC_List.append(get_GC(G))


	return (GC_List,size_dball,size_ball,degree_list,removal_order)



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



def get_fStar(ABA_list, dball_list):

	counter = 0

	big_counter = 0

	while big_counter < 10:

		if dball_list[counter] <= ABA_list[counter]:

			counter += 1

			big_counter = 0

		else:

			counter += 1

			big_counter += 1

	return counter - big_counter






def do_perc(G,radius,num_nodes_to_remove):

	N = G.numberOfNodes()

	GC_List_ABA = ABA_attack(G, num_nodes_to_remove)

	print(GC_List_ABA)

	(GC_List_dball,size_dball,size_ball,degree_list,removal_order) = perc_process_dBalls_removalOrder(G,radius,num_nodes_to_remove)

	print(GC_List_dball)

	fstar = get_fStar(GC_List_ABA,GC_List_dball)

	print("fstar")
	print(fstar)

	list_to_check = GC_List_dball[:fstar]

	removal_order_to_check = removal_order[:(fstar - 1)]

	original_list = list_to_check.copy()

	accumulation_list = swap_fun(G, removal_order_to_check, list_to_check)

	new_list = get_GC_list(G,removal_order_to_check)

	print(accumulation_list)

	return (original_list,new_list,accumulation_list)




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


def big_sim_SF(N,k,exp_out,SEED,radius,perc_to_remove,num_sims):

	big_GC_List = []

	big_size_ball = []

	big_size_dball = []

	big_dg_list = []

	for i in range(num_sims):

		G_nk = make_SF_Graph(N,k,exp_out,SEED)

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



def printAll(G,node,radius):

	curr_list = [node]

	checked = set([node])

	for i in range(radius):

		nbors = []

		for n in curr_list:

			print(n, G.neighbors(n))

			nbors += G.neighbors(n)

		curr_list = nbors.copy()

		curr_list = list(filter(lambda x : x not in checked, curr_list))

		print(curr_list)

		for n in nbors:

			checked.add(n)









#WS
"""
dim = int(sys.argv[1])

size = int(sys.argv[2])

nei = int(sys.argv[3])

p = float(sys.argv[4])

SEED = int(sys.argv[5])

start_radius = int(sys.argv[6])

end_radius = int(sys.argv[7])
"""


#ER
"""
N=int(sys.argv[1]) # number of nodes

k=int(sys.argv[2])

SEED=int(sys.argv[3])

start_radius = int(sys.argv[4])

end_radius = int(sys.argv[5])
"""

#SF
"""
N=int(sys.argv[1]) # number of nodes

k=int(sys.argv[2])

exp_out=float(sys.argv[3])

start_radius = int(sys.argv[4])

end_radius = int(sys.argv[5])

"""

N = 10000
k = 4
radius = 3
exp_out=3.5
SEED = 4255

#G_nx = nx.erdos_renyi_graph(N, k/(N-1), seed = SEED) 

#G_nk = make_SF_Graph(N,k,exp_out)

G_nk = make_SF_Graph(N,k,exp_out,SEED)

#(GC_List,size_dball,size_ball,degree_list,counter_list) = perc_process_dBalls_track_balls(G_nk,radius)

#print(degree_list[:300])
#print(size_dball[:300])



#Raidus 2
#2.5 k = 4
#[(5362, 3), (513, 2), (1914, 2), (1966, 2), (2107, 2), (2243, 2), (2835, 2), (3178, 2), (4168, 2), (4301, 2), (4766, 2), (4960, 2), (4970, 2), (5308, 2), (5419, 2), (5453, 2), (7436, 2), (2317, 3), (3628, 3), (4762, 3), (6892, 3), (9552, 3), (12, 1), (34, 1), (90, 1), (94, 1), (111, 1), (129, 1), (192, 1), (199, 1), (200, 1), (275, 1), (281, 1), (314, 1), (368, 1), (370, 1), (404, 1), (437, 1), (491, 1), (517, 1), (520, 1), (542, 1), (548, 1), (582, 1), (621, 1), (625, 1), (698, 1), (867, 1), (971, 1), (1059, 1)]

#3.5 k = 4
#[(881, 2), (3111, 2), (5085, 2), (8245, 2), (2994, 3), (3313, 3), (3552, 2), (5892, 3), (91, 1), (214, 1), (137, 1), (247, 1), (271, 1), (370, 1), (617, 1), (648, 1), (795, 1), (796, 1), (800, 1), (814, 1), (826, 1), (827, 1), (902, 1), (1142, 1), (1175, 1), (1193, 1), (1218, 1), (1300, 1), (1420, 1), (1499, 1), (1512, 1), (1517, 1), (1572, 1), (1574, 1), (1631, 1), (1654, 1), (1753, 1), (3880, 3), (1805, 1), (1963, 1), (1982, 1), (2074, 1), (2081, 1), (2175, 1), (2289, 1), (2417, 1), (2558, 1), (2573, 1), (2692, 1), (2330, 1), (2711, 1), (2712, 1), (2880, 1), (2921, 1), (3094, 1), (3185, 1), (3202, 1), (3335, 1), (3669, 1), (3748, 1), (3753, 1), (3789, 1), (3826, 1), (4065, 1), (4233, 1), (4251, 1), (4320, 1), (4324, 1), (4447, 1), (4521, 1), (920, 1), (4604, 1), (4658, 1), (4689, 1), (943, 1), (4727, 1), (4733, 1), (4938, 1), (5207, 1), (5332, 1), (5505, 1), (5516, 1), (5577, 1), (5579, 1), (5655, 1), (5661, 1), (5793, 1), (5821, 1), (5835, 1), (4608, 1), (5952, 1), (5998, 1), (6077, 1), (6303, 1), (4237, 1), (3807, 1), (6421, 1), (6655, 1), (6728, 1), (6859, 1), (6929, 1), (7170, 1), (2715, 1), (7390, 1), (7737, 1), (7971, 1), (8126, 1), (8149, 1), (8197, 1), (833, 2), (7483, 1), (7047, 4), (1253, 2), (1520, 2), (1557, 2), (1576, 2), (1061, 1), (6843, 1), (2015, 2), (2073, 2), (2284, 2), (2420, 2), (3384, 2), (3008, 1), (4634, 2), (5247, 2), (5780, 2), (6260, 2), (6554, 2), (6986, 2), (96, 3), (659, 3), (1235, 3), (3219, 3), (3897, 3), (4211, 3), (5725, 3), (5739, 3), (7853, 3), (1076, 1), (1860, 4), (1361, 1), (3419, 1), (1880, 2), (2259, 2), (3188, 2), (3677, 4), (456, 1), (1954, 2), (5146, 4), (7782, 4), (6815, 4), (3426, 2), (959, 1), (7238, 4), (4524, 2), (7952, 4), (3273, 1), (3323, 1), (8253, 4), (3445, 2), (5140, 3), (9434, 4), (114, 3), (383, 3), (1240, 3), (765, 4), (542, 2), (7495, 1), (1862, 2), (2448, 4), (7964, 1), (8153, 1), (926, 3), (2364, 1), (4772, 2), (2584, 2), (1399, 3), (5702, 3), (1418, 3), (1687, 3), (8003, 1), (2313, 3), (4712, 1), (2359, 3), (574, 1), (2779, 3), (3090, 3), (3363, 3), (3594, 3), (3907, 3), (2041, 1), (2599, 1), (4332, 1), (7338, 1), (4109, 3), (2432, 1), (8826, 1), (4317, 3), (2611, 3), (2860, 1), (1479, 1), (5558, 2), (3890, 4), (2117, 2), (4703, 3), (699, 1), (5161, 3), (5264, 3), (782, 1), (5632, 3), (5876, 3), (6330, 3), (9258, 4), (7551, 2), (9111, 3), (1010, 1), (4479, 1), (6789, 3), (7263, 3), (7370, 3), (7427, 3), (1815, 1), (7460, 3), (8846, 3), (6211, 1), (56, 2), (1735, 1), (546, 2), (1304, 3), (103, 2), (204, 2), (6225, 1), (5158, 1), (461, 2), (5336, 4), (4711, 3), (7593, 1), (501, 2), (538, 2), (6952, 1), (702, 2), (934, 2), (8526, 4), (2776, 2), (3198, 1), (2944, 1), (7630, 4), (1359, 1), (3692, 3), (8104, 2), (5227, 3), (334, 2), (1068, 2), (1988, 1), (1139, 2), (2933, 2), (2453, 1), (2565, 1), (146, 2), (4871, 1), (1872, 2), (1909, 2), (5251, 1), (1923, 2), (2150, 1), (4076, 1), (4738, 2), (567, 1), (8419, 3), (8626, 1), (4041, 1), (2054, 2), (225, 1), (2439, 2), (5594, 3), (2539, 2), (8525, 3), (5014, 2), (1554, 3), (2016, 1), (8300, 4), (367, 1), (2910, 2), (3029, 2), (2533, 2), (3266, 2), (4410, 1), (6619, 1), (7267, 1), (8458, 3), (9377, 3), (1409, 2), (3539, 1), (7102, 3), (5038, 1), (3954, 1), (4374, 3), (8341, 1), (8448, 1)]
#[1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 4, 1, 1, 2, 2, 2, 4, 1, 2, 4, 4, 5, 1, 1, 5, 2, 5, 1, 1, 5, 2, 3, 5, 4, 4, 4, 5, 1, 1, 2, 5, 1, 1, 4, 1, 2, 2, 4, 3, 4, 4, 1, 4, 1, 4, 1, 4, 4, 4, 4, 4, 1, 1, 1, 1, 4, 1, 1, 4, 4, 1, 1, 2, 5, 2, 4, 1, 4, 4, 1, 4, 4, 4, 5, 2, 3, 1, 1, 4, 4, 4, 4, 1, 4, 4, 1, 3, 1, 2, 4, 3, 3, 1, 1, 2, 5, 4, 1, 3, 3, 1, 3, 3, 5, 2, 1, 1, 5, 1, 4, 2, 4, 3, 3, 1, 3, 2, 1, 1, 3, 1, 3, 3, 1, 3, 1, 1, 2, 1, 4, 1, 1, 3, 1, 3, 4, 3, 3, 2, 4, 1, 5, 1, 3, 3, 3, 3, 1, 1, 1, 4, 4, 3, 1, 4, 1, 1, 2, 1, 1]


#Radius 3
#2.5 k = 4
#b = [(2581, 1), (4970, 2), (5419, 2), (137, 1), (498, 1), (1025, 1), (1651, 1), (1683, 1), (1911, 1), (2078, 1), (2227, 1), (2557, 1), (3429, 1), (3834, 1), (3969, 1), (4705, 1), (7256, 1), (4620, 3), (1091, 1), (2017, 1), (3609, 1), (3835, 1), (4337, 1), (1520, 2), (178, 2), (4576, 2), (94, 1), (621, 1), (698, 1), (1211, 1), (1221, 1), (1605, 1), (1705, 1), (2507, 1), (2654, 1), (3446, 1), (3501, 1), (3659, 1), (3737, 1), (3793, 1), (3907, 1), (1729, 1), (4041, 1), (4353, 1), (4403, 1), (5635, 1), (6624, 1), (7404, 1), (7544, 1), (7740, 1), (1308, 1), (1526, 2), (5964, 1), (23, 1), (242, 1), (2249, 1), (3594, 1), (5978, 1), (1401, 1), (1754, 1), (4352, 2), (5246, 2), (270, 1), (1510, 1), (2707, 1), (1765, 2), (3714, 1), (2430, 2), (3406, 2), (2293, 2), (5770, 1), (7312, 1), (438, 1), (732, 1), (986, 1), (7464, 1), (81, 2), (7865, 2), (3169, 1), (4723, 1), (2744, 1), (3165, 1), (239, 1), (8033, 2), (467, 1), (647, 1), (5491, 1), (910, 1), (1293, 1), (2587, 1), (5618, 1), (2603, 2), (3053, 1), (75, 1), (3272, 1), (3853, 1), (4096, 1), (4783, 1), (1219, 1), (7309, 1), (628, 1), (34, 1), (111, 1), (129, 1), (192, 1), (2684, 1), (314, 1), (444, 1), (582, 1), (1120, 1), (1484, 1), (1527, 1), (4579, 1), (997, 1), (1267, 1), (1822, 1), (1825, 1), (848, 1), (2071, 1), (8675, 1), (2362, 1), (2769, 1), (3569, 1), (2793, 2), (254, 1), (3476, 1), (5242, 2), (3563, 1), (4954, 2), (1317, 1), (147, 1), (1697, 1), (3666, 1), (1378, 1), (7075, 1), (3668, 1), (4067, 1), (3614, 1), (4434, 1), (4572, 1), (5026, 1), (5426, 1), (5599, 1), (5779, 1), (3364, 1), (5950, 1), (5953, 1), (6545, 1), (6681, 1), (8200, 1), (2883, 1), (4709, 3), (933, 1), (8532, 1), (121, 1), (2742, 1), (3678, 1), (8051, 1), (6330, 1), (4837, 1), (8345, 1), (1010, 3), (7475, 1), (6848, 2), (7299, 2), (8105, 2), (2130, 1), (5159, 1), (7419, 1), (4609, 1), (2794, 1), (1684, 1), (4931, 1), (1695, 1), (639, 1), (5669, 3), (1936, 1), (775, 1), (3013, 1), (1623, 1), (1459, 1), (346, 1), (2709, 2), (4593, 2), (3865, 1), (2814, 1), (6412, 1), (7873, 1), (102, 2), (6229, 2), (5908, 1), (240, 1), (1169, 1), (5177, 1), (7994, 1), (701, 1), (3283, 1), (1511, 1), (8797, 3), (2286, 1), (6309, 1), (3788, 1), (1018, 1), (7820, 1), (3821, 2), (2967, 1), (7742, 2), (1038, 1), (1118, 1), (3307, 2), (618, 1), (490, 1), (945, 1), (7413, 1), (165, 1), (1946, 1), (112, 3), (1748, 1), (5400, 1), (4787, 2), (4377, 1), (5145, 1), (3241, 1), (1514, 1), (1949, 1), (2513, 1), (3379, 1), (7268, 2), (738, 2), (1330, 1), (442, 2), (2933, 1), (1960, 1), (275, 1), (5484, 1), (1898, 1), (435, 1), (1287, 1), (3632, 2), (2917, 1), (6297, 2), (4940, 1), (623, 1), (2098, 1), (558, 1), (1660, 1), (5433, 3), (2416, 1), (4984, 2), (4282, 1), (2753, 1), (1559, 1), (2326, 1), (2585, 1), (3177, 1), (8088, 1), (3529, 1), (5477, 1), (1379, 1), (396, 1), (8061, 1), (1346, 1), (6072, 1), (4878, 1), (1553, 1), (4268, 1), (2715, 1), (4346, 1), (5316, 2), (8617, 1), (4500, 2), (8187, 1), (4012, 3), (906, 1), (6215, 1), (4555, 2), (4584, 1), (3077, 1), (305, 3), (1580, 1), (4165, 2), (487, 1), (6459, 3), (858, 1), (7557, 1), (1155, 2), (4989, 1), (4877, 1), (867, 1), (882, 1), (2155, 2), (3075, 1), (2473, 1), (3849, 1), (805, 2), (2763, 1), (5518, 1), (4054, 1), (3880, 4), (6878, 1)]
#a = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 2, 2, 2, 2, 3, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 2, 3, 3, 3, 3, 2, 4, 4, 4, 6, 6, 6, 3, 6, 2, 4, 3, 6, 6, 3, 5, 3, 5, 5, 5, 3, 2, 2, 2, 5, 4, 5, 4, 4, 1, 4, 4, 6, 1, 4, 4, 4, 4, 4, 4, 4, 5, 2, 6, 3, 3, 3, 3, 1, 2, 3, 3, 3, 3, 3, 3, 4, 3, 3, 3, 1, 3, 3, 3, 3, 3, 5, 1, 2, 7, 3, 3, 5, 4, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 9, 4, 1, 2, 3, 3, 2, 3, 3, 3, 9, 2, 8, 8, 3, 5, 4, 1, 2, 4, 3, 3, 1, 3, 9, 2, 3, 5, 3, 2, 8, 7, 7, 3, 4, 7, 7, 6, 7, 1, 6, 2, 3, 1, 5, 1, 6, 3, 3, 3, 2, 5, 3, 6, 4, 8, 1, 1, 7, 6, 4, 1, 3, 3, 3, 8, 2, 1, 4, 4, 2, 3, 3, 3, 3, 3, 9, 2, 4, 2, 7, 4, 3, 3, 7, 4, 1, 3, 3, 11, 5, 4, 2, 4, 1, 7, 3, 1, 4, 4, 3, 3, 3, 3, 2, 3, 3, 5, 3, 3, 4, 3, 1, 3, 6, 4, 6, 5, 3, 6, 3, 10, 1, 3, 5, 4, 1, 7, 8, 8, 4, 7, 2, 2, 6, 4, 4, 3, 3, 3, 1, 2, 5, 3, 2, 3, 2, 10, 3]

#3.5 k = 4
b = [(1832, 1), (2734, 1), (3710, 1), (7362, 1), (2793, 1), (3298, 1), (6752, 1), (4226, 1), (6377, 2), (648, 1), (796, 1), (1175, 1), (1872, 2), (1512, 1), (1654, 1), (2558, 1), (4727, 1), (5655, 1), (5998, 1), (6421, 1), (6728, 1), (8197, 1), (1224, 2), (3295, 4), (408, 1), (1953, 1), (2920, 1), (3844, 1), (6336, 1), (7527, 1), (8438, 1), (558, 1), (9086, 1), (827, 1), (1361, 1), (2261, 2), (4215, 2), (4910, 2), (8139, 1), (2605, 2), (4453, 2), (310, 1), (4408, 1), (943, 1), (1008, 1), (2829, 1), (2891, 1), (4921, 1), (5367, 1), (41, 1), (6986, 2), (3080, 1), (2432, 1), (3912, 1), (3957, 1), (4555, 1), (5781, 1), (7812, 1), (370, 1), (1076, 1), (1142, 1), (1193, 1), (2406, 1), (1631, 1), (4234, 1), (2643, 1), (2531, 1), (1963, 1), (2354, 1), (2074, 1), (2712, 1), (2921, 1), (3094, 1), (3335, 1), (2497, 1), (4236, 1), (2772, 1), (5303, 1), (4065, 1), (4251, 1), (7391, 1), (4324, 1), (8956, 1), (3300, 1), (4479, 1), (967, 1), (5579, 1), (6344, 3), (1681, 1), (8353, 3), (6823, 2), (3555, 1), (1174, 1), (3776, 1), (3769, 1), (2630, 1), (1229, 1), (5518, 2), (3659, 2), (2130, 2), (1971, 1), (8660, 2), (135, 1), (3943, 1), (776, 2), (873, 1), (6970, 1), (7383, 2), (8623, 2), (3151, 2), (6938, 3), (2505, 1), (1735, 1), (7410, 1), (546, 2), (56, 2), (5984, 1), (3636, 1), (8099, 2), (1500, 1), (2694, 1), (85, 1), (2459, 1), (4429, 1), (817, 1), (514, 1), (3687, 1), (3815, 1), (6687, 2), (7430, 1), (6735, 1), (7971, 1), (1838, 2), (1038, 3), (6175, 1), (421, 1), (1436, 1), (1987, 1), (2227, 2), (2105, 1), (2037, 1), (4036, 3), (2649, 1), (1648, 1), (6878, 1), (6186, 1), (5151, 1), (1702, 1), (4969, 1), (4308, 1), (4277, 1), (1061, 1), (3695, 1), (7737, 1), (4675, 2), (4026, 1), (204, 2), (820, 1), (2073, 2), (2284, 2), (1556, 1), (4811, 1), (3237, 1), (2051, 1), (8323, 1), (333, 1), (9195, 1), (3519, 1), (860, 2), (1385, 1), (4733, 1), (471, 1), (482, 1), (4992, 1), (2247, 2), (6730, 1), (1930, 1), (1414, 1), (3585, 1), (8043, 1), (2353, 1), (9224, 1), (6083, 3), (823, 1), (3292, 1), (3257, 1), (6337, 2), (6587, 1), (8030, 2), (1591, 1), (7007, 1), (6351, 1), (6710, 2), (1699, 1), (1843, 1), (6801, 2), (1645, 2), (8163, 2), (852, 1), (430, 1), (3886, 1), (5249, 1), (7543, 1), (6613, 1), (2179, 1), (1014, 3), (1051, 1), (5847, 2), (2632, 4), (857, 1), (4634, 2), (8375, 1), (837, 2), (645, 1), (8543, 1), (3789, 1), (4466, 1), (2846, 2), (285, 1), (6131, 2), (6236, 2), (5907, 1), (8126, 1), (911, 3), (3679, 1), (6338, 2), (1642, 1), (3931, 1), (9210, 1), (2730, 1), (497, 1), (8561, 1), (2790, 1), (4390, 1), (1848, 1), (5038, 1), (1139, 2), (2407, 1), (1572, 1), (1120, 1), (2453, 1), (3580, 1), (2278, 2), (3562, 1), (3041, 1), (3410, 1), (5521, 1), (326, 1), (2932, 1), (7517, 1), (3377, 1), (774, 2), (5830, 1), (2155, 1), (5975, 1), (4483, 1), (2598, 1), (6929, 1), (4539, 1), (7374, 3), (6556, 1), (8345, 2), (5954, 2), (4345, 2), (4220, 1), (367, 1), (5794, 1), (7179, 2), (3670, 1), (7489, 2), (6017, 2), (6848, 1), (1479, 1), (247, 1), (2041, 1), (2991, 1), (422, 1), (6346, 1), (7488, 1), (5457, 1), (4871, 2), (7708, 2), (8246, 1), (6228, 3), (6633, 3), (2444, 1), (856, 1), (3351, 2), (3019, 1), (7714, 3), (7314, 2), (2058, 1), (7848, 2), (8771, 2), (259, 2), (1774, 1), (2485, 1), (7149, 1), (7731, 3), (356, 1)]
a = [1, 1, 1, 1, 2, 2, 1, 2, 3, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 9, 3, 3, 3, 3, 3, 3, 3, 4, 4, 2, 2, 5, 5, 6, 2, 7, 6, 5, 2, 2, 5, 5, 1, 5, 5, 3, 5, 4, 2, 4, 4, 4, 4, 4, 3, 3, 3, 3, 2, 3, 4, 3, 2, 3, 5, 3, 3, 3, 3, 3, 1, 5, 2, 4, 3, 3, 3, 3, 1, 4, 3, 2, 3, 11, 4, 9, 8, 5, 5, 4, 4, 7, 4, 7, 5, 6, 4, 10, 2, 3, 9, 2, 4, 7, 7, 1, 9, 1, 2, 2, 4, 5, 1, 5, 5, 4, 4, 3, 3, 5, 1, 5, 3, 3, 6, 3, 3, 3, 8, 8, 5, 4, 4, 4, 9, 2, 3, 10, 4, 1, 4, 2, 3, 7, 4, 1, 1, 2, 4, 3, 7, 1, 6, 6, 6, 6, 6, 1, 6, 3, 4, 3, 4, 1, 8, 3, 3, 8, 3, 4, 4, 2, 1, 3, 3, 5, 2, 3, 12, 2, 4, 4, 5, 3, 9, 1, 3, 3, 5, 3, 4, 5, 8, 8, 2, 7, 5, 4, 4, 1, 3, 10, 3, 8, 14, 3, 5, 1, 3, 4, 4, 3, 3, 7, 4, 7, 8, 3, 3, 10, 4, 4, 3, 3, 3, 2, 4, 3, 1, 2, 2, 2, 5, 1, 3, 3, 3, 2, 8, 5, 4, 4, 3, 2, 3, 3, 4, 8, 1, 4, 5, 4, 4, 3, 3, 11, 1, 9, 12, 4, 5, 2, 2, 7, 5, 6, 5, 2, 3, 3, 3, 3, 3, 3, 2, 5, 6, 7, 3, 12, 12, 1, 3, 7, 6, 9, 4, 1, 9, 3, 4, 2, 2, 3, 10, 2]


#print(a.index(6))

ind = 4

print(a[ind])
print(b[ind])

printAll(G_nk,b[ind][0],radius)


"""
(original_list,new_list,accumulation_list) =  do_perc(G_nk,radius,int(perc_to_remove * N))

filename_OL = "dballSwapOL_N_" + str(N) + "_k_" + str(k) + "_SEED_" + str(SEED) + "_radius_" + str(radius) + ".pickle"

filename_NL = "dballSwapNL_N_" + str(N) + "_k_" + str(k) + "_SEED_" + str(SEED) + "_radius_" + str(radius) + ".pickle"

filename_AL = "dballSwapAL_N_" + str(N) + "_k_" + str(k) + "_SEED_" + str(SEED) + "_radius_" + str(radius) + ".pickle"
"""

#(big_GC_List_dball,big_counter_list_dball,GC_list_ADA,GC_list_RAN) = big_sims_SF(G_nk,start_radius,end_radius)


#WS
"""
filename_GC = "dballTrackRadius" +  "_GC_WS_dim_" + str(dim) + "_size_" + str(size) + "_nei_" + str(nei) + "_p_" + str(p) + "_SEED_" + str(SEED) + "_startRadius_" + str(start_radius) + "_endRadius_" + str(end_radius) + ".pickle"

filename_CL = "dballTrackRadius" +  "_CL_WS_dim_" + str(dim) + "_size_" + str(size) + "_nei_" + str(nei) + "_p_" + str(p) + "_SEED_" + str(SEED) + "_startRadius_" + str(start_radius) + "_endRadius_" + str(end_radius) + ".pickle"

filename_ADA = "dballTrackRadiusADA" +  "_GC_WS_dim_" + str(dim) + "_size_" + str(size) + "_nei_" + str(nei) + "_p_" + str(p) + "_SEED_" + str(SEED) + "_startRadius_" + str(start_radius) + "_endRadius_" + str(end_radius) + ".pickle"

filename_RAN = "dballTrackRadiusRAN" +  "_GC_WS_dim_" + str(dim) + "_size_" + str(size) + "_nei_" + str(nei) + "_p_" + str(p) + "_SEED_" + str(SEED) + "_startRadius_" + str(start_radius) + "_endRadius_" + str(end_radius) + ".pickle"
"""

#SF
"""
filename_GC = "dballTrackRadius" +  "_GC_SF_N_" + str(N) + "_k_" + str(k) + "_expout_" + str(exp_out) + "_startRadius_" + str(start_radius) + "_endRadius_" + str(end_radius) + ".pickle"

filename_CL = "dballTrackRadius" +  "_CL_SF_N_" + str(N) + "_k_" + str(k) + "_expout_" + str(exp_out) + "_startRadius_" + str(start_radius) + "_endRadius_" + str(end_radius) + ".pickle"

filename_ADA = "dballTrackRadiusADA" +  "_GC_SF_N_" + str(N) + "_k_" + str(k) + "_expout_" + str(exp_out) + "_startRadius_" + str(start_radius) + "_endRadius_" + str(end_radius) + ".pickle"

filename_RAN = "dballTrackRadiusRAN" +  "_GC_SF_N_" + str(N) + "_k_" + str(k) + "_expout_" + str(exp_out) + "_startRadius_" + str(start_radius) + "_endRadius_" + str(end_radius) + ".pickle"

"""


"""
with open(filename_GC, 'wb') as handle:
	pickle.dump(big_GC_List_dball, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(filename_CL,'wb') as handle:
	pickle.dump(big_counter_list_dball, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(filename_ADA, 'wb') as handle:
	pickle.dump(GC_list_ADA, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(filename_RAN,'wb') as handle:
	pickle.dump(GC_list_RAN, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""



"""
num_nodes_to_remove = int(perc_to_remove * N)

(big_GC_List,big_size_dball,big_size_ball,big_dg_list) = big_sim_SF(N,k,exp_out,radius,perc_to_remove,num_sims)

filename_GC = "dballSF" +  "_GC_SF_N_" + str(N) + "_k_" + str(k) + "_expout_" + str(exp_out) + "_radius_" + str(radius) + "_perctoremove_" + str(perc_to_remove) + ".pickle"
filename_ball = "dballSF" +  "_ball_SF_N_" + str(N) + "_k_" + str(k) + "_expout_" + str(exp_out) + "_radius_" + str(radius) + "_perctoremove_" + str(perc_to_remove) + ".pickle"
filename_dball = "dballSF"  + "_dball_SF_N_" + str(N) + "_k_" + str(k) + "_expout_" + str(exp_out) + "_radius_" + str(radius) + "_perctoremove_" + str(perc_to_remove) + ".pickle"
filename_dg = "dballSF"  + "_dg_SF_N_" + str(N) + "_k_" + str(k) + "_expout_" + str(exp_out) + "_radius_" + str(radius) + "_perctoremove_" + str(perc_to_remove) + ".pickle"


with open(filename_GC, 'wb') as handle:
	pickle.dump(big_GC_List, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(filename_ball,'wb') as handle:
	pickle.dump(big_size_ball, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(filename_dball,'wb') as handle:
	pickle.dump(big_size_dball, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(filename_dg,'wb') as handle:
	pickle.dump(big_dg_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

#(GC_List1,size_dball1,size_ball1,dg_list1) = perc_process_dBalls_bigBalls(G_nk,radius,num_nodes_to_remove)

#(GC_List2,size_dball2,size_ball2,dg_list2) = perc_process_dBalls_bigDBalls(G_nk,radius,num_nodes_to_remove)

#(GC_List3,size_dball3,size_ball3,dg_list3) = perc_process_dBalls(G_nk,radius,num_nodes_to_remove)


#print(GC_List1)

#print(GC_List2)

#print(GC_List3)

#print(list(zip(zip(GC_List1[:1000],GC_List2[:1000]),GC_List3)))

"""



"""
N=int(sys.argv[1]) # number of nodes

k=int(sys.argv[2])

SEED=int(sys.argv[3])

radius = int(sys.argv[4])

perc_to_remove = float(sys.argv[5])

num_sims = int(sys.argv[6])

(big_GC_List,big_size_dball,big_size_ball,big_dg_list) = big_sim(N,k,SEED,radius,perc_to_remove,num_sims)


filename_GC = "dballAttBigBall_" +  "_GC_ER_N_" + str(N) + "_k_" + str(k) + "_SEED_" + str(SEED) + "_radius_" + str(radius) + "_perctoremove_" + str(perc_to_remove) + ".pickle"
filename_ball = "dballAttBigBall_" +  "_ball_ER_N_" + str(N) + "_k_" + str(k) + "_SEED_" + str(SEED) + "_radius_" + str(radius) + "_perctoremove_" + str(perc_to_remove) + ".pickle"
filename_dball = "dballAttBigBall_"  + "_dball_ER_N_" + str(N) + "_k_" + str(k) + "_SEED_" + str(SEED) + "_radius_" + str(radius) + "_perctoremove_" + str(perc_to_remove) + ".pickle"
filename_dg = "dballAttBigBall_"  + "_dg_ER_N_" + str(N) + "_k_" + str(k) + "_SEED_" + str(SEED) + "_radius_" + str(radius) + "_perctoremove_" + str(perc_to_remove) + ".pickle"

print(big_GC_List)

with open(filename_GC, 'wb') as handle:
	pickle.dump(big_GC_List, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(filename_ball,'wb') as handle:
	pickle.dump(big_size_ball, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(filename_dball,'wb') as handle:
	pickle.dump(big_size_dball, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(filename_dg,'wb') as handle:
	pickle.dump(big_dg_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""



"""
N = 10000
k = 4
SEED = 42316

radius_list = [2,3,4]

radius = 4

perc_to_remove = 0.4

G_nx = nx.erdos_renyi_graph(N, k/(N-1), seed = SEED) 

G_nk = nk.nxadapter.nx2nk(G_nx)

G_lattice = nx.grid_graph(dim = [int(math.sqrt(N)),int(math.sqrt(N))],periodic=True)

G_WS = nx.watts_strogatz_graph(N, k, p=0)

G_lattice_nk = nk.nxadapter.nx2nk(G_lattice)

G_WS_nk = nk.nxadapter.nx2nk(G_WS)
"""









"""
(GC_List,size_dball,size_ball,radius_track) = big_attack(G_nk,radius_list,int(perc_to_remove * N))

with open("GC_list.pickle",'wb') as handle:
	pickle.dump(GC_List, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("size_dball.pickle",'wb') as handle:
	pickle.dump(size_dball, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("size_ball.pickle",'wb') as handle:
	pickle.dump(size_ball, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("radius_track.pickle",'wb') as handle:
	pickle.dump(radius_track, handle, protocol=pickle.HIGHEST_PROTOCOL)




print(GC_List)
print(list(zip(size_dball,size_ball)))
print(radius_track)
print(len(radius_track))
"""

#filename_plt_Gnx = "dball_ER_N_" + str(N) + "_k_" + str(k) + "_SEED_" + str(SEED) + ".png"
#filename_pickle_Gnx_dball = "dball_size_ER_N_" + str(N) + "_k_" + str(k) + "_SEED_" + str(SEED) + ".pickle"
#filename_pickle_Gnx_ball = "ball_size_ER_N_" + str(N) + "_k_" + str(k) + "_SEED_" + str(SEED) + ".pickle"


#get_graphs(G_nk, radius_list,int(perc_to_remove*N),filename_plt_Gnx,filename_pickle_Gnx_dball,filename_pickle_Gnx_ball)


"""

filename_plt_lattice = "dball_sims_lattice_N_" + str(N) + "_k_" + str(k) + "_SEED_" + str(SEED) + ".png"
filename_pickle_lattice_dball = "dball_sims_dball_lattice_N_" + str(N) + "_k_" + str(k) + "_SEED_" + str(SEED) + ".pickle"
filename_pickle_lattice_ball = "dball_sims_ball_lattice_N_" + str(N) + "_k_" + str(k) + "_SEED_" + str(SEED) + ".pickle"


filename_plt_WS = "dball_sims_WS_N_" + str(N) + "_k_" + str(k) + "_SEED_" + str(SEED) + ".png"
filename_pickle_WS_dball = "dball_sims_dball_WS_N_" + str(N) + "_k_" + str(k) + "_SEED_" + str(SEED) + ".pickle"
filename_pickle_WS_ball = "dball_sims_ball_WS_N_" + str(N) + "_k_" + str(k) + "_SEED_" + str(SEED) + ".pickle"


get_graphs(G_lattice_nk,radius_list, int(perc_to_remove*N),filename_plt_lattice,filename_pickle_lattice_dball,filename_pickle_lattice_ball)
get_graphs(G_WS_nk,radius_list, int(perc_to_remove*N),filename_plt_WS,filename_pickle_WS_dball,filename_pickle_WS_ball)

"""



"""

print(get_dBN(G,145,radius))
print(get_dBN(G,324,radius))
print(get_dBN(G,551,radius))

print(G.neighbors(551))
print(G.neighbors(145))
print(G.neighbors(324))




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
