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
radius = 2
exp_out=3.5
SEED = 4255

#G_nx = nx.erdos_renyi_graph(N, k/(N-1), seed = SEED) 

#G_nk = make_SF_Graph(N,k,exp_out)

G_nk = make_SF_Graph(N,k,exp_out,SEED)

#(GC_List,size_dball,size_ball,degree_list,counter_list) = perc_process_dBalls_track_balls(G_nk,radius)

#print(degree_list[:300])
#print(size_dball[:300])


printAll(G_nk,96,radius)


#2.5 k = 4
#[(5362, 3), (513, 2), (1914, 2), (1966, 2), (2107, 2), (2243, 2), (2835, 2), (3178, 2), (4168, 2), (4301, 2), (4766, 2), (4960, 2), (4970, 2), (5308, 2), (5419, 2), (5453, 2), (7436, 2), (2317, 3), (3628, 3), (4762, 3), (6892, 3), (9552, 3), (12, 1), (34, 1), (90, 1), (94, 1), (111, 1), (129, 1), (192, 1), (199, 1), (200, 1), (275, 1), (281, 1), (314, 1), (368, 1), (370, 1), (404, 1), (437, 1), (491, 1), (517, 1), (520, 1), (542, 1), (548, 1), (582, 1), (621, 1), (625, 1), (698, 1), (867, 1), (971, 1), (1059, 1)]

#3.5 k = 4
#[(881, 2), (3111, 2), (5085, 2), (8245, 2), (2994, 3), (3313, 3), (3552, 2), (5892, 3), (91, 1), (214, 1), (137, 1), (247, 1), (271, 1), (370, 1), (617, 1), (648, 1), (795, 1), (796, 1), (800, 1), (814, 1), (826, 1), (827, 1), (902, 1), (1142, 1), (1175, 1), (1193, 1), (1218, 1), (1300, 1), (1420, 1), (1499, 1), (1512, 1), (1517, 1), (1572, 1), (1574, 1), (1631, 1), (1654, 1), (1753, 1), (3880, 3), (1805, 1), (1963, 1), (1982, 1), (2074, 1), (2081, 1), (2175, 1), (2289, 1), (2417, 1), (2558, 1), (2573, 1), (2692, 1), (2330, 1), (2711, 1), (2712, 1), (2880, 1), (2921, 1), (3094, 1), (3185, 1), (3202, 1), (3335, 1), (3669, 1), (3748, 1), (3753, 1), (3789, 1), (3826, 1), (4065, 1), (4233, 1), (4251, 1), (4320, 1), (4324, 1), (4447, 1), (4521, 1), (920, 1), (4604, 1), (4658, 1), (4689, 1), (943, 1), (4727, 1), (4733, 1), (4938, 1), (5207, 1), (5332, 1), (5505, 1), (5516, 1), (5577, 1), (5579, 1), (5655, 1), (5661, 1), (5793, 1), (5821, 1), (5835, 1), (4608, 1), (5952, 1), (5998, 1), (6077, 1), (6303, 1), (4237, 1), (3807, 1), (6421, 1), (6655, 1), (6728, 1), (6859, 1), (6929, 1), (7170, 1), (2715, 1), (7390, 1), (7737, 1), (7971, 1), (8126, 1), (8149, 1), (8197, 1), (833, 2), (7483, 1), (7047, 4), (1253, 2), (1520, 2), (1557, 2), (1576, 2), (1061, 1), (6843, 1), (2015, 2), (2073, 2), (2284, 2), (2420, 2), (3384, 2), (3008, 1), (4634, 2), (5247, 2), (5780, 2), (6260, 2), (6554, 2), (6986, 2), (96, 3), (659, 3), (1235, 3), (3219, 3), (3897, 3), (4211, 3), (5725, 3), (5739, 3), (7853, 3), (1076, 1), (1860, 4), (1361, 1), (3419, 1), (1880, 2), (2259, 2), (3188, 2), (3677, 4), (456, 1), (1954, 2), (5146, 4), (7782, 4), (6815, 4), (3426, 2), (959, 1), (7238, 4), (4524, 2), (7952, 4), (3273, 1), (3323, 1), (8253, 4), (3445, 2), (5140, 3), (9434, 4), (114, 3), (383, 3), (1240, 3), (765, 4), (542, 2), (7495, 1), (1862, 2), (2448, 4), (7964, 1), (8153, 1), (926, 3), (2364, 1), (4772, 2), (2584, 2), (1399, 3), (5702, 3), (1418, 3), (1687, 3), (8003, 1), (2313, 3), (4712, 1), (2359, 3), (574, 1), (2779, 3), (3090, 3), (3363, 3), (3594, 3), (3907, 3), (2041, 1), (2599, 1), (4332, 1), (7338, 1), (4109, 3), (2432, 1), (8826, 1), (4317, 3), (2611, 3), (2860, 1), (1479, 1), (5558, 2), (3890, 4), (2117, 2), (4703, 3), (699, 1), (5161, 3), (5264, 3), (782, 1), (5632, 3), (5876, 3), (6330, 3), (9258, 4), (7551, 2), (9111, 3), (1010, 1), (4479, 1), (6789, 3), (7263, 3), (7370, 3), (7427, 3), (1815, 1), (7460, 3), (8846, 3), (6211, 1), (56, 2), (1735, 1), (546, 2), (1304, 3), (103, 2), (204, 2), (6225, 1), (5158, 1), (461, 2), (5336, 4), (4711, 3), (7593, 1), (501, 2), (538, 2), (6952, 1), (702, 2), (934, 2), (8526, 4), (2776, 2), (3198, 1), (2944, 1), (7630, 4), (1359, 1), (3692, 3), (8104, 2), (5227, 3), (334, 2), (1068, 2), (1988, 1), (1139, 2), (2933, 2), (2453, 1), (2565, 1), (146, 2), (4871, 1), (1872, 2), (1909, 2), (5251, 1), (1923, 2), (2150, 1), (4076, 1), (4738, 2), (567, 1), (8419, 3), (8626, 1), (4041, 1), (2054, 2), (225, 1), (2439, 2), (5594, 3), (2539, 2), (8525, 3), (5014, 2), (1554, 3), (2016, 1), (8300, 4), (367, 1), (2910, 2), (3029, 2), (2533, 2), (3266, 2), (4410, 1), (6619, 1), (7267, 1), (8458, 3), (9377, 3), (1409, 2), (3539, 1), (7102, 3), (5038, 1), (3954, 1), (4374, 3), (8341, 1), (8448, 1)]
#[1, 1, 1, 1, 2, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 3, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 4, 1, 1, 2, 2, 2, 4, 1, 2, 4, 4, 5, 1, 1, 5, 2, 5, 1, 1, 5, 2, 3, 5, 4, 4, 4, 5, 1, 1, 2, 5, 1, 1, 4, 1, 2, 2, 4, 3, 4, 4, 1, 4, 1, 4, 1, 4, 4, 4, 4, 4, 1, 1, 1, 1, 4, 1, 1, 4, 4, 1, 1, 2, 5, 2, 4, 1, 4, 4, 1, 4, 4, 4, 5, 2, 3, 1, 1, 4, 4, 4, 4, 1, 4, 4, 1, 3, 1, 2, 4, 3, 3, 1, 1, 2, 5, 4, 1, 3, 3, 1, 3, 3, 5, 2, 1, 1, 5, 1, 4, 2, 4, 3, 3, 1, 3, 2, 1, 1, 3, 1, 3, 3, 1, 3, 1, 1, 2, 1, 4, 1, 1, 3, 1, 3, 4, 3, 3, 2, 4, 1, 5, 1, 3, 3, 3, 3, 1, 1, 1, 4, 4, 3, 1, 4, 1, 1, 2, 1, 1]






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
