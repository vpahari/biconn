import csv
import glob
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle
import math
from functools import reduce
import networkx as nx
import random
import sys
from operator import itemgetter
import os
import igraph as ig
import networkit as nk


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


def make_graphs_into_one_multiple_graphs_alpha(G_list,num_edges_to_connect,alpha):

	counter = 0

	G = nk.Graph()

	print(G.nodes())
	print(G.edges())

	nodes_list = []

	alpha_nodes_list = []

	size_G_nodes = 0

	for i in G_list:

		G_nodes = list(i.nodes())
		G_edges = list(i.edges())

		G_nodes = list(map(lambda x : x + size_G_nodes, G_nodes))
		size_G_nodes += len(G_nodes)

		nodes_list.append(G_nodes)

		num_nodes_to_sample = len(G_nodes) * alpha

		curr_alpha = random.sample(G_nodes, num_nodes_to_sample)

		alpha_nodes_list.append(curr_alpha)

		for n in G_nodes:
			G.addNode()

		for i,j in G_edges:
			u = size_G_nodes + i
			v = size_G_nodes + j
			G.addEdge(u,v)

	for i in range(num_edges_to_connect):
		
		(u,v) = get_random_u_v(alpha_nodes_list)
		G.addEdge(u,v)

	return G


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
	

	(G_nk, set_of_connected_nodes) = make_graphs_into_one_multiple_graphs(list_Graphs,num_edges)

	G_nx = nk.nxadapter.nk2nx(G_nk)

	return (G_nx, set_of_connected_nodes)



def make_WS_graph(dim,size,nei,p,SEED):

	N = size ** dim

	random.seed(SEED)

	igG = ig.Graph.Watts_Strogatz(dim,size,nei,p)

	allEdges = igG.get_edgelist()

	fixed_G = nx.Graph()

	listOfNodes = [i for i in range(N)]

	fixed_G.add_nodes_from(listOfNodes)

	fixed_G.add_edges_from(allEdges)

	return fixed_G



def turn_list_to_str(l):

	s = ""

	for i in l:

		s += str(i) 
		s += "," 

	return s


def get_GC(G):

	conn = list(nx.connected_component_subgraphs(G))
	GC = max(conn, key = len)

	return GC.number_of_nodes()


def get_GC_component(G):
	conn = list(nx.connected_component_subgraphs(G))
	GC = max(conn, key = len)

	return GC


def get_GC_nodes(G):

	conn = list(nx.connected_component_subgraphs(G))
	GC = max(conn, key = len)

	return list(GC.nodes())



def get_betweenness_score_list(G, node_list):

	between = nx.betweenness_centrality(G)

	final_list = []

	for node in node_list:

		final_list.append(round(between[node], 4))

	return final_list


def get_degree_score_list(G,node_list):

	final_list = []

	for node in node_list:

		final_list.append(G.degree(node))

	return final_list


def get_coreness_score_list(G,node_list):

	coreness = nx.core_number(G)

	final_list = []

	for node in node_list:

		final_list.append(coreness[node])

	return final_list



def add_into_set(s,new_s):

	for i in new_s:

		s.add(i)

	return s

def get_dBN(G,node,radius):

	dBall = set([node])
	ball = set([node])
	edge_set = set([])

	for i in range(radius):

		neighbor = []

		for j in dBall:

			for n in list(G.neighbors(j)):

				if n in ball:
					continue

				neighbor.append(n)

				edge_set.add((n,j))

		ball = add_into_set(ball,neighbor)

		dBall = set(neighbor.copy())

	return (list(dBall),list(ball),list(edge_set))



def get_all_dBN(G,radius):

	all_nodes = get_GC_nodes(G)

	assert(get_GC(G) == len(all_nodes))  
	

	dict_nodes_dBall = {}
	dict_nodes_ball = {}
	dict_nodes_x_i = {}

	for n in all_nodes:

		(dBall,ball,edge_set) = get_dBN(G,n,radius)


		dict_nodes_dBall[n] = len(dBall)
		dict_nodes_ball[n] = len(ball)
		dict_nodes_x_i[n] = len(dBall) / len(ball)

	return (dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i)



def get_all_same_x_i(sorted_list,x_i_value):

	node_list = []

	for i in sorted_list:

		if i[1] == x_i_value:

			node_list.append(i[0])

	return node_list




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




def dBalls_attack(G_copy,radius, init_filename, position, path):

	G = G_copy.copy()

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

	num_nodes_to_remove = G.number_of_nodes()

	main_node_list = []

	dball_list =[]

	ball_list = []

	counterNew = 0

	try:
		print("A")
		os.mkdir(path)
		print("A")

	except:
		pass


	init_GC_nodes = get_GC_nodes(G)

	while counter < num_nodes_to_remove:

		print(counter)

		(dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i) = get_all_dBN(G,radius)

		list_to_remove = dict_to_sorted_list(dict_nodes_x_i)

		if len(list_to_remove) == 0:
			break
		

		node = get_random_dball(list_to_remove)
		(dBall,ball,e_set) = get_dBN(G,node,radius) 

		combined_list = [node] + dBall

		#between_list = get_betweenness_score_list(G,combined_list)
		degree_list = get_degree_score_list(G,combined_list)
		#coreness_list = get_coreness_score_list(G,combined_list)



		#degree_list_mainNode.append(degree_list[0])
		#betweenness_list_mainNode.append(between_list[0])
		#coreness_list_mainNode.append(coreness_list[0])

		#degree_list_removedNode += degree_list[1:]
		#betweenness_list_removedNode += between_list[1:]
		#coreness_list_removedNode += coreness_list[1:]
		

		size_dball.append(len(dBall))
		size_ball.append(len(ball))

		main_node_list.append(node)
		dball_list.append(dBall)
		ball_list.append(ball)

		removed_degree_str = turn_list_to_str(degree_list[1:])
		#removed_bet_str = turn_list_to_str(between_list[1:])
		#removed_core_str = turn_list_to_str(coreness_list[1:])

		removed_bet_str = ""
		removed_core_str = ""

		between_list = [0]
		coreness_list = [0]


		curr_GC = get_GC(G)

		#print(dBall)
		#print(ball)

		create_graphs(G, node, dBall, init_filename, position, counterNew, radius, path,degree_list[0],between_list[0],coreness_list[0],removed_degree_str,removed_bet_str,removed_core_str,len(dBall),len(ball),curr_GC, init_GC_nodes)

		for i in dBall:
			G.remove_node(i)
			init_GC_nodes.remove(i)
			counter += 1

		GC_List.append(get_GC(G))

		counter_list.append(counter)

		counterNew += 1



	return (GC_List,counter_list,size_dball,size_ball,degree_list_mainNode,betweenness_list_mainNode,coreness_list_mainNode,degree_list_removedNode,betweenness_list_removedNode,coreness_list_removedNode,main_node_list,dball_list,ball_list)



def get_all_edges(G,node,edge_list_set):

	neighbors = G.neighbors(node)



def remove_all_edges_with_node(edge_list,node):

	new_edges = []

	for i in edge_list:

		if node in i:

			edge_list.remove(i)

			new_edges.append(i)

	return new_edges

def remove_all_nodes_from(all_nodes, nodes_list):

	for i in nodes_list:

		try:	

			all_nodes.remove(i)

		except:
			continue


def get_edges_with_list(G,nodes_list):

	edge_list = set([])

	for node in nodes_list:

		neighbors = G.neighbors(node)

		for i in neighbors:

			edge_list.add((node, i))

	return edge_list





def create_graphs(G, main_node, nodes_list, init_filename, position, counter, radius, path,degree_mainNode,between_mainNode,coreness_mainNode,removed_degree_str,removed_bet_str,removed_core_str,dBall_len,ball_len,curr_GC,init_GC_nodes):

	new_filename = path + init_filename + str(counter) + ".png"

	G_nodes = init_GC_nodes.copy()

	G_edges = list(get_edges_with_list(G, G_nodes))

	remove_all_nodes_from(G_nodes, nodes_list)

	G_nodes.remove(main_node)

	new_edges = []

	plt.figure(figsize=(10,10))

	(dBall,ball,edge_set) = get_dBN(G,main_node,radius)

	for i in nodes_list:

		edges_remaining = remove_all_edges_with_node(G_edges, i)

		new_edges += edges_remaining

	new_edges += edge_set

	nx.drawing.nx_pylab.draw_networkx_nodes(G, nodelist = G_nodes, pos = position, node_size = 0.4, with_labels = False, node_color = 'r')

	nx.drawing.nx_pylab.draw_networkx_edges(G, edgelist = G_edges, pos = position, arrowsize = 0.1, with_labels = False, edge_color = 'b')

	nx.drawing.nx_pylab.draw_networkx_nodes(G, pos = position, nodelist = ball, node_size = 10, with_labels = False, node_color = 'k')

	nx.drawing.nx_pylab.draw_networkx_nodes(G, pos = position, nodelist = dBall, node_size = 10, with_labels = False, node_color = 'magenta')

	nx.drawing.nx_pylab.draw_networkx_nodes(G, pos = position, nodelist = [main_node],  node_size = 10, with_labels = False, node_color = 'r')

	nx.drawing.nx_pylab.draw_networkx_edges(G, pos = position, edgelist = new_edges, arrowsize = 4, with_labels = False, edge_color = 'y')

	(x,y) = change_position_to_array(new_position)

	delta = 0.02

	plt.text(x,y,"GC_size = " + str(curr_GC))

	plt.text(x, y - delta,"dBall_size = " + str(dBall_len))

	plt.text(x, y - delta * 2,"ball_size = " + str(ball_len))

	plt.text(x, y - delta * 3,"degree_main = " + str(degree_mainNode))

	plt.text(x, y - delta * 4,"between_main = " + str(between_mainNode))

	plt.text(x, y - delta * 5,"coreness_main = " + str(coreness_mainNode))

	plt.text(x, y - delta * 6,"degree_removed = " + str(removed_degree_str))

	plt.text(x, y - delta * 7,"between_removed = " + str(removed_bet_str))

	plt.text(x, y - delta * 8,"coreness_removed = " + str(removed_core_str))

	plt.savefig(new_filename)

	plt.clf()






"""
for fn in glob.glob('*_MAINNODE*.pickle'):
	
	with open(fn, 'rb') as handle:

		main_node_track = pickle.load(handle)

	split_list = fn.split("_")

	print(split_list)

	N = int(split_list[5])

	k = int(split_list[7])

	SEED = int(split_list[9])

	radius = int(split_list[11])


for fn in glob.glob('*_DBALLNODE*.pickle'):
	
	with open(fn, 'rb') as handle:

		dball_node = pickle.load(handle)
"""


def dict_to_sorted_list_NA(d):

	new_list = list(d.items())

	random.shuffle(new_list)

	final_list = sorted(new_list, key = itemgetter(1))

	return final_list

def create_all_graphs(G, radius,pathNA,pathadp):

	init_fname = "WS_N_" + str(N) + "_dim_" + str(dim)  + "_nei_" +  str(nei)+ "_p_" + str(p) + "_SEED_" + str(SEED) + "_radius_" + str(radius) + "_numBalls_"  

	counter = 0

	#position=nx.spring_layout(G) 

	plt.figure(figsize=(10,10))

	nx.drawing.nx_pylab.draw_networkx(G, pos=position, arrowsize = 0.1, node_size = 0.4, with_labels = False, edge_color = 'b')

	curr_name = init_fname + str(counter) + ".png"

	plt.savefig(curr_name)

	plt.clf()

	#try:

	(GC_List,counter_list,size_dball,size_ball,degree_list_mainNode,betweenness_list_mainNode,coreness_list_mainNode,degree_list_removedNode,betweenness_list_removedNode,coreness_list_removedNode,main_node_list,dball_list,ball_list) = dBalls_attack_NA(G,radius, init_fname, position, pathNA)

	(GC_List,counter_list,size_dball,size_ball,degree_list_mainNode,betweenness_list_mainNode,coreness_list_mainNode,degree_list_removedNode,betweenness_list_removedNode,coreness_list_removedNode,main_node_list,dball_list,ball_list) = dBalls_attack(G,radius, init_fname, position, pathadp)

	#except:
	#pass

	"""
	while counter < len(main_node_list):

		print(counter)

		main_node = main_node_list[counter]

		nodes_list = dball_list[counter]

		create_graphs(G, main_node, nodes_list, init_fname, position, counter, radius, path)

		counter += 1
	"""
def turn_dict_to_list(d):

	new_list = []

	for k in list(d.keys()):

		new_list.append((k,d[k]))

	return new_list



def change_position_to_array(position):

	list_values = list(position.values())

	change_type_list_values = list(map(lambda x : list(x), list_values))

	#print(change_type_list_values)

	x_values = list(map(lambda x : x[0], change_type_list_values))
	y_values = list(map(lambda x : x[1], change_type_list_values))

	#print(x_values)
	#print(y_values)

	top_y_value = max(y_values)
	left_x_value = min(x_values)

	return (left_x_value, top_y_value)



def ADA_attack(G_copy,num_nodes_to_remove, init_filename, position, path):

	print(position)

	G = G_copy.copy()

	GC_List = []

	GC_List.append(get_GC(G))

	degree_list = []

	#path += init_filename

	try:
		print("A")
		os.mkdir(path)
		print("A")

	except:
		pass


	init_GC_nodes = get_GC_nodes(G)


	for i in range(num_nodes_to_remove):

		print(i)

		new_filename =  path + init_filename + str(i) + ".png"

		GC = get_GC_component(G)

		#G_nodes = init_GC_nodes

		degree_sequence_dict = dict(GC.degree())

		degree_sequence = turn_dict_to_list(degree_sequence_dict)

		random.shuffle(degree_sequence)

		degree_sequence.sort(key = itemgetter(1), reverse = True)

		node_to_remove = degree_sequence[0][0]

		degree_of_node = degree_sequence[0][1]

		degree_list.append(G.degree(node_to_remove))

		G_edges = list(get_edges_with_list(G, init_GC_nodes))

		plt.figure(figsize=(10,10))

		init_GC_nodes.remove(node_to_remove)

		edges_remaining = remove_all_edges_with_node(G_edges, node_to_remove)

		nx.drawing.nx_pylab.draw_networkx_nodes(G, nodelist = init_GC_nodes, pos = position, node_size = 0.4, with_labels = False, node_color = 'r')

		nx.drawing.nx_pylab.draw_networkx_edges(G, edgelist = G_edges, pos = position, arrowsize = 0.1, with_labels = False, edge_color = 'b')

		nx.drawing.nx_pylab.draw_networkx_nodes(G, pos = position, nodelist = [node_to_remove], node_size = 10, with_labels = False, node_color = 'magenta')

		nx.drawing.nx_pylab.draw_networkx_edges(G, pos = position, edgelist = edges_remaining, arrowsize = 4, with_labels = False, edge_color = 'y')

		G.remove_node(node_to_remove)

		curr_GC = get_GC(G)

		GC_List.append(curr_GC)

		(x,y) = change_position_to_array(position)

		delta = 0.05

		plt.text(x,y,"degree = " + str(degree_of_node))

		plt.text(x,y-delta,"GC_size = " + str(curr_GC))

		plt.savefig(new_filename)

		plt.clf()

	return (GC_List, degree_list)



def dBalls_attack_NA(G_copy,radius, init_filename, position, path):

	G = G_copy.copy()

	#print(G.nodes())

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

	GC_List.append(get_GC(G))
	counter_list.append(counter)

	num_nodes_to_remove = G.number_of_nodes()

	(dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i) = get_all_dBN(G,radius)

	list_to_remove = dict_to_sorted_list_NA(dict_nodes_x_i)

	counter_for_nodes = 0

	print(dict_nodes_x_i)

	print(list_to_remove)

	init_GC_nodes = get_GC_nodes(G)

	counterNew = 0

	main_node_list = []

	dball_list =[]

	ball_list = []

	try:
		print("A")
		os.mkdir(path)
		print("A")

	except:
		pass

	while counter_for_nodes < len(list_to_remove):

		curr_nodes_set = set(list(G.nodes()))

		node = list_to_remove[counter_for_nodes][0]

		print(node,dict_nodes_dBall[node])


		if node not in curr_nodes_set:
			counter_for_nodes += 1
			continue


		(dBall,ball,e_set) = get_dBN(G,node,radius) 

		#print(G.nodes())


		if len(dBall) == 0:
			counter_for_nodes += 1
			continue


		size_dball.append(len(dBall))
		size_ball.append(len(ball))

		combined_list = [node] + dBall

		#between_list = get_betweenness_score_list(G,combined_list)
		degree_list = get_degree_score_list(G,combined_list)
		#coreness_list = get_coreness_score_list(G,combined_list)

		#degree_list_mainNode.append(degree_list[0])
		#betweenness_list_mainNode.append(between_list[0])
		#coreness_list_mainNode.append(coreness_list[0])

		#degree_list_removedNode += degree_list[1:]
		#betweenness_list_removedNode += between_list[1:]
		#coreness_list_removedNode += coreness_list[1:]

		main_node_list.append(node)

		dball_list.append(dBall)

		ball_list.append(ball)

		curr_GC = get_GC(G)

		GC_List.append(GC)

		removed_degree_str = turn_list_to_str(degree_list[1:])
		#removed_bet_str = turn_list_to_str(between_list[1:])
		#removed_core_str = turn_list_to_str(coreness_list[1:])

		#removed_degree_str = ""
		removed_bet_str = ""
		removed_core_str = ""

		between_list = [0]
		coreness_list = [0]



		create_graphs(G, node, dBall, init_filename, position, counterNew, radius, path,degree_list[0],between_list[0],coreness_list[0],removed_degree_str,removed_bet_str,removed_core_str,len(dBall),len(ball),curr_GC, init_GC_nodes)

		for i in dBall:
			G.remove_node(i)
			init_GC_nodes.remove(i)
			counter += 1

		counter_list.append(counter)

		counter_for_nodes += 1

		counterNew += 1


	return (GC_List,counter_list,size_dball,size_ball,degree_list_mainNode,betweenness_list_mainNode,coreness_list_mainNode,degree_list_removedNode,betweenness_list_removedNode,coreness_list_removedNode,main_node_list,dball_list,ball_list)





def create_graphs_degree(G, position,path):

	num_nodes_to_remove = int(G.number_of_nodes() * 0.5)

	init_filename = "WS_N_" + str(N) + "_dim_" + str(dim)  + "_nei_" +  str(nei)+ "_p_" + str(p) + "_SEED_" + str(SEED) + "_radius_" + str(radius) + "/" + "_numBalls_" 

	try:
		(GC_List, degree_list) = ADA_attack(G,num_nodes_to_remove, init_filename, position, path)

	except:
		pass



def create_graphs_between(G, position,path):

	num_nodes_to_remove = int(G.number_of_nodes() * 0.5)

	init_filename = "ER_" + str(N) + "_" + str(k) + "_" +  str(SEED)+ "_" + str(radius) + "_numBalls_" 

	(GC_List, between_list) = ABA_attack(G,num_nodes_to_remove, init_filename, position, path)



def get_position_GC(GC_nodes,position):

	d = {}

	for i in GC_nodes:

		pos = position[i]

		d[i] = pos

	return d



def get_four_corners(GC_position):

	x_values = list(map(lambda x : x[0], GC_position))
	y_values = list(map(lambda x : x[1], GC_position))

	max_x = max(x_values)
	min_x = min(x_values)

	max_y = max(y_values)
	min_y = min(y_values)

	return (min_x, max_x, min_y, max_y) 



N=int(sys.argv[1]) 

k_intra=float(sys.argv[2])

k_inter=float(sys.argv[3])

SEED=int(sys.argv[4])

num_modules = int(sys.argv[5])

radius = int(sys.argv[6])

#num_times = int(sys.argv[7])

G = make_WS_graph(dim,N,nei,p,SEED)

position=nx.spring_layout(G)

GC_nodes = get_GC_nodes(G)

new_position = get_position_GC(GC_nodes,position)

#plt.figure(figsize=(10,10))

path = os.getcwd() + "/" + "MOD_N_" + ".png"

GC = get_GC_nodes(G)

try:
	os.mkdir(path)

except:
	pass

nx.drawing.nx_pylab.draw(G)

plt.savefig(path)

plt.clf()


"""
path_dball_NA = path + "NADBALL" + "/"

path_dball = path + "DBALL" + "/"

path_degree = path + "DEG" + "/"

path_bet = path + "BET" + "/"

create_all_graphs(G, radius, path_dball_NA, path_dball)

create_graphs_degree(G, new_position,path_degree)

#create_graphs_between(G, position,path_bet)

#change_position_to_array(position)

"""



