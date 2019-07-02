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

def make_SF_Graph(N,k,exp_out,SEED):

	random.seed(SEED)

	num_edges = int((N * k) / 2)

	igG = ig.Graph.Static_Power_Law(N,num_edges,exp_out)

	allEdges = igG.get_edgelist()

	fixed_G = nx.Graph()

	listOfNodes = [i for i in range(N)]

	fixed_G.add_nodes_from(listOfNodes)

	fixed_G.add_edges_from(allEdges)

	return fixed_G

def get_GC(G):

	conn = list(nx.connected_component_subgraphs(G))
	GC = max(conn, key = len)

	return GC.number_of_nodes()

def get_GC_nodes(G):

	conn = list(nx.connected_component_subgraphs(G))
	GC = max(conn, key = len)

	return list(GC.nodes())



def get_betweenness_score_list(G, node_list):

	between = nx.betweenness_centrality(G)

	final_list = []

	for node in node_list:

		final_list.append(between[node])

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

	while counter < num_nodes_to_remove:

		print(counter)

		(dict_nodes_dBall,dict_nodes_ball,dict_nodes_x_i) = get_all_dBN(G,radius)

		list_to_remove = dict_to_sorted_list(dict_nodes_x_i)

		if len(list_to_remove) == 0:
			break
		

		node = get_random_dball(list_to_remove)
		(dBall,ball,e_set) = get_dBN(G,node,radius) 

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

		main_node_list.append(node)
		dball_list.append(dBall)
		ball_list.append(ball)


		#print(dBall)
		#print(ball)

		create_graphs(G, node, dBall, init_filename, position, counterNew, radius, path)

		for i in dBall:
			G.remove_node(i)
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

		all_nodes.remove(i)


def create_graphs(G, main_node, nodes_list, init_filename, position, counter, radius, path):

	new_filename = path + init_filename + str(counter) + ".png"

	G_nodes = list(G.nodes())
	G_edges = list(G.edges())

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

def create_all_graphs(G, radius,path):

	init_fname = "SF_" + str(N) + "_" + str(k) + "_"+ str(exp_out) + "_" +  str(SEED)+ "_" + str(radius) + "_numBalls_"  

	counter = 0

	position=nx.spring_layout(G) 

	plt.figure(figsize=(10,10))

	nx.drawing.nx_pylab.draw_networkx(G, pos=position, arrowsize = 0.1, node_size = 0.4, with_labels = False, edge_color = 'b')

	curr_name = init_fname + str(counter) + ".png"

	plt.savefig(curr_name)

	plt.clf()

	(GC_List,counter_list,size_dball,size_ball,degree_list_mainNode,betweenness_list_mainNode,coreness_list_mainNode,degree_list_removedNode,betweenness_list_removedNode,coreness_list_removedNode,main_node_list,dball_list,ball_list) = dBalls_attack(G,radius, init_fname, position, path)

	"""
	while counter < len(main_node_list):

		print(counter)

		main_node = main_node_list[counter]

		nodes_list = dball_list[counter]

		create_graphs(G, main_node, nodes_list, init_fname, position, counter, radius, path)

		counter += 1
	"""



N=int(sys.argv[1]) 

k=int(sys.argv[2])

exp_out = float(sys.argv[3])

SEED=int(sys.argv[4])

radius = int(sys.argv[5])

G = make_SF_Graph(N,k,exp_out,SEED)

path = os.getcwd() + "/"

path += "SF_" + str(N) + "_" + str(k) + "_"+ str(exp_out) + "_" +  str(SEED)+ "_" + str(radius) + "/"

os.mkdir(path)

#create_all_graphs(G, radius, path)


















