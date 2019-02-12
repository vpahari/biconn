import networkx as nx
import networkit as nk
import random
import sys
import math
from functools import reduce
import csv
from operator import itemgetter
import matplotlib.pyplot as plt






#N=int(sys.argv[1]) # number of nodes
#k=int(sys.argv[2])
#SEED=int(sys.argv[3])
#M_out = int(sys.argv[4])
#r = float(sys.argv[5])

"""
N=int(sys.argv[1]) # number of nodes
k=int(sys.argv[2]) # average degree
SEED=int(sys.argv[3])
alpha = float(sys.argv[4])
r = float(sys.argv[5])
#gType = str(sys.argv[6])

N = 10000

k = 4

SEED = 312

alpha = 0.01

r = 0.1


numModules = 2

M = int((numModules * N * k) / 2)

M_out = int(M * alpha)

M_in = int(M * (1 - alpha))

k_in = (M_in * 2) / (numModules * N)

p_in = k_in / (N-1)

numNodesToConnect = int(r * N)


print(M_in)
print(k_in)
print(p_in)
"""



def indexToTake(graphList,index):
	#print(index)

	gList = list(map(lambda x : x[index], graphList))

	gAvg = reduce(lambda x,y : x+y, gList)

	gAvg = gAvg / len(gList)

	gStdDevList = list(map(lambda x : ((x - gAvg)**2),gList))

	gStdDev = math.sqrt((reduce(lambda x,y: x+y,gStdDevList)) / len(gList))

	return (gAvg,gStdDev)



def createOrder(G):
	allNodes = list(G.nodes())
	degreeDict = dict(G.degree(allNodes))
	degreeDictItems = list(degreeDict.items())
	random.shuffle(degreeDictItems)
	degreeDictItemsSorted = sorted(degreeDictItems, key = itemgetter(1),reverse = True)
	onlyNodes = list(map(lambda x:x[0],degreeDictItemsSorted))
	return onlyNodes

def createOrder_BI(G):
	BI_dict = dict(nx.betweenness_centrality(G,None,True))
	BI_DictItems = list(BI_dict.items())
	random.shuffle(BI_DictItems)
	BI_DictItemsSorted = sorted(BI_DictItems, key = itemgetter(1),reverse = True)
	onlyNodes = list(map(lambda x:x[0],BI_DictItemsSorted))
	return onlyNodes




def adaptive_degree(G,numNodesToRemove):
	degree = nk.centrality.DegreeCentrality(G)
	degree.run()
	listToRemove = degree.ranking()[:numNodesToRemove]

	listToRemove = list(map(lambda x : x[0], listToRemove))

	for n in listToRemove:
		G.removeNode(n)

	return listToRemove


def adaptive_betweenness(G,numNodesToRemove):
	between = nk.centrality.DynBetweenness(G)
	between.run()
	listToRemove = between.ranking()[:numNodesToRemove]

	listToRemove = list(map(lambda x : x[0], listToRemove))

	for n in listToRemove:
		G.removeNode(n)

	return listToRemove


def get_degree_sequence(G):
	degree = nk.centrality.DegreeCentrality(G)
	degree.run()

	degree_sequence = degree.ranking()
	degree_sequence = list(map(lambda x : x[1], listToRemove))

	return degree_sequence


def getSize(GC_nodes, size):
	counterGC1 = 0
	counterGC2 = 0

	for node in GC_nodes:

		if node < size:
			counterGC1 += 1

		else:
			counterGC2 += 1


	return (counterGC1,counterGC2)



def get_percolation_threshold(sgc_List):
	return sgc_List.index(max(sgc_List))


def percolation(G_copy, step_size, typeOfAttack, percentage_to_attack):

	G = copy_graph(G_copy)

	counter = 0

	gc1_List = []
	gc2_List = []

	gc_List = []

	sgc_List = []

	GC_nodes_List = []

	originalSize = G.numberOfNodes()

	numNodesToRemove = int(originalSize * step_size)

	print(numNodesToRemove)

	while counter < percentage_to_attack:

		#print("conn")

		comp = nk.components.DynConnectedComponents(G)
		comp.run()
		
		#connected_comps_sizes = comp.getComponentSizes()

		connected_comps = comp.getComponents()

		connected_comps.sort(key = len, reverse = True)

		#print(connected_comps_sizes)

		GC_size = len(connected_comps[0])
		SGC_size = len(connected_comps[1])

		GC_nodes = max(connected_comps, key = len)

		(GC1_size, GC2_size) = getSize(GC_nodes,int(originalSize / 2))

		gc1_List.append(GC1_size)
		gc2_List.append(GC2_size)

		gc_List.append(GC_size)
		sgc_List.append(SGC_size)

		

		if typeOfAttack == "ABA":
			listToRemove = adaptive_betweenness(G,numNodesToRemove)

		elif typeOfAttack == "ADA":
			listToRemove = adaptive_degree(G,numNodesToRemove)


		GC_nodes_List += listToRemove

		counter += step_size

	return (gc_List,gc1_List,gc2_List,GC_nodes_List, sgc_List)



#takes two graphs and then makes it into one modular networks of size 2
def change_nodes(G1,G2):
	G1_num_nodes = G1.numberOfNodes()
	G2_num_nodes = G2.numberOfNodes()

	for i in range(G2_num_nodes):
		G1.addNode()

	allEdges = list(G2.edges())

	allEdges = map(lambda x : (G1_num_nodes + x[0], G1_num_nodes + x[1]) ,allEdges)

	for (i,j) in allEdges:
		G1.addEdge(i,j)



def connecting_graphs(G,nodes_to_connect_1,nodes_to_connect_2):
	for i in range(len(nodes_to_connect1)):
		G.addEdge(nodes_to_connect1[i], nodes_to_connect2[i])



def all_possible_connections(G,number_of_edges):

	all_nodes = G.nodes()

	comp = nk.components.DynConnectedComponents(G)
	comp.run()

	connected_comps = comp.getComponents()

	connected_comps.sort(key = len, reverse = True)
	
	GC1_nodes = connected_comps[0]
	GC2_nodes = connected_comps[1]

	connected_comps.sort(key = len, reverse = True)

	all_combinations_1 = list(itertools.combinations(GC1_nodes, number_of_edges))
	all_combinations_2 = list(itertools.combinations(GC2_nodes, number_of_edges))

	return (all_combinations_1, all_combinations_2)


def do_percolation(G,number_of_edges,percentage_to_attack):

	(all_combinations_1, all_combinations_2) = all_possible_connections(G,number_of_edges)

	gc_List = []
	gc1_List = []
	gc2_List = []
	GC_nodes_List = []
	sgc_List = []

	gc_min = G.numberOfNodes()

	best_combinations_1 = []
	best_combinations_2 = []

	for i in range(len(all_combinations_1)):

		G_copy = G.copy()

		connecting_graphs(G_copy,all_combinations_1[i],all_combinations_2[i])

		(gc_List_ABA,gc1_List_ABA,gc2_List_ABA,GC_nodes_List_ABA, sgc_List_ABA) = percolation(G_copy, 0.01, "ABA",percentage_to_attack)
		(gc_List_ADA,gc1_List_ADA,gc2_List_ADA,GC_nodes_List_ADA, sgc_List_ADA) = percolation(G_copy, 0.01, "ADA",percentage_to_attack)

		if gc_List_ABA[-1] < gc_min and gc_List_ABA[-1] <= gc_List_ADA[-1]:

			gc_min = gc_List_ABA[-1]
			(gc_List,gc1_List,gc2_List,GC_nodes_List, sgc_List) = (gc_List_ABA,gc1_List_ABA,gc2_List_ABA,GC_nodes_List_ABA, sgc_List_ABA)

			best_combinations_1 = all_combinations_1
			best_combinations_2 = all_combinations_2

		if gc_List_ADA[-1] < gc_min and gc_List_ADA[-1] <= gc_List_ABA[-1]:

			gc_min = gc_List_ADA[-1]
			(gc_List,gc1_List,gc2_List,GC_nodes_List, sgc_List) = (gc_List_ADA,gc1_List_ADA,gc2_List_ADA,GC_nodes_List_ADA, sgc_List_ADA)

			best_combinations_1 = all_combinations_1
			best_combinations_2 = all_combinations_2




def intersection(l1,l2):
	set_l1 = set(l1)

	l3 = []

	for n in l2:
		if n in set_l1:
			l3.append(n)

	return l3



def find_best_nodes(G,step_size,percentage_to_attack):
	G_copy = G.copy()

	(gc_List_ABA,gc1_List_ABA,gc2_List_ABA,GC_nodes_List_ABA, sgc_List_ABA) = percolation(G_copy,step_size,"ABA",percentage_to_attack)
	(gc_List_ADA,gc1_List_ADA,gc2_List_ADA,GC_nodes_List_ADA, sgc_List_ADA) = percolation(G_copy,step_size,"ADA",percentage_to_attack)

	intersection_nodes = intersection(GC_nodes_List_ABA,GC_nodes_List_ADA)



	return intersection_nodes


def connect_random_nodes(G,numEdges):
	single_module_size = int(G.numberOfNodes() / 2)

	nodes1 = G.nodes()

	comp = nk.components.DynConnectedComponents(G)
	comp.run()

	connected_comps = comp.getComponents()

	connected_comps.sort(key = len, reverse = True)

	GC1_nodes = connected_comps[0]
	GC2_nodes = connected_comps[1]

	connections = set([])

	counter = 0

	while counter < numEdges:

		i = random.choice(GC1_nodes)
		j = random.choice(GC2_nodes)

		if (i,j) not in connections:
			G.addEdge(i,j)
			connections.add((i,j))

			counter += 1

	return connections



def copy_graph(G):
	G_copy = G.copyNodes()

	edges = G.edges()

	for (i,j) in edges:
		G_copy.addEdge(i,j)

	return G_copy



def get_GC(G):
	comp = nk.components.DynConnectedComponents(G)
	comp.run()

	return comp.getComponentSizes()[0]



def check_GC(G_copy,nodesToRemove):
	G = copy_graph(G_copy)

	for n in nodesToRemove:
		G.removeNode(n)

	GC_final = get_GC(G)

	return GC_final




def changing_edge_percentages(G):

	edges_percentage = 0.05
	percentage_to_add = 0.05
	num_nodes = G.numberOfNodes()
	intersection_list = []
	percolation_threshold_list = []

	while edges_percentage < 0.9:

		edges_to_add = int(num_nodes * edges_percentage)

		G_copy = copy_graph(G) 

		connect_random_nodes(G_copy,edges_to_add)

		print(G_copy.numberOfEdges())

		step_size = 0.01
		percentage_to_attack = 0.5

		(gc_List1,gc1_List1,gc2_List1,GC_nodes_List1, sgc_List1) = percolation(G_copy, step_size, "ABA", percentage_to_attack)
		(gc_List2,gc1_List2,gc2_List2,GC_nodes_List2, sgc_List2) = percolation(G_copy, step_size, "ADA", percentage_to_attack)

		toCheck = int(len(GC_nodes_List1) / 2)

		intersect = intersection(GC_nodes_List1[:toCheck],GC_nodes_List2[:toCheck])

		p_c1 = get_percolation_threshold(sgc_List1)
		p_c2 = get_percolation_threshold(sgc_List2)

		percolation_threshold_list.append((p_c1,p_c2))

		intersection_list.append(intersect)

		edges_percentage += percentage_to_add

	return (percolation_threshold_list, intersection_list)


def get_percentage(nodes_removed,nodes_in_modular):

	counter = 0

	for i in nodes_removed:
		if i in nodes_in_modular:
			counter += 1

	return float(counter / len(nodes_in_modular))



def remove_nodes_from_list(G_nodes,nodes_removed):
	new_nodes_removed = set(nodes_removed)
	final_list = []

	for i in G_nodes:
		if i not in new_nodes_removed:
			final_list.append(i)

	return final_list


def get_optimal_set(G_init, edge_percentage,percentage_to_attack,typeOfAttack):

	G_size = G_init.numberOfNodes() 

	G_all_nodes = G_init.nodes()

	edges_to_add = int(edge_percentage * G_size / 2)

	print(edges_to_add)

	connections = list(connect_random_nodes(G_init,edges_to_add))

	nodes_1 = set(list(map(lambda x : x[0],connections)) + list(map(lambda x : x[1],connections)))

	print(len(nodes_1))

	G = copy_graph(G_init)

	num_nodes_to_remove = int(percentage_to_attack * G_size)

	if typeOfAttack == "ABA":
		nodes_removed = adaptive_betweenness(G,num_nodes_to_remove)

	elif typeOfAttack == "ADA":
		nodes_removed = adaptive_degree(G,num_nodes_to_remove)

	curr_GC = get_GC(G)

	percentage_in_modular = get_percentage(nodes_removed,nodes_1)

	actual_nodes_removed = nodes_removed.copy()

	#print(actual_nodes_removed)

	G_nodes_removed = remove_nodes_from_list(G_all_nodes,nodes_removed)

	for i in range(G_size):

		G = copy_graph(G_init)

		random_node_1 = random.choice(nodes_removed)

		random_node_2 = random.choice(G_nodes_removed)
 
		new_nodes_to_remove = nodes_removed.copy()

		new_nodes_to_remove.append(random_node_2)

		#print(new_nodes_to_remove)

		#print(nodes_removed)

		new_nodes_to_remove.remove(random_node_1)

		new_GC = check_GC(G,new_nodes_to_remove)

		if new_GC < curr_GC:

			new_GC = curr_GC

			percentage_in_modular = get_percentage(new_nodes_to_remove,nodes_1)

			actual_nodes_removed = new_nodes_to_remove.copy()

			nodes_removed = new_nodes_to_remove.copy()

			G_nodes_removed.remove(random_node_2)

			G_nodes_removed.append(random_node_1)

	return (new_GC,percentage_in_modular,actual_nodes_removed)








N = 1000
k = 3

#edgesPercentage = 0.15


#edgesToAdd = int(N * edgesPercentage)

Gnx_1 = nx.erdos_renyi_graph(N, k/(N-1), seed = 1223)

Gnx_2 = nx.erdos_renyi_graph(N, k/(N-1), seed = 345)

Gnk_1 = nk.nxadapter.nx2nk(Gnx_1)

Gnk_2 = nk.nxadapter.nx2nk(Gnx_2)

change_nodes(Gnk_1, Gnk_2)

edge_perc_to_connect = 0.1

percentage_to_attack = 0.05

(new_GC_ABA,percentage_in_modular_ABA,actual_nodes_removed_ABA) = get_optimal_set(Gnk_1,edge_perc_to_connect,percentage_to_attack,"ABA")
#(new_GC_ADA,percentage_in_modular_ADA,actual_nodes_removed_ADA) = get_optimal_set(Gnk_1,edge_perc_to_connect,percentage_to_attack,"ADA")

print(new_GC_ABA,percentage_in_modular_ABA,actual_nodes_removed_ABA)

print(get_GC(Gnk_1))





"""



(percolation_threshold_list, intersection_list) = changing_edge_percentages(Gnk_1)

print(percolation_threshold_list)

for i in intersection_list:
	print(len(i))


"""

"""
step_size = 0.01

percentage_to_attack = 0.3

(gc_List1,gc1_List1,gc2_List1,GC_nodes_List1, sgc_List1) = percolation(Gnk_1, step_size, "ABA", percentage_to_attack)
(gc_List2,gc1_List2,gc2_List2,GC_nodes_List2, sgc_List2) = percolation(Gnk_1, step_size, "ADA", percentage_to_attack)

intersect = intersection(GC_nodes_List1,GC_nodes_List2)

#print(len(GC_nodes_List1))
#print(len(GC_nodes_List2))

print(len(intersect))

#print(intersect)

print(gc_List1)
print(gc_List2)

(GC_init,GC_final) = check_GC(Gnk_1,intersect)

print(GC_init)
print(GC_final)


print(get_percolation_threshold(sgc_List1))
print(get_percolation_threshold(sgc_List2))

#print(sgc_List1)
#print(sgc_List2)
"""

