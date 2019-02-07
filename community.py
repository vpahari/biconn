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

	#print("between")

	#print(listToRemove)

	for n in listToRemove:
		G.removeNode(n)

	return listToRemove


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
	return index(max(sgcList))


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
		
		connected_comps_sizes = comp.getComponentSizes()

		connected_comps = comp.getComponents()

		GC_size = connected_comps_sizes[0]
		SGC_size = connected_comps_sizes[1]

		GC_nodes = connected_comps[0]

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

	nTC1 = random.sample(GC1_nodes, numEdges)
	nTC2 = random.sample(GC2_nodes, numEdges)

	for i in range(len(nTC1)):
		G.addEdge(nTC1[i], nTC2[i])


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

	GC_init = get_GC(G)

	for n in nodesToRemove:
		G.removeNode(n)

	GC_final = get_GC(G)

	return (GC_init,GC_final)



N = 2000
k = 3

edgesPercentage = 0.2


edgesToAdd = int(N * edgesPercentage)

Gnx_1 = nx.erdos_renyi_graph(N, k/(N-1), seed = 123)

Gnx_2 = nx.erdos_renyi_graph(N, k/(N-1), seed = 532)

Gnk_1 = nk.nxadapter.nx2nk(Gnx_1)

Gnk_2 = nk.nxadapter.nx2nk(Gnx_2)


change_nodes(Gnk_1, Gnk_2)


connect_random_nodes(Gnk_1,edgesToAdd)



step_size = 0.01

percentage_to_attack = 0.25

print("start")
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






