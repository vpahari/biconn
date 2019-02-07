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
	listToRemove = between.ranking()[:nodesToRemove]

	for n in listToRemove:
		G.removeNode(n)


def adaptive_betweenness(G,numNodesToRemove):
	between = nk.centrality.DynBetweenness(G)
	between.run()
	listToRemove = between.ranking()[:nodesToRemove]

	for n in listToRemove:
		G.removeNode(n)

	


def getSize(GC_nodes):
	counterGC1 = 0
	counterGC2 = 0

	for node in GC_nodes:

		if node < N:
			counterGC1 += 1

		else:
			counterGC2 += 1


	return (counterGC1,counterGC2)



def get_percolation_threshold(sgc_List):
	return index(max(sgcList))


def percolation(G, step_size, typeOfAttack, percentage_to_attack):
	counter = 0

	gc1_List = []
	gc2_List = []

	gc_List = []

	sgc_List = []

	GC_nodes_List = []

	originalSize = G.numberOfNodes()

	numNodesToRemove = originalSize * step_size

	while counter < percentage_to_attack:

		comp = nk.components.DynConnectedComponents(G)
		comp.run()
		
		connected_comps_sizes = comp.getComponentSizes()

		connected_comps = comp.getComponents()

		GC_size = connected_comps_sizes[0]
		SGC_size = connected_comps_sizes[1]

		GC_nodes = connected_comps[0]

		(GC1_size, GC2_size) = getSize(GC_nodes)

		gc1_List.append(GC1_size)
		gc2_List.append(GC2_size)

		gc_List.append(GC_size)
		sgc_List.append(SGC_size)

		GC_nodes_List += GC_nodes

		if typeOfAttack == "ABA":
			adaptive_betweenness(G,numNodesToRemove)

		elif typeOfAttack == "ADA":
			adaptive_degree(G,numNodesToRemove)


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

	print(connected_comps)

	GC1_nodes = connected_comps[0]
	GC2_nodes = connected_comps[1]



	nTC1 = random.sample(GC1_nodes, numEdges)
	nTC2 = random.sample(GC2_nodes, numEdges)

	for i in range(len(nTC1)):
		G.addEdge(nTC1[i], nTC2[i])



Gnx_1 = nx.erdos_renyi_graph(1000, 3/999, seed = 4123)

Gnx_2 = nx.erdos_renyi_graph(1000, 3/999, seed = 41232)

Gnk_1 = nk.nxadapter.nx2nk(Gnx_1)

Gnk_2 = nk.nxadapter.nx2nk(Gnx_2)

print(Gnk_1.numberOfNodes())
print(Gnk_2.numberOfNodes())

print(Gnk_1.numberOfEdges())
print(Gnk_2.numberOfEdges())

change_nodes(Gnk_1, Gnk_2)

print(Gnk_1.numberOfNodes())
print(Gnk_1.numberOfEdges())

connect_random_nodes(Gnk_1,100)



