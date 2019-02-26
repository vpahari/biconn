import networkx as nx
import networkit as nk
import random
import sys
import math
from functools import reduce
import csv
from operator import itemgetter
import matplotlib.pyplot as plt
import pickle





#N=int(sys.argv[1]) # number of nodes
#k=int(sys.argv[2])
#SEED=int(sys.argv[3])
#M_out = int(sys.argv[4])
#r = float(sys.argv[5])





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



def random_removal(G,numNodesToRemove):

	all_nodes = G.nodes()

	listToRemove = random.sample(all_nodes, numNodesToRemove)

	for n in listToRemove:
		G.removeNode(n)

	return listToRemove


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

	GC1_nodes = [i for i in range(single_module_size)]

	GC2_nodes = [single_module_size + i for i in range(single_module_size)]

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


def connect_random_nodes_GC(G,numEdges):
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


def get_optimal_set(G_init, nodes_1,percentage_to_attack,typeOfAttack):

	G_size = G_init.numberOfNodes() 

	G_all_nodes = G_init.nodes()

	G = copy_graph(G_init)

	num_nodes_to_remove = int(percentage_to_attack * G_size)

	if typeOfAttack == "ABA":
		nodes_removed = adaptive_betweenness(G,num_nodes_to_remove)

	elif typeOfAttack == "ADA":
		nodes_removed = adaptive_degree(G,num_nodes_to_remove)

	elif typeOfAttack == "RAN":
		nodes_removed = random_removal(G,num_nodes_to_remove)

	curr_GC = get_GC(G)

	percentage_in_modular = get_percentage(nodes_removed,nodes_1)

	actual_nodes_removed = nodes_removed.copy()

	#print(actual_nodes_removed)

	G_nodes_removed = remove_nodes_from_list(G_all_nodes,nodes_removed)

	counter = 0

	time_stamp = [curr_GC]

	print(G_size)

	while counter < (G_size * 10):

		print(counter)

		#print(counter)

		G = copy_graph(G_init)

		#print(G.numberOfNodes())
		#print(G.numberOfEdges())

		#add assertion

		random_node_1 = random.choice(nodes_removed)

		random_node_2 = random.choice(G_nodes_removed)
 
		new_nodes_to_remove = nodes_removed.copy()

		new_nodes_to_remove.append(random_node_2)

		#print(new_nodes_to_remove)

		#print(nodes_removed)

		new_nodes_to_remove.remove(random_node_1)

		new_GC = check_GC(G,new_nodes_to_remove)

		time_stamp.append(new_GC)

		if new_GC < curr_GC:

			#print(curr_GC,new_GC)

			counter = 0

			curr_GC = new_GC

			percentage_in_modular = get_percentage(new_nodes_to_remove,nodes_1)

			actual_nodes_removed = new_nodes_to_remove.copy()

			nodes_removed = new_nodes_to_remove.copy()

			G_nodes_removed.remove(random_node_2)

			G_nodes_removed.append(random_node_1)

		else:

			counter += 1

	return (new_GC,percentage_in_modular,actual_nodes_removed,time_stamp)





def changing_percentages_attack(G,edge_percentage,step_size,max_to_attack,typeOfAttack):

	counter = 0.01

	copy_G = copy_graph(G)

	G_size = copy_G.numberOfNodes() 

	print(copy_G.numberOfEdges())

	edges_to_add = int(edge_percentage * G_size / 2)

	connections = list(connect_random_nodes(copy_G,edges_to_add))

	nodes_1 = set(list(map(lambda x : x[0],connections)) + list(map(lambda x : x[1],connections)))

	print(copy_G.numberOfEdges())

	GC_List = []
	percentage_in_modular_List = []
	actual_nodes_removed_List = []

	time_stamp_List = []

	while counter <= max_to_attack:

		num_nodes_to_remove = int(counter * G_size)

		if typeOfAttack == "RAN":

			(new_GC,percentage_in_modular,actual_nodes_removed,time_stamp) = get_optimal_set(copy_G,nodes_1,counter,"RAN")

		elif typeOfAttack == "ABA":
			(new_GC,percentage_in_modular,actual_nodes_removed,time_stamp) = get_optimal_set(copy_G,nodes_1,counter,"ABA")

		elif typeOfAttack == "ADA":
			(new_GC,percentage_in_modular,actual_nodes_removed,time_stamp) = get_optimal_set(copy_G,nodes_1,counter,"ADA")

		GC_List.append(new_GC)

		percentage_in_modular_List.append(percentage_in_modular)

		actual_nodes_removed_List.append(actual_nodes_removed)

		time_stamp_List.append(time_stamp)

		counter += step_size

	#return (GC_ABA_List,GC_ADA_List,percentage_in_modular_ABA_List,percentage_in_modular_ADA_List,actual_nodes_removed_ABA_List,actual_nodes_removed_ADA_List)
	return (GC_List,percentage_in_modular_List,actual_nodes_removed_List,time_stamp_List)



def create_new_List(l):
	new_list = []
	for i in l:
		new_list.append(i)

	return new_list


def changing_percentages_edges(G,max_edge_percentage,step_size,typeOfAttack):

	max_to_attack = 0.02

	counter = 0.05

	step_size_for_attack = 0.01

	GC_dict = {}

	actual_nodes_removed_dict = {}

	percentage_in_modular_dict = {}

	time_stamp_dict = {}


	while counter <= max_edge_percentage:

		(GC_List,percentage_in_modular_List,actual_nodes_removed_List,time_stamp_List) = changing_percentages_attack(G,counter,step_size_for_attack,max_to_attack,typeOfAttack)

		GC_dict[counter] = GC_List

		percentage_in_modular_dict[counter] = percentage_in_modular_List

		actual_nodes_removed_dict[counter] = actual_nodes_removed_List

		time_stamp_dict[counter] = time_stamp_List

		counter += step_size


	return (GC_dict,percentage_in_modular_dict,actual_nodes_removed_dict,time_stamp_dict)



def getIntersectionList(l1,l2):

	intersection_list = []


	for c in range(len(l1)):
		i = l1[c]
		j = l2[c]
		new_list = intersection(i,j)
		intersection_list.append(len(new_list) / len(i))

	return intersection_list



def change_dict_to_list(time_stamp_dict):

	keys = list(time_stamp_dict.keys())

	new_list = []

	for k in keys:

		curr_list = time_stamp_dict[k]

		new_list.append(curr_list)






def plot_time_stamps(time_stamp_dict):

	keys = list(time_stamp_dict.keys())

	for k in keys:

		ts_list = time_stamp_dict[k]

		label_str = "edge percentage : " + str(k) + ".png"

		for ts in ts_list:

			x_axis_list = [i+1 for i in range(len(ts))]

			print(ts)
			print(x_axis_list)

			print(len(ts))
			print(len(x_axis_list))

			plt.xlabel("time",fontsize = 20)
			plt.ylabel("GC",fontsize = 20)

			plt.title("timestamps", fontsize=20)

			plt.plot(x_axis_list,ts,label='GC with time stamp')


		plt.clf()

		plt.savefig(label_str)




def adaptive_connections_degree_attack(G,numNodesToRemove,nodes_to_remove_stepwise,connections):

	G_copy = copy_graph(G)

	nodes_connections = list(set(list(map(lambda x : x[0],connections)) + list(map(lambda x : x[1],connections))))

	nodes_connections_degree = list(map(lambda x : (x,G_copy.degree(x)),nodes_connections))

	random.shuffle(nodes_connections_degree)

	sorted_list = sorted(nodes_connections_degree, key = itemgetter(1), reverse = True)

	counter = 0

	GC_List = []

	count = 0

	for i in range(numNodesToRemove):

		if i % nodes_to_remove_stepwise == 0:
			print(i)
			print(count)
			GC_List.append(get_GC(G_copy))
			count += 1

		node_to_remove = sorted_list[0][0]

		#print(sorted_list)

		#print(node_to_remove)

		G_copy.removeNode(node_to_remove)

		nodes_connections.remove(node_to_remove)

		if len(nodes_connections) == 0:
			break

		nodes_connections_degree = list(map(lambda x : (x,G_copy.degree(x)),nodes_connections))

		sorted_list = sorted(nodes_connections_degree, key = itemgetter(1), reverse = True)


	GC_List.append(get_GC(G_copy))

	return GC_List



def ADA_attack(G,num_nodes_to_remove,nodes_to_remove_stepwise):

	G_copy = copy_graph(G)

	GC_List = []

	print(num_nodes_to_remove % nodes_to_remove_stepwise)

	assert(num_nodes_to_remove % nodes_to_remove_stepwise == 0)

	for i in range(num_nodes_to_remove):

		if i % nodes_to_remove_stepwise == 0:
			GC_List.append(get_GC(G_copy))

		degree = nk.centrality.DegreeCentrality(G_copy)

		degree.run()

		degree_sequence = degree.ranking()

		random.shuffle(degree_sequence)

		degree_sequence.sort(key = itemgetter(1), reverse = True)

		node_to_remove = degree_sequence[0][0]

		G_copy.removeNode(node_to_remove)

		print(degree_sequence[0])

	GC_List.append(get_GC(G_copy))

	return GC_List



def ADA_ADCA_attack(G,edge_percentage,perc_nodes_to_remove,num_sims):

	GC_List_ADA = []
	GC_List_ADCA = []

	G_size = G.numberOfNodes() 

	edges_to_add = int(edge_percentage * (G_size / 2))

	num_nodes_to_remove = int(perc_nodes_to_remove * G_size)

	print(num_nodes_to_remove)

	for i in range(num_sims):

		G_copy = copy_graph(G)

		connections = connect_random_nodes(G_copy,edges_to_add)

		#print(connections)

		print(G_copy.numberOfEdges())

		print(get_GC(G_copy))

		curr_GC_ADA = ADA_attack(G_copy,num_nodes_to_remove)

		curr_GC_ADCA = adaptive_connections_degree_attack(G_copy,num_nodes_to_remove,connections)

		GC_List_ADA.append(curr_GC_ADA)

		GC_List_ADCA.append(curr_GC_ADCA)

	return (GC_List_ADA,GC_List_ADCA)



def ADA_ADCA_attack_full(G,edge_percentage,num_sims,step_size):

	GC_List_ADA = []
	GC_List_ADCA = []

	G_size = G.numberOfNodes() 

	edges_to_add = int(edge_percentage * (G_size / 2))

	num_nodes_to_remove = int(edge_percentage * G_size)

	nodes_to_remove_stepwise = int(step_size * G_size)

	for i in range(num_sims):

		G_copy = copy_graph(G)

		connections = connect_random_nodes(G_copy,edges_to_add)

		curr_GC_ADA = ADA_attack(G_copy,num_nodes_to_remove,nodes_to_remove_stepwise)

		curr_GC_ADCA = adaptive_connections_degree_attack(G_copy,num_nodes_to_remove,nodes_to_remove_stepwise,connections)

		GC_List_ADA.append(curr_GC_ADA)

		GC_List_ADCA.append(curr_GC_ADCA)

	return (GC_List_ADA,GC_List_ADCA)	




def convert_lists(l):

	len_of_list = len(l[0])

	final_avg_list = []

	for i in range(len_of_list):

		avg_list = list(map(lambda x : x[i], l))

		avg = sum(avg_list) / len(avg_list)

		final_avg_list.append(avg)

	return final_avg_list





#def fixed_percentage_attack(G_init, typeOfAttack, edges_percentage, attack_percentage):



N=int(sys.argv[1]) # number of nodes
k=int(sys.argv[2]) # average degree
SEED1=int(sys.argv[3])
SEED2 = int(sys.argv[4])
edge_perc = float(sys.argv[5])
#nodes_to_remove = float(sys.argv[6])


"""
attack_type = str(sys.argv[5])
step_size = float(sys.argv[6])
max_edge_percentage = float(sys.argv[7])

#step_size = 0.05
#max_edge_percentage = 0.9
"""


Gnx_1 = nx.erdos_renyi_graph(N, k/(N-1), seed = SEED1)

Gnx_2 = nx.erdos_renyi_graph(N, k/(N-1), seed = SEED2)

Gnk_1 = nk.nxadapter.nx2nk(Gnx_1)

Gnk_2 = nk.nxadapter.nx2nk(Gnx_2)

change_nodes(Gnk_1, Gnk_2)

num_sims = 3

(GC_List_ADA,GC_List_ADCA) = ADA_ADCA_attack_full(Gnk_1, edge_perc, num_sims,0.01)


print(GC_List_ADA)

print(GC_List_ADCA)

for i in GC_List_ADA:
	print(len(i))

for i in GC_List_ADCA:
	print(len(i))


"""
filename_ADA = 'ADA_SEED1_N_' + str(N) + "_k_" + str(k) + "_SEED1_" + str(SEED1) + "_SEED2_" + str(SEED2) + "_edgeperc_" + str(edge_perc) + "_nodestoremove_" + str(nodes_to_remove) + '.pickle'

filename_ADCA = 'ADCA_SEED1_N_' + str(N) + "_k_" + str(k) + "_SEED1_" + str(SEED1) + "_SEED2_" + str(SEED2) + "_edgeperc_" + str(edge_perc) + "_nodestoremove_" + str(nodes_to_remove) + '.pickle'


with open(filename_ADA,'wb') as handle:
	pickle.dump(GC_List_ADA, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(filename_ADCA,'wb') as handle:
	pickle.dump(GC_List_ADCA, handle, protocol=pickle.HIGHEST_PROTOCOL)
"""








#(GC_ABA_List,GC_ADA_List,percentage_in_modular_ABA_List,percentage_in_modular_ADA_List,actual_nodes_removed_ABA_List,actual_nodes_removed_ADA_List) = changing_percentages_attack(Gnk_1,edge_perc_to_connect,step_size,max_to_attack)

#print(GC_ABA_List)
#print(GC_ADA_List)

#print(percentage_in_modular_ABA_List)
#print(percentage_in_modular_ADA_List)

"""
(GC_dict,percentage_in_modular_dict,actual_nodes_removed_dict,time_stamp_dict) = changing_percentages_edges(Gnk_1,max_edge_percentage,step_size,attack_type)

connect_random_nodes(Gnk_1, int(Gnk_1.numberOfNodes() * 0.05))

print(GC_dict)
print(percentage_in_modular_dict)
print(time_stamp_dict)

for k in time_stamp_dict.keys():
	print(len(time_stamp_dict[k]))


#plot_time_stamps(time_stamp_dict)

filename = 'edge_perc_dict_SEED1_N_' + str(N) + "_k_" + str(k) + "_SEED1_" + str(SEED1) + "_SEED2_" + str(SEED2) + "_attack_type_" + attack_type + "_max_edge_percentage_" + str(max_edge_percentage) + '.pickle'


with open(filename,'wb') as handle:
	pickle.dump(time_stamp_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""
































