import networkx as nx
import networkit as nk
from operator import itemgetter
import random
import pickle
import sys
import igraph as ig


#import matplotlib.pyplot as plt

N=int(sys.argv[1]) # number of nodes
k=int(sys.argv[2])
SEED=int(sys.argv[3])
alpha = float(sys.argv[4])
percentageSample = float(sys.argv[5])


#N = 1000

#k = 4.0

#SEED = 1099 

p = k / float(N-1)

step_size = 0.01

#random.seed(SEED)


def findAllDisjointPaths(G,s,t,shortestPath):
	distPaths = []

	distPaths.append(shortestPath)

	if len(shortestPath) == 2:
		G.removeEdge(s,t)
		G.removeEdge(t,s)

	else:
		nodesToRemove = shortestPath[1:len(shortestPath)-1]

		for node in nodesToRemove:
			G.removeNode(node)


	newDist = nk.distance.Dijkstra(G, s, True, False)

	newDist.run()

	hasPath = newDist.numberOfPaths(t)

	while hasPath != 0:
		
		shortestPath = newDist.getPath(t)

		for i in shortestPath[1:len(shortestPath) - 1]:
			G.removeNode(i)

		
		distPaths.append(shortestPath)
		
		newDist = nk.distance.Dijkstra(G, s, True, True)
		
		newDist.run()
		
		hasPath = newDist.numberOfPaths(t)

	#print(distPaths)
		
	return distPaths

def makeSFGraph(N,k,exp_out):
	num_edges = int((N * k) / 2)

	igG = ig.Graph.Static_Power_Law(N,num_edges,exp_out)

	allEdges = igG.get_edgelist()

	fixed_G = nx.Graph()

	listOfNodes = [i for i in range(N)]

	fixed_G.add_nodes_from(listOfNodes)

	fixed_G.add_edges_from(allEdges)

	return fixed_G



def takeOutNodes(G,sortedList):
	numNodesToRemove = int(step_size * N)
	listToRemove = sortedList[:numNodesToRemove]
	print(listToRemove)
	for (node, num) in listToRemove:
		G.removeNode(node)


def findLargestNodes(uniqueNodes):
	keysValues = uniqueNodes.items()
	sortedKeysValues = sorted(keysValues, key = itemgetter(1), reverse = True)
	return sortedKeysValues

number_of_sims = 10

for sim in range(number_of_sims):
	#GER = nx.read_gpickle("sageGraph.gpickle")

	newSEED = SEED*(sim+1)

	random.seed(newSEED)

	GER = makeSFGraph(N,k,alpha)

	#print(type(GER))

	#print(GER.edges())
	#print(GER.nodes())

	conn = list(nx.connected_component_subgraphs(GER))

	G = max(conn, key=len)

	print(len(G))

	#print(len(G))

	sizeOfNewList = G.number_of_nodes()

	Gnk = nk.nxadapter.nx2nk(G)

	newGraph = nk.graph.Graph(n = sizeOfNewList, weighted = False, directed = True)

	lastEdges = Gnk.edges()

	for (i,j) in lastEdges:
		newGraph.addEdge(i,j)
		newGraph.addEdge(j,i)

	#newEdges = newGraph.edges()

	listOfNodes = newGraph.nodes()

	#print(newGraph.isDirected())


	lengthOfNodes = len(listOfNodes)

	#print(newGraph.edges())

	num_Node_Sample = int(percentageSample * N)

	#print(num_Node_Sample)

	print(num_Node_Sample)
	print(len(listOfNodes))

	node_Sample = random.sample(listOfNodes, num_Node_Sample)

	degree = list(map(lambda x : newGraph.degree(x), node_Sample))

	#print(node_Sample)

	number_of_DP_List = []

	DP_List = []

	for s in node_Sample:

		number_of_DP_List_s = []

		DP_List_s = []

		dijk = nk.distance.Dijkstra(newGraph, s, True, False)
		
		dijk.run()

		for t in listOfNodes:

			if t == s:
				DP_List_s.append([])
				number_of_DP_List_s.append(0)
				continue
			
			isPath = dijk.numberOfPaths(t)
			
			tempG = newGraph.copyNodes()
			
			for (e1,e2) in newGraph.edges():
				tempG.addEdge(e1,e2)

			if isPath != 0:

				shortestPath = dijk.getPath(t)
				
				DP = findAllDisjointPaths(tempG,s,t, shortestPath)
				number_of_DP = len(DP)

				DP_List_s.append(DP)
				number_of_DP_List_s.append(number_of_DP)

		DP_List.append(DP_List_s)
		number_of_DP_List.append(number_of_DP_List_s)

	DP_List_string = "DPListSF_N_%d_k_%d_SEED_%d_alpha_%g_percentage_%g.pkl"%(N,k,newSEED,alpha,percentageSample)
	number_of_DP_List_string = "numberofDPSF_N_%d_k_%d_SEED_%d_alpha_%g_percentage_%g.pkl"%(N,k,newSEED,alpha,percentageSample)
	degree_string = "DegreeSF_N_%d_k_%d_SEED_%d_alpha_%g_percentage_%g.pkl"%(N,k,newSEED,alpha,percentageSample)

	with open(DP_List_string, 'wb') as f:
		pickle.dump(DP_List, f)

	with open(number_of_DP_List_string, 'wb') as f:
		pickle.dump(number_of_DP_List, f)

	with open(degree_string, 'wb') as f:
		pickle.dump(degree, f)


#print(len(listOfAllSP))
#a = findLargestNodes(uniqueNodesDict)
#print(len(a))
#print(a)


