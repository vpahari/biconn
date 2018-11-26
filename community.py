import networkx as nx
#import networkit as nk
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

N=int(sys.argv[1]) # number of nodes
k=int(sys.argv[2]) # average degree
SEED=int(sys.argv[3])
alpha = float(sys.argv[4])
r = float(sys.argv[5])

"""
N = 10000

k = 4

SEED = 312

alpha = 0.01

r = 0.1
"""

numModules = 2

M = N * k

M_out = M * alpha

M_in = M * (1 - alpha)

k_in = (M_in * 2) / (numModules * N)

p_in = k_in / (N-1)


print(M_in)
print(k_in)
print(p_in)


def connect(G,nTC1,nTC2,M_out):
	newSet = set([])
	counter = 0
	while counter < M_out:
		i = random.choice(nTC1)
		j = random.choice(nTC2)
		if (i,j) not in newSet:
			G.add_edge(i,j)
			newSet.add((i,j))
			counter += 1
			#print(i,j)


def indexToTake(graphList,index):
	#print(index)

	gList = list(map(lambda x : x[index], graphList))

	gAvg = reduce(lambda x,y : x+y, gList)

	gAvg = gAvg / len(gList)

	gStdDevList = list(map(lambda x : ((x - gAvg)**2),gList))

	gStdDev = math.sqrt((reduce(lambda x,y: x+y,gStdDevList)) / len(gList))

	return (gAvg,gStdDev)


def getSecondMax(conn):
	if len(conn) < 2:
		return 0

	newConn = conn.copy()

	sortedConn = sorted(newConn, key = len,reverse = True)

	secondBiggest = sortedConn[1]

	"""
	counter1 = 0
	counter2 = 0

	for node in list(secondBiggest.nodes()):
		if node < N:
			counter1 += 1
		else:
			counter2 += 1


	"""


	return len(sortedConn[1])

def createOrder(G):
	allNodes = list(G.nodes())
	degreeDict = dict(G.degree(allNodes))
	degreeDictItems = list(degreeDict.items())
	random.shuffle(degreeDictItems)
	degreeDictItemsSorted = sorted(degreeDictItems, key = itemgetter(1),reverse = True)
	onlyNodes = list(map(lambda x:x[0],degreeDictItemsSorted))
	return onlyNodes


def degree_removal(G, numNodesToRemove, remove_order, counter):
	startIndex = int(numNodesToRemove * counter)
	endIndex = int(startIndex + numNodesToRemove)

	if endIndex > len(remove_order):
		G.remove_nodes_from(random.sample(list(G.nodes()),int(numNodesToRemove)))
		return

	#if endIndex > len(remove_order):
	#	nodesToRemove = remove_order[startIndex:]
	#else:
	nodesToRemove = remove_order[startIndex:endIndex]

	G.remove_nodes_from(nodesToRemove)

def coreHD(G, numNodesToRemove, found):
	if found:
		G.remove_nodes_from(random.sample(list(G.nodes()),int(numNodesToRemove)))
		return True

	G_2core = nx.k_core(G,2)
	allNodes = list(G_2core.nodes())
	degreeDict = dict(G_2core.degree(allNodes))
	degreeDictItems = list(degreeDict.items())
	random.shuffle(degreeDictItems)
	degreeDictItemsSorted = sorted(degreeDictItems, key = itemgetter(1),reverse = True)
	onlyNodes = list(map(lambda x:x[0],degreeDictItemsSorted))

	if len(onlyNodes) >= numNodesToRemove:
		print("core")
		nodesToRemove = onlyNodes[:numNodesToRemove]
		G.remove_nodes_from(nodesToRemove)
		return False

	else:
		G.remove_nodes_from(random.sample(list(G.nodes()),int(numNodesToRemove)))
		return True




def getSize(GC_nodes,GBC_nodes):
	counterGC1 = 0
	counterGC2 = 0

	counterGBC1 = 0
	counterGBC2 = 0

	for node in GC_nodes:

		if node < N:
			counterGC1 += 1

		else:
			counterGC2 += 1


	for node in GBC_nodes:

		if node < N:
			counterGBC1 += 1

		else:
			counterGBC2 += 1

	return (counterGC1,counterGC2,counterGBC1,counterGBC2)



step_size = 0.01

gc1 = []
gbc1 = []

gc2 = []
gbc2 = []

gc = []
gbc = []

sgc1 = []
sgbc1 = []

sgc2 = []
sgbc2 = []

sgc = []
sgbc = []


numDifferentGraphs = 5

numSimsOfGraphs = 1



for net_rep in range(numDifferentGraphs):

	found = False

	G = nx.erdos_renyi_graph(N, p_in, seed = (SEED*2*net_rep)+2)

	G2 = nx.erdos_renyi_graph(N, p_in, seed = (SEED*net_rep)+1)

	print(G.number_of_edges())
	print(G2.number_of_edges())

	print(G.number_of_edges() + G2.number_of_edges())

	l = [i for i in range(N, 2*N)]

	G.add_nodes_from(l)

	allEdges = list(G2.edges())

	allEdges = map(lambda x : (N + x[0], N + x[1]) ,allEdges)

	G.add_edges_from(allEdges)

	allNodes = list(G.nodes())

	G.edges()

	nodes1 = allNodes[:N]
	nodes2 = allNodes[N:]


	numNodesToConnect = int(r * N)


	#nodes to connect
	nTC1 = random.sample(nodes1, numNodesToConnect)
	nTC2 = random.sample(nodes2, numNodesToConnect)


	connect(G,nTC1,nTC2,M_out)

	#G_2core = nx.k_core(G,2)


	#print(G.number_of_nodes())
	#print(G_2core.number_of_nodes())


	f=0
	counter = 0

	gc1List = []
	gbc1List = []

	gc2List = []
	gbc2List = []

	gcList = []
	gbcList = []

	sgc1List = []
	sgbc1List = []

	sgc2List = []
	sgbc2List = []

	sgcList = []
	sgbcList = []

	
	#G = G_2core.copy()

	numNodesToRemove = int(2 * N * step_size)

	print(numNodesToRemove)

	#removeOrder = createOrder(G_2core)

	while counter <= 100:


		print(G.number_of_nodes())

		if G.number_of_nodes() <= numNodesToRemove:
			break

		#assert G.number_of_nodes() == (N - counter*int(step_size*N))

		conn = list(nx.connected_component_subgraphs(G))
		biConn = list(nx.biconnected_component_subgraphs(G))

		if len(conn) == 0:
			GC = nx.empty_graph(0)
		else:
			GC = max(conn, key=len)

		if len(biConn) == 0:
			GBC = 0
		else:
			GBC =  max(biConn, key=len)


		GC_nodes = list(GC.nodes())
		GBC_nodes = list(GBC.nodes())

		(counterGC1,counterGC2,counterGBC1,counterGBC2) = getSize(GC_nodes,GBC_nodes)

		secondGC = getSecondMax(conn)
		secondGBC = getSecondMax(biConn)


		gcList.append(GC.number_of_nodes())
		gbcList.append(GBC.number_of_nodes())

		sgcList.append(secondGC)
		sgbcList.append(secondGBC)

		gc1List.append(counterGC1)
		gbc1List.append(counterGBC1)

		gc2List.append(counterGC2)
		gbc2List.append(counterGBC2)

		found = coreHD(G, numNodesToRemove, found)

		counter += 1
		f = f + step_size


	gbc.append(gbcList)
	gc.append(gcList)

	sgc.append(sgcList)
	sgbc.append(sgbcList)

	gc1.append(gc1List)
	gbc1.append(gbc1List)

	gc2.append(gc2List)
	gbc2.append(gbc2List)



fractions = 0

counter = 0

finalList = []



while (fractions < 0.99):
	print(counter)
	(avgGc,stdGc) = indexToTake(gc, counter)
	(avgGBc,stdGBc) = indexToTake(gbc, counter)

	(avgGc1,stdGc1) = indexToTake(gc1, counter)
	(avgGBc1,stdGBc1) = indexToTake(gbc1, counter)

	(avgGc2,stdGc2) = indexToTake(gc2, counter)
	(avgGBc2,stdGBc2) = indexToTake(gbc2, counter)

	(avgSgc,stdSgc) = indexToTake(sgc, counter)
	(avgSgbc,stdSgbc) = indexToTake(sgbc, counter)

	finalList.append((fractions,avgGc,stdGc,avgGBc,stdGBc,avgGc1,stdGc1,avgGc2,stdGc2,avgGBc1,stdGBc1,avgGBc2,stdGBc2,avgSgc,stdSgc,avgSgbc,stdSgbc))

	fractions = fractions + step_size
	counter = counter + 1


output_file_name = "community_N_%d_k_%d_alpha_%g_r_%g_SEED_%d.csv"%(N,k,alpha,r,SEED)

fh = open(output_file_name, 'w')
writer = csv.writer(fh)

writer.writerow(["f","GC avg", "GC std","GBC avg","GBC std","GC1 avg","GC1 std","GC2 avg", "GC2 std","GBC1 avg","GBC1 std","GBC2 avg", "GBC2 std", "SGC avg", "SGC std", "SGBC avg", "SGBC std" ])

for (fractions,avgGc,stdGc,avgGBc,stdGBc,avgGc1,stdGc1,avgGc2,stdGc2,avgGBc1,stdGBc1,avgGBc2,stdGBc2,avgSgc,stdSgc,avgSgbc,stdSgbc) in finalList:
	writer.writerow([fractions,avgGc,stdGc,avgGBc,stdGBc,avgGc1,stdGc1,avgGc2,stdGc2,avgGBc1,stdGBc1,avgGBc2,stdGBc2,avgSgc,stdSgc,avgSgbc,stdSgbc])



fh.close()



