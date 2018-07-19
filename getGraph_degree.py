import networkx as nx
import osmnx as ox
import random
import math
from operator import itemgetter
import csv
from functools import reduce
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def createOrder(G):
    allNodes = list(G.nodes())
    degreeDict = dict(G.degree(allNodes))
    degreeDictItems = list(degreeDict.items())
    degreeDictItemsSorted = sorted(degreeDictItems, key = itemgetter(1),reverse = True)
    onlyNodes = list(map(lambda x:x[0],degreeDictItemsSorted))
    return onlyNodes

def getGraphs(G, fileToSave):
	N = G.number_of_nodes()
	numToRemove = int(N * 0.05)

	order = createOrder(G)

	counter = 0

	for i in range(18):
		print(i)

		nodesList = list(G.nodes())

		conn = list(nx.connected_component_subgraphs(G))
		biconn = list(nx.biconnected_component_subgraphs(G))

		GC = max(conn, key=len)
		GBC = max(biconn, key=len)

		GCnodes = list(GC.nodes())
		GBCnodes = list(GBC.nodes())

		try:
			fig, ax = ox.plot_graph(G)
		except:
			break

		for n in GCnodes:
			dictInfo = G.node[n]
			ax.scatter(dictInfo['x'], dictInfo['y'], c='green')

		GCName = fileToSave + "_Deg_GCPlot_%d.png" % (i) 
		GBCName = fileToSave + "_Deg_GBCPlot_%d.png" % (i) 

		plt.savefig(GCName)
		plt.clf()

		try:
			fig, ax = ox.plot_graph(G)
		except:
			break

		for n in GBCnodes:
			dictInfo = G.node[n]
			ax.scatter(dictInfo['x'], dictInfo['y'], c='red')

		plt.savefig(GBCName)
		plt.clf()

		startIndex = counter * numToRemove
		endIndex = startIndex + numToRemove

		nodesToRemove = order[startIndex : endIndex]

		G.remove_nodes_from(nodesToRemove)

		counter = counter + 1


G = nx.read_gpickle("NYU.gpickle")

getGraphs(G,"NY")






