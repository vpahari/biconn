import networkx as nx
import osmnx as ox
import random
import math
from operator import itemgetter
import csv
from functools import reduce
import matplotlib.pyplot as plt

plt.switch_backend('agg')

def getGraphs(G, fileToSave):
	N = G.number_of_nodes()
	numToRemove = int(N * 0.1)

	for i in range(10):
		print(i)

		nodesList = list(G.nodes())

		conn = list(nx.connected_component_subgraphs(G))
		biconn = list(nx.biconnected_component_subgraphs(G))

		GC = max(conn, key=len)
		GBC = max(biconn, key=len)

		GCnodes = list(GC.nodes())
		GBCnodes = list(GBC.nodes())

		try:
			fig, ax = ox.plot_graph(G,fig_height=30, fig_width=30)
		except:
			break

		for n in GCnodes:
			dictInfo = G.node[n]
			ax.scatter(dictInfo['x'], dictInfo['y'], c='green')

		GCName = fileToSave + "_GCPlot_%d.png" % (i) 
		GBCName = fileToSave + "_GBCPlot_%d.png" % (i) 

		plt.savefig(GCName)
		plt.clf()

		try:
			fig, ax = ox.plot_graph(G,fig_height=30, fig_width=30)
		except:
			break

		for n in GBCnodes:
			dictInfo = G.node[n]
			ax.scatter(dictInfo['x'], dictInfo['y'], c='red')

		plt.savefig(GBCName)
		plt.clf()

		nodesToRemove = random.sample(nodesList,numToRemove)

		G.remove_nodes_from(nodesToRemove)


G = nx.read_gpickle("BelgradeDriveU.gpickle")

getGraphs(G,"Belgrade")






