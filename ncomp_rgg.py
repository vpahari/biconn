#This is for investigating the bi-components present inside a connected component

import random
import networkx as nx
import sys
import math
from functools import reduce
import csv
from operator import itemgetter
import matplotlib.pyplot as plt

#import matplotlib.pyplot as plt
#import numpy as np

#import plotly.plotly as py

#effdfd

# fixed parameters
N=int(sys.argv[1]) # number of nodes
k=int(sys.argv[2]) # average degree
SEED=int(sys.argv[3])
#dim=int(sys.argv[3]) # rewiring prob

dim = 2

radius = math.sqrt(k / (math.pi * N))

step_size = 0.01 # every time, we remove step_size*N nodes from the network

random.seed(SEED)

def takeNodesOut(G, numNodesToRemove):
    G.remove_nodes_from(random.sample(list(G.nodes()),numNodesToRemove))
"""
numDifferentGraphs = 10
numSimsOfGraphs = 25


for net_rep in range(numDifferentGraphs):
    fixed_G = nx.erdos_renyi_graph(N, p, seed=SEED*(net_rep+1))
    assert fixed_G.number_of_nodes() == N
    if abs(fixed_G.number_of_edges()-k*N/2.0)>0.01*(k*N/2.0):
        continue
    for sim_rep in range(numSimsOfGraphs):

        G = fixed_G.copy()
        f = 0
        counter = 0

        while G.number_of_nodes() > 0:

        	assert G.number_of_nodes() == (N - counter*int(step_size*N))

        	conn = list(nx.connected_component_subgraphs(G))
        	GC = max(conn, key=len)

        	biConn = list(nx.biconnected_component_subgraphs(G))
        	biConnInConn = list(nx.biconnected_component_subgraphs(Gc))

        	takeNodesOut(G,int(step_size*N))
"""

def indexToTake(graphList,index):
    gList = list(map(lambda x : x[index], graphList))

    gAvg = reduce(lambda x,y : x+y, gList)

    gAvg = gAvg / len(gList)

    gStdDevList = list(map(lambda x : ((x - gAvg)**2),gList))

    gStdDev = math.sqrt((reduce(lambda x,y: x+y,gStdDevList)) / len(gList))

    return (gAvg,gStdDev)



numDifferentGraphs = 10
numSimsOfGraphs = 25

def getNO(G):
    counter = 0
    for item in G:
        if len(item) == 1:
            counter = counter + 1
    return counter



def getN1(G):
    counter = 0
    newGraph = []
    for item in G:
        if len(item) > 1:
            counter = counter + 1
    return counter


def getBN_1(G):
    counter = 0
    for item in G:
        if len(item) == 2:
            counter = counter + 1
    return counter


def getBN_2(G):
    counter = 0
    for item in G:
        if len(item) > 2:
            counter = counter + 1
    return counter

def getBiCompInConnComp(G):
    counter = 0
    for item in G:
        biConn = list(nx.biconnected_component_subgraphs(item))
        counter = counter + len(biConn)
    return counter


bc = []
bc2 = []
c = []
bcinc = []

lbc = []
lc = []

N_0L = []
N_1L = []

NB_1L = []
NB_2L = []


numDifferentGraphs = 10
numSimsOfGraphs = 25

for net_rep in range(numDifferentGraphs):
    fixed_G = nx.random_geometric_graph(N, radius, dim)
    assert fixed_G.number_of_nodes() == N

    for sim_rep in range(numSimsOfGraphs):

        f=0
        counter = 0

        bcList = []
        bc2List = []
        cList = []
        bcincList = []

        lbcList = []
        lcList = []

        N_0List = []
        N_1List = []

        NB_1List = []
        NB_2List = []

        G = fixed_G.copy()


        while G.number_of_nodes() > 0:


            assert G.number_of_nodes() == (N - counter*int(step_size*N))

            conn = list(nx.connected_component_subgraphs(G))
            if len(conn) == 0:
                GC = nx.empty_graph(0)
                GCLen = 0
            else:
                GC = max(conn, key=len)
                GCLen = len(GC)

            biConn = list(nx.biconnected_component_subgraphs(G))

            biConnInConn = list(nx.biconnected_component_subgraphs(GC))


            N_0 = getNO(conn)
            N_1 = getN1(conn)

            biConn2 = getBiCompInConnComp(conn)

            NB_1 = getBN_1(biConn)
            NB_2 = getBN_2(biConn)


            #NB_0 = getNOBiconn(biConn)

            if len(biConnInConn) == 0:
                GBCLen = 0
            else:
                GBCLen =  len(max(biConnInConn, key=len))


            cList.append(len(conn))
            bcList.append(len(biConn))
            bcincList.append(len(biConnInConn))
            bc2List.append(biConn2)
            lbcList.append(GBCLen)
            lcList.append(GCLen)

            N_0List.append(N_0)
            N_1List.append(N_1)

            NB_1List.append(NB_1)
            NB_2List.append(NB_2)

            #f1.append(f)

            #print(str(counter) + "\t" + str(len(biConn)) + "\t" + str(len(biConnInConn)))

            #print(len(biConnInConn))

            takeNodesOut(G,int(step_size*N))

            counter += 1
            f = f + step_size

        bc.append(bcList)
        bc2.append(bc2List)
        c.append(cList)
        bcinc.append(bcincList)

        lbc.append(lbcList)
        lc.append(lcList)

        N_0L.append(N_0List)
        N_1L.append(N_1List)

        NB_1L.append(NB_1List)
        NB_2L.append(NB_2List)




fractions = 0

counter = 0

finalList = []

while (fractions <= 1.0):
    (avgCList,stdCList) = indexToTake(c, counter)
    (avgBCList,stdBCList) = indexToTake(bc, counter)
    (avgBCInCList,stdBCInCList) = indexToTake(bcinc, counter)

    (avgGC, stdGC) = indexToTake(lc, counter)
    (avgGBC, stdGBC) = indexToTake(lbc,counter)

    (avgN_0,stdN_0) = indexToTake(N_0L,counter)
    (avgN_1,stdN_1) = indexToTake(N_1L,counter)

    (avgNB_1,stdNB_1) = indexToTake(NB_1L,counter)
    (avgNB_2,stdNB_2) = indexToTake(NB_2L,counter)

    finalList.append((fractions,avgCList,stdCList,avgBCList,stdBCList,avgBCInCList,stdBCInCList,avgGC,stdGC,avgGBC,stdGBC,avgN_0,stdN_0,avgN_1,stdN_1,avgNB_1,stdNB_1,avgNB_2,stdNB_2))

    fractions = fractions + step_size
    counter = counter + 1


output_file_name = "resultRGG_NC_N_%d_k_%d_radius_%g_dim_%d.csv"%(N,k,radius,dim)

fh = open(output_file_name, 'w')
writer = csv.writer(fh)

writer.writerow(["f","NC avg", "NC std","NBC avg","NBC std","NBINC avg","NBCINC std","GC avg","GC std","GBC avg", "GBC std", "N_0 avg", "N_0 std", "N_1 avg", "N_1 std", "NB_1 avg", "NB_1 std", "NB_2 avg", "NB_2 std" ])

for (fractions,avgCList,stdCList,avgBCList,stdBCList,avgBCInCList,stdBCInCList,avgGC,stdGC,avgGBC,stdGBC,avgN_0,stdN_0,avgN_1,stdN_1,avgNB_1,stdNB_1,avgNB_2,stdNB_2) in finalList:
    writer.writerow([fractions,avgCList,stdCList,avgBCList,stdBCList,avgBCInCList,stdBCInCList,avgGC,stdGC,avgGBC,stdGBC,avgN_0,stdN_0,avgN_1,stdN_1,avgNB_1,stdNB_1,avgNB_2,stdNB_2])



fh.close()















