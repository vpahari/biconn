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



# fixed parameters
N=int(sys.argv[1]) # number of nodes
k=int(sys.argv[2])
SEED=int(sys.argv[3])

p = float(k / (N - 1))

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


def getAvg(conn):
    newConn = conn.copy()
    newGC = max(newConn, key = len)
    newConn.remove(newGC)

    avgSize = 0

    counter = 0

    for comp in newConn:
        avgSize = avgSize + len(comp)
        counter = counter + 1

    if counter == 0:
        return 0

    return (avgSize / counter)

def getSecondMax(conn):
    if len(conn) < 2:
        return 0

    newConn = conn.copy()

    sortedConn = sorted(newConn, key = len,reverse = True)

    return len(sortedConn[1])



bc = []
c = []

lbc = []
lc = []

N_0L = []
N_1L = []

NB_1L = []
NB_2L = []

AVG_SIZE_CONNL = []
AVG_SIZE_BICONNL = []

SECOND_GCL = []
SECOND_GBCL = []


numDifferentGraphs = 100

numSimsOfGraphs = 25

for net_rep in range(numDifferentGraphs):
    fixed_G = nx.barabasi_albert_graph(N, k, seed=SEED * (net_rep+1))
    assert fixed_G.number_of_nodes() == N


    f=0
    counter = 0

    bcList = []
    bc2List = []
    cList = []

    lbcList = []
    lcList = []

    N_0List = []
    N_1List = []

    NB_1List = []
    NB_2List = []

    AVG_SIZE_CONNList = []
    AVG_SIZE_BICONNList = []

    SECOND_GCList = []
    SECOND_GBCList = []

    G = fixed_G.copy()


    while G.number_of_nodes() > 0:


        assert G.number_of_nodes() == (N - counter*int(step_size*N))

        conn = list(nx.connected_component_subgraphs(G))

        if len(conn) == 0:
            GC = nx.empty_graph(0)
            GCLen = 0
            avgConn = 0
        else:
            GC = max(conn, key=len)
            GCLen = len(GC)
            avgConn = getAvg(conn)

        biConn = list(nx.biconnected_component_subgraphs(G))

        N_0 = getNO(conn)
        N_1 = getN1(conn)

        NB_1 = getBN_1(biConn)
        NB_2 = getBN_2(biConn)


        #NB_0 = getNOBiconn(biConn)

        if len(biConn) == 0:
            GBCLen = 0
            avgBiconn = 0
        else:
            GBCLen =  len(max(biConn, key=len))
            avgBiconn = getAvg(biConn)


        secondGC = getSecondMax(conn)
        secondGBC = getSecondMax(biConn)

        cList.append(len(conn))
        bcList.append(len(biConn))
        lbcList.append(GBCLen)
        lcList.append(GCLen)

        N_0List.append(N_0)
        N_1List.append(N_1)

        NB_1List.append(NB_1)
        NB_2List.append(NB_2)


        AVG_SIZE_CONNList.append(avgConn)
        AVG_SIZE_BICONNList.append(avgBiconn)

        SECOND_GCList.append(secondGC)
        SECOND_GBCList.append(secondGBC)



        #f1.append(f)

        #print(str(counter) + "\t" + str(len(biConn)) + "\t" + str(len(biConnInConn)))

        #print(len(biConnInConn))

        takeNodesOut(G,int(step_size*N))

        counter += 1
        f = f + step_size

    bc.append(bcList)
    c.append(cList)

    lbc.append(lbcList)
    lc.append(lcList)

    N_0L.append(N_0List)
    N_1L.append(N_1List)

    NB_1L.append(NB_1List)
    NB_2L.append(NB_2List)

    AVG_SIZE_CONNL.append(AVG_SIZE_CONNList)
    AVG_SIZE_BICONNL.append(AVG_SIZE_BICONNList)

    SECOND_GCL.append(SECOND_GCList)
    SECOND_GBCL.append(SECOND_GBCList)



fractions = 0

counter = 0

finalList = []

while (fractions <= 1.0):
    (avgCList,stdCList) = indexToTake(c, counter)
    (avgBCList,stdBCList) = indexToTake(bc, counter)

    (avgGC, stdGC) = indexToTake(lc, counter)
    (avgGBC, stdGBC) = indexToTake(lbc,counter)

    (avgN_0,stdN_0) = indexToTake(N_0L,counter)
    (avgN_1,stdN_1) = indexToTake(N_1L,counter)

    (avgNB_1,stdNB_1) = indexToTake(NB_1L,counter)
    (avgNB_2,stdNB_2) = indexToTake(NB_2L,counter)

    (avgSizeComp, stdSizeComp) = indexToTake(AVG_SIZE_CONNL,counter)
    (avgSizeBiComp, stdSizeBiComp) = indexToTake(AVG_SIZE_BICONNL,counter)

    (avgSecondGC, stdSecondGC) = indexToTake(SECOND_GCL,counter)
    (avgSecondGBC, stdSecondGBC) = indexToTake(SECOND_GBCL,counter)

    finalList.append((fractions,avgCList,stdCList,avgBCList,stdBCList,avgGC,stdGC,avgGBC,stdGBC,avgN_0,stdN_0,avgN_1,stdN_1,avgNB_1,stdNB_1,avgNB_2,stdNB_2,avgSizeComp, stdSizeComp,avgSizeBiComp, stdSizeBiComp,avgSecondGC, stdSecondGC,avgSecondGBC, stdSecondGBC))

    fractions = fractions + step_size
    counter = counter + 1


output_file_name = "resultRandomBA_N_%d_k_%d_SEED_%d.csv"%(N,k,SEED)

fh = open(output_file_name, 'w')
writer = csv.writer(fh)

writer.writerow(["f","NC avg", "NC std","NBC avg","NBC std","GC avg","GC std","GBC avg", "GBC std", "N_0 avg", "N_0 std", "N_1 avg", "N_1 std", "NB_1 avg", "NB_1 std", "NB_2 avg", "NB_2 std", "Avg_Comp_Size avg", "Avg_Comp_Size std", "Avg_BiComp_Size avg", "Avg_BiComp_Size std", "SGC avg", "SGC std", "SGBC avg", "SGBC std" ])

for (fractions,avgCList,stdCList,avgBCList,stdBCList,avgGC,stdGC,avgGBC,stdGBC,avgN_0,stdN_0,avgN_1,stdN_1,avgNB_1,stdNB_1,avgNB_2,stdNB_2,avgSizeComp, stdSizeComp,avgSizeBiComp, stdSizeBiComp,avgSecondGC, stdSecondGC,avgSecondGBC, stdSecondGBC) in finalList:
    writer.writerow([fractions,avgCList,stdCList,avgBCList,stdBCList,avgGC,stdGC,avgGBC,stdGBC,avgN_0,stdN_0,avgN_1,stdN_1,avgNB_1,stdNB_1,avgNB_2,stdNB_2,avgSizeComp, stdSizeComp,avgSizeBiComp, stdSizeBiComp,avgSecondGC, stdSecondGC,avgSecondGBC, stdSecondGBC])



fh.close()















