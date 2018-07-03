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

def makeEqualSize(l, largestLen):
    for i in l:
        while len(i) < largestLen:
            i.append(0)

def getLargestSize(l):
    largestLen = 0
    for i in l:
        if len(i) > largestLen:
            largestLen = len(i)

    return largestLen
    

def takeNodesOut(G, numNodesToRemove):
    randomNodes = random.sample(list(G.nodes()),10)
    nodesToRemoveTemp = []
    for i in randomNodes:
        neighbors = list(G.neighbors(i))
        nodesToRemoveTemp = nodesToRemoveTemp + neighbors

    nodesToRemoveTemp = nodesToRemoveTemp + randomNodes

    nodesToRemove = list(set(nodesToRemoveTemp))

    #print(len(nodesToRemove))

    G.remove_nodes_from(nodesToRemove)


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

steps = []

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

REM_NODESL = []


numDifferentGraphs = 25

numSimsOfGraphs = 10

for net_rep in range(numDifferentGraphs):

    print(net_rep)

    fixed_G = nx.erdos_renyi_graph(N, p, seed=SEED * (net_rep+1))

    for sim_rep in range(numSimsOfGraphs):


        f=0
        counter = 0

        stepsList = []

        REM_NODESList = []

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


            #assert G.number_of_nodes() == (N - counter*int(step_size*N))

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

            stepsList.append(f)

            REM_NODESList.append(G.number_of_nodes())


            """

            nodesTakeOut = int(step_size * N)

            if nodesTakeOut <= G.number_of_nodes():

                takeNodesOut(G,G.number_of_nodes())

            else:

                takeNodesOut(G,nodesTakeOut)

            """

            if G.number_of_nodes() < 10:
                break

            takeNodesOut(G,int(step_size * N))


            counter += 1

            f = f + 1
        
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

        steps.append(stepsList)

        REM_NODESL.append(REM_NODESList)


fractions = 0

counter = 0

finalList = []

longestList = getLargestSize(steps)

makeEqualSize(c, longestList)
makeEqualSize(bc, longestList)

makeEqualSize(lc, longestList)
makeEqualSize(lbc, longestList)

makeEqualSize(N_0L, longestList)
makeEqualSize(N_1L, longestList)

makeEqualSize(NB_1L, longestList)
makeEqualSize(NB_2L, longestList)

makeEqualSize(AVG_SIZE_CONNL, longestList)
makeEqualSize(AVG_SIZE_BICONNL, longestList)

makeEqualSize(SECOND_GCL, longestList)
makeEqualSize(SECOND_GBCL, longestList)

makeEqualSize(steps, longestList)
makeEqualSize(REM_NODESL, longestList)

lenForAvg = len(steps[0])


while (counter < lenForAvg):
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

    (avgSteps, stdSteps) = indexToTake(steps, counter)
    (avgNodeRem, stdNodeRem) = indexToTake(REM_NODESL, counter)

    finalList.append((avgSteps, stdSteps, avgNodeRem, stdNodeRem, avgCList,stdCList,avgBCList,stdBCList,avgGC,stdGC,avgGBC,stdGBC,avgN_0,stdN_0,avgN_1,stdN_1,avgNB_1,stdNB_1,avgNB_2,stdNB_2,avgSizeComp, stdSizeComp,avgSizeBiComp, stdSizeBiComp,avgSecondGC, stdSecondGC,avgSecondGBC, stdSecondGBC))

    fractions = fractions + step_size
    counter = counter + 1


output_file_name = "resultNewUrinal_N_%d_k_%d_SEED_%d.csv"%(N,k,SEED)

fh = open(output_file_name, 'w')
writer = csv.writer(fh)

writer.writerow(["f avg","f std","Node_Rem avg", "Node_Rem std", "NC avg", "NC std","NBC avg","NBC std","GC avg","GC std","GBC avg", "GBC std", "N_0 avg", "N_0 std", "N_1 avg", "N_1 std", "NB_1 avg", "NB_1 std", "NB_2 avg", "NB_2 std", "Avg_Comp_Size avg", "Avg_Comp_Size std", "Avg_BiComp_Size avg", "Avg_BiComp_Size std", "SGC avg", "SGC std", "SGBC avg", "SGBC std" ])

for (avgSteps,stdSteps,avgNodeRem, stdNodeRem,avgCList,stdCList,avgBCList,stdBCList,avgGC,stdGC,avgGBC,stdGBC,avgN_0,stdN_0,avgN_1,stdN_1,avgNB_1,stdNB_1,avgNB_2,stdNB_2,avgSizeComp, stdSizeComp,avgSizeBiComp, stdSizeBiComp,avgSecondGC, stdSecondGC,avgSecondGBC, stdSecondGBC) in finalList:
    writer.writerow([avgSteps,stdSteps,avgNodeRem, stdNodeRem,avgCList,stdCList,avgBCList,stdBCList,avgGC,stdGC,avgGBC,stdGBC,avgN_0,stdN_0,avgN_1,stdN_1,avgNB_1,stdNB_1,avgNB_2,stdNB_2,avgSizeComp, stdSizeComp,avgSizeBiComp, stdSizeBiComp,avgSecondGC, stdSecondGC,avgSecondGBC, stdSecondGBC])



fh.close()









