import networkx as nx
import osmnx as ox
import random
import math
from operator import itemgetter
import csv
from functools import reduce
import matplotlib.pyplot as plt

plt.switch_backend('agg')

def indexToTake(graphList,index):
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
    degreeDictItemsSorted = sorted(degreeDictItems, key = itemgetter(1),reverse = True)
    onlyNodes = list(map(lambda x:x[0],degreeDictItemsSorted))

    return onlyNodes


def degree_removal(G, numNodesToRemove, remove_order, counter):
    startIndex = numNodesToRemove * counter
    endIndex = startIndex + numNodesToRemove
    if endIndex > len(remove_order):
        nodesToRemove = remove_order[startIndex:]
    else:
        nodesToRemove = remove_order[startIndex:endIndex]
    
    G.remove_nodes_from(nodesToRemove)
    

NY = nx.read_gpickle("NYU.gpickle")
#NY_GBC = nx.read_gpickle("GBC_NY.gpickle")

lbc = []
lc = []

N = NY.number_of_nodes()

k = float(2*(NY.number_of_nodes()/NY.number_of_edges()))

step_size = 0.01

numSimsOfGraphs = 2

for net_rep in range(numSimsOfGraphs):
    print(net_rep)

    f=0
    counter = 0

    lbcList = []
    lcList = []

    G = NY.copy()

    order = createOrder(G)

    while G.number_of_nodes() > 0:

        #assert G.number_of_nodes() == (N - counter*int(step_size*N))

        conn = list(nx.connected_component_subgraphs(G))

        if len(conn) == 0:
            GC = nx.empty_graph(0)
            GCLen = 0
        else:
            GC = max(conn, key=len)
            GCLen = len(GC)

        biConn = list(nx.biconnected_component_subgraphs(G))

        if len(biConn) == 0:
            GBCLen = 0
        else:
            GBCLen =  len(max(biConn, key=len))

        lbcList.append(GBCLen)
        lcList.append(GCLen)

        if G.number_of_nodes() < int(step_size*N):
            break

        degree_removal(G,int(step_size*N), order, counter)

        counter += 1
        f = f + step_size

    lbc.append(lbcList)
    lc.append(lcList)


fractions = 0

counter = 0

finalList = []

lenList = len(lbc[0])

while (counter < lenList):
    (avgGC, stdGC) = indexToTake(lc, counter)
    (avgGBC, stdGBC) = indexToTake(lbc,counter)

    finalList.append((fractions,avgGC, stdGC, avgGBC, stdGBC))

    fractions = fractions + step_size
    counter = counter + 1


output_file_name = "NYRoadRandom_N_%d_k_%g.csv"%(N,k)

fh = open(output_file_name, 'w')
writer = csv.writer(fh)

writer.writerow(["f","GC avg","GC std","GBC avg", "GBC std"])

for (fractions,avgGC, stdGC, avgGBC, stdGBC) in finalList:
    writer.writerow([fractions,avgGC, stdGC, avgGBC, stdGBC])



fh.close()












