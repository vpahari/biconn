import networkit as nk
import pickle
import networkx as nx

def findAllDisjointPaths(G,s,t,shortestPath, wt):
	distPaths = []
	weights = []

	distPaths.append(shortestPath)
	weights.append(wt)

	if len(shortestPath) == 2:
		G.removeEdge(s,t)
		G.removeEdge(t,s)

	else:
		nodesToRemove = shortestPath[1:len(shortestPath)-1]

		for node in nodesToRemove:
			tempG.removeNode(node)


	newDist = nk.distance.Dijkstra(G, s, True, False)

	newDist.run()

	hasPath = newDist.numberOfPaths(t)

	while hasPath != 0:
		
		shortestPath = newDist.getPath(t)

		spWt = newDist.distance(t)

		for i in shortestPath[1:len(shortestPath) - 1]:
			G.removeNode(i)

		
		distPaths.append(shortestPath)
		weights.append(spWt)
		
		newDist = nk.distance.Dijkstra(G, s, True, True)
		
		newDist.run()
		
		hasPath = newDist.numberOfPaths(t)

	#print(distPaths)
		
	return (distPaths, weights)



file_name= "USairport500.txt"

#file_obj = open(file_name,'r')

#file_obj.readline()

lines = [line.rstrip('\n') for line in open(file_name)]

newGraph = nk.graph.Graph(n = 500, weighted = True, directed = True)

for line in lines:

	data = line.split(" ")

	print(data)

	print(len(data))

	assert len(data) == 2

	source = int(data[0]) - 1
	target = int(data[1]) - 1
	weight = int(data[2])

	newGraph.addEdge(source,target, w = weight)


listOfNodes = newGraph.nodes()

lengthOfNodes = len(listOfNodes)

listWeights = []

listDistPaths = []

listNumberOfPaths = []

listOfDegree = []

for s in listOfNodes:

	weights = []

	distPaths = []

	numberOfPaths = []

	dijk = nk.distance.Dijkstra(newGraph, s, True, False)

	dijk.run()

	for t in listOfNodes:

		if s == t:

			weights.append(0)
			distPaths.append([])
			numberOfPaths.append(0)

			continue

		isPath = dijk.numberOfPaths(t)

		if isPath == 0:
			distPaths.append([])
			weights.append(0)
			numberOfPaths.append(0)

			continue


		tempG = newGraph.copyNodes()

		for (e1,e2) in newGraph.edges():
			tempG.addEdge(e1,e2)

		
		shortestPath = dijk.getPath(t)
		wt = dijk.distance(t)

		(DP, wtList) = findAllDisjointPaths(tempG,s,t, shortestPath,wt)
		number_of_DP = len(DP)
		

		weights.append(wtList)

		distPaths.append(DP)

		numberOfPaths.append(number_of_DP)

	listWeights.append(weights)
	listOfDegree.append(newGraph.degree(s))
	listDistPaths.append(distPaths)
	listNumberOfPaths.append(numberOfPaths)


listWeights_str = "airport_weights.pkl"
listOfDegree_str = "airport_degrees.pkl"
listDistPaths_str =  "airport_distPaths.pkl"
listNumberOfPaths_str = "airport_numPaths.pkl"

with open(listWeights_str, 'wb') as f:
	pickle.dump(listWeights, f)

with open(listOfDegree_str, 'wb') as f:
	pickle.dump(listOfDegree, f)

with open(listDistPaths_str, 'wb') as f:
	pickle.dump(listDistPaths, f)

with open(listNumberOfPaths_str, 'wb') as f:
	pickle.dump(listNumberOfPaths, f)

























