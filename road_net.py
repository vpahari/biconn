import osmnx as ox
import networkx as nx

Belgrade = ox.graph_from_place('Belgrade, Serbia')

NY = ox.graph_from_place('New York City, New York, USA')

Beijing = ox.graph_from_place('Beijing, China', which_result=2, network_type='drive')

biconn_Belgrade = list(nx.biconnected_component_subgraphs(Belgrade)) 

biconn_NY = list(nx.biconnected_component_subgraphs(NY)) 

biconn_Beijing = list(nx.biconnected_component_subgraphs(Beijing)) 

GBC_Belgrade = max(biconn_Belgrade, key=len)

GBC_NY = max(biconn_NY, key=len)

GBC_Beijing = max(biconn_Beijing, key=len)

nx.write_gpickle(GBC_Belgrade,"GBC_Belgrade.gpickle")

nx.write_gpickle(GBC_NY,"GBC_NY.gpickle")

nx.write_gpickle(GBC_Beijing,"GBC_Beijing.gpickle")