import pickle
import networkx as nx
import osmnx as ox

BGU = nx.read_gpickle("BelgradeDriveU.gpickle")
BJU = nx.read_gpickle("BeijingU.gpickle")
NYU = nx.read_gpickle("NYU.gpickle")

GBC_BG = nx.read_gpickle("GBC_BG.gpickle")
GBC_BJ = nx.read_gpickle("GBC_BJ.gpickle")
GBC_NY = nx.read_gpickle("GBC_NY.gpickle")

print(len(BGU))
print(len(BJU))
print(len(NYU))

print(len(GBC_BG))
print(len(GBC_BJ))
print(len(GBC_NY))