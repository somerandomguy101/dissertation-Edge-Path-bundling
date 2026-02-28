import networkx as nx
from edge_bundling import edge_path_bundling
import math

G = nx.Graph()

pos = {
    "A": (0, 0),
    "B": (1, 3),
    "C": (3, 6),
    "D": (5, 8),
    "E": (7, 10),
    "F": (9, 11),
    "G": (12, 12),
}

def dist(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

edges = [
    ("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "G"), ("A", "G"), ("C", "F")
]

for u,v in edges:
    G.add_edge(u,v,length=dist(pos[u],pos[v]))

bundles = edge_path_bundling(G, 2, 1)

print("Bundled paths:")
for e,p in bundles.items():
    print(e,"→",p)