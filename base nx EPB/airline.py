import networkx as nx

from drawv2 import draw_bundle


#-------- LOAD GRAPH--------
G = nx.read_edgelist("airlines.edges")


with open("airlines.nodes", "r") as f:
    for line in f:
        if not line.strip():
            continue
        n, x, y = line.split()
        G.add_node(n, pos=(float(x), float(y)))


draw_bundle(G, 2, 2, highlight_node=None, draw_orig=False)