import networkx as nx

from processing.drawv2 import draw_bundle


#-------- LOAD GRAPH--------
G = nx.read_edgelist("../datasets/airlines.edges")


with open("../datasets/airlines.nodes", "r") as f:
    for line in f:
        if not line.strip():
            continue
        n, x, y = line.split()
        G.add_node(n, pos=(float(x), float(y)))

# bundling strength ~0.7 or higher recommended
draw_bundle(G, 2, 2, highlight_node=None, draw_orig=True, bundle_strength=0.8, highlight_radius=5, snap_strength=0.75)