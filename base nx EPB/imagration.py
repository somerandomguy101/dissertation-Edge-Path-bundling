import networkx as nx

from drawv2 import draw_bundle


#-------- LOAD GRAPH--------
G = nx.read_edgelist("migrations.edges")


with open("migrations.nodes", "r") as f:
    for line in f:
        if not line.strip():
            continue
        n, x, y = line.split()
        G.add_node(n, pos=(float(x), float(y)))


#draw_bundle(G, 2, 2, draw_orig=False, highlight_node="235", highlight_radius=5)
draw_bundle(G, 2, 2, draw_orig=False, bundle_strength=0.9, highlight_radius=5)
