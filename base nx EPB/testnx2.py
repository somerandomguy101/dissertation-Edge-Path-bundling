import networkx as nx
import matplotlib.pyplot as plt
import math
from edge_bundling import edge_path_bundling
from scipy.interpolate import make_interp_spline
import numpy as np

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

# draw original graph
nx.draw(G,pos,with_labels=True,edge_color="lightgray", node_color="black", font_color="white", edge_width=0.5)

# draw bundled edges
for e,path in bundles.items():
    xs=[pos[n][0] for n in path]
    ys=[pos[n][1] for n in path]

    # only spline if a path has enough points
    if len(xs) >=3:
        t = np.linspace(0,1,len(xs))
        t_smooth = np.linspace(0,1,100)

        spl_x = make_interp_spline(t, xs, k=2)
        spl_y = make_interp_spline(t, ys, k=2)

        xs_smooth = spl_x(t_smooth)
        ys_smooth = spl_y(t_smooth)

        plt.plot(xs_smooth, ys_smooth, color="black", linewidth=0.5, alpha=0.25)

    else:
        plt.plot(xs,ys, color="black", linewidth=0.5, alpha=0.25)

plt.title("Bundled Graph")
plt.axis("equal")
plt.show()