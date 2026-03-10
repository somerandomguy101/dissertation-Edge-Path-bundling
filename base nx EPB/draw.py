import networkx as nx
import matplotlib.pyplot as plt
import math
from edge_bundling import edge_path_bundling
from scipy.interpolate import make_interp_spline
import numpy as np

def draw_bundle(G, k = 2, d = 2,  draw_orig = True):
    # extract positions
    pos = nx.get_node_attributes(G, "pos")

    # --------COMPUTE EDGE LENGTHS--------
    for u, v in G.edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        length = math.hypot(x1 - x2, y1 - y2)
        G.edges[u, v]["length"] = length

    # ---------- RUN BUNDLING ----------
    bundle = edge_path_bundling(G, k, d)


    # ---------- DRAW ORIGINAL GRAPH ----------
    plt.figure(figsize=(12, 8))

    if draw_orig:
        nx.draw(
            G,
            pos,
            with_labels=False,
            edge_color="lightgray",
            node_color="black",
            font_color="white",
            node_size=1,
        )


    # -------- DRAW BUNDLED GRAPH--------
    for e, path in bundle.items():

        xs = [pos[n][0] for n in path]
        ys = [pos[n][1] for n in path]

        if len(xs) >= 3:
            t = np.linspace(0, 1, len(xs))
            t_smooth = np.linspace(0, 1, 100)

            spl_x = make_interp_spline(t, xs, k=2)
            spl_y = make_interp_spline(t, ys, k=2)

            plt.plot(spl_x(t_smooth), spl_y(t_smooth), color="black")

        else:
            plt.plot(xs, ys, color="black")

    plt.title("Bundled Graph")
    plt.axis("equal")
    plt.show()

