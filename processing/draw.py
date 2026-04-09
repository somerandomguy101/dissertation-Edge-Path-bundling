import networkx as nx
import matplotlib.pyplot as plt
import math
from edge_bundling import edge_path_bundling
from scipy.interpolate import make_interp_spline
import numpy as np


def point_distance(p, a, b):
    px, py = p
    x1, y1 = a
    x2, y2 = b

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)

    t = ((px-x1)*dx + (py-y1)*dy) / (dx*dx + dy*dy)
    t = max(0, min(1, t))

    closest_x = x1+t * dx
    closest_y = y1+t * dy

    return math.hypot(px - closest_x, py - closest_y)



def bundle_near_node(path_nodes, pos, node, radius = 10):
    node_pos = pos[node]

    for i in range(len(path_nodes)-1):
        a = pos[path_nodes[i]]
        b = pos[path_nodes[i+1]]

        if point_distance(node_pos, a, b) < radius:
            return True

    return False



def draw_bundle(G, k = 2, d = 2,  draw_orig = True, highlight_node = None, highlight_radius = 10):
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
        highlight = False

        xs = [pos[n][0] for n in path]
        ys = [pos[n][1] for n in path]

        if highlight_node is not None:
            highlight = bundle_near_node(path, pos, highlight_node, highlight_radius)

        colour = "blue" if highlight else "black"

        if len(xs) >= 3:
            t = np.linspace(0, 1, len(xs))
            t_smooth = np.linspace(0, 1, 100)

            spl_x = make_interp_spline(t, xs, k=2)
            spl_y = make_interp_spline(t, ys, k=2)

            plt.plot(spl_x(t_smooth), spl_y(t_smooth), color=colour, linewidth=0.5)

        else:
            plt.plot(xs, ys, color=colour, linewidth=0.5)

    plt.title("Bundled Graph")
    plt.axis("equal")
    plt.show()

