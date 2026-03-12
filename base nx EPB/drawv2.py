import networkx as nx
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Circle

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


def segment_in_lens(a, b, center, radius):
    return point_distance(center, a, b) < radius


def segment_angle(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return math.atan2(dy, dx)


def snap_perpendicular(angle):
    return round(angle / (math.pi / 2)) * (math.pi / 2)


def relax_path_in_lens(xs, ys, center, radius, strength=0.7):
    """
    adjusts points sequentially to align towards orthagonal angles inside the lens.
    includes error correction to ensure path still hits its final target node
    """
    new_xs, new_ys = [xs[0]], [ys[0]]

    for i in range(1, len(xs)):
        a = (new_xs[i-1], new_ys[i-1])
        b = (xs[i], ys[i])

        if segment_in_lens(a, b, center, radius):
            angle = segment_angle(a, b)
            target_angle = snap_perpendicular(angle)

            #calculate shortest angular distance to path
            diff = (target_angle - angle + math.pi) % (2 * math.pi) - math.pi

            #relax blend towards the target angle based on 'strength'
            relaxed_angle = angle + diff * strength

            #reconstruct point 'b' by preserving the original segment length
            length = math.hypot(b[0] - a[0], b[1] - a[1])
            b = (a[0] + length * math.cos(relaxed_angle), a[1] + length * math.sin(relaxed_angle))

        new_xs.append(b[0])
        new_ys.append(b[1])

    #error correction - sequential angle will cause the end of the path to miss its target node, we linearly distribute that drift back over the path
    if len(xs) > 1:
        err_x = xs[-1] - new_xs[-1]
        err_y = ys[-1] - new_ys[-1]
        for i in range(len(xs)):
            t = i / (len(new_xs) - 1)
            new_xs[i] += err_x * t
            new_ys[i] += err_y * t

    return new_xs, new_ys



def draw_bundle(G, k = 2, d = 2,  draw_orig = True, highlight_node = None, highlight_radius = 10, lens_center = None, lens_radius = 25, snap_strength = 0.7):
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

    #--------CREATE LENS--------
    if lens_center is not None:
        ax = plt.gca()
        lens = Circle((lens_center[0], lens_center[1]), lens_radius, fill=False, edgecolor="black", linewidth=2)
        ax.add_patch(lens)

    # -------- DRAW BUNDLED GRAPH--------
    for e, path in bundle.items():
        highlight = False

        xs = [pos[n][0] for n in path]
        ys = [pos[n][1] for n in path]

        if highlight_node is not None:
            highlight = bundle_near_node(path, pos, highlight_node, highlight_radius)

        colour = "blue" if highlight else "black"

        # apply relaxed lens
        if lens_center is not None:
            xs, ys = relax_path_in_lens(xs, ys, lens_center, lens_radius, strength=snap_strength)

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

