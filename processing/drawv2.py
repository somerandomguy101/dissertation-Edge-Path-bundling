import networkx as nx

import matplotlib
matplotlib.use('TkAgg') # done so that the window stays open allowing the user to click

import matplotlib.pyplot as plt
import math
from matplotlib.patches import Circle
from matplotlib.widgets import Slider

from processing.edge_bundling import edge_path_bundling
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
    # Convert NumPy arrays to standard Python lists of floats to avoid scalar indexing errors
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]

    new_xs = list(xs)
    new_ys = list(ys)

    i = 0
    while i < len(xs) - 1:
        # Check if the current segment is inside the lens
        if segment_in_lens((xs[i], ys[i]), (xs[i + 1], ys[i + 1]), center, radius):
            start_idx = i

            # Fast-forward to find the end of this continuous in-lens section
            while i < len(xs) - 1 and segment_in_lens((xs[i], ys[i]), (xs[i + 1], ys[i + 1]), center, radius):
                i += 1
            end_idx = i

            # Extract just the sub-path that is inside the lens
            sub_xs = xs[start_idx:end_idx + 1]
            sub_ys = ys[start_idx:end_idx + 1]

            # Ensure these are strictly initialized as LISTS, not scalar variables
            rel_xs = [sub_xs[0]]
            rel_ys = [sub_ys[0]]

            # Relax angles sequentially JUST for this sub-path
            for j in range(1, len(sub_xs)):
                pa = (rel_xs[j - 1], rel_ys[j - 1])
                pb = (sub_xs[j], sub_ys[j])

                angle = segment_angle(pa, pb)
                target_angle = snap_perpendicular(angle)

                # Calculate shortest angular distance
                diff = (target_angle - angle + math.pi) % (2 * math.pi) - math.pi

                # Relax towards the target angle
                relaxed_angle = angle + diff * strength

                # Reconstruct point 'b' preserving segment length
                length = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
                new_bx = pa[0] + length * math.cos(relaxed_angle)
                new_by = pa[1] + length * math.sin(relaxed_angle)

                rel_xs.append(new_bx)
                rel_ys.append(new_by)

            # Local error correction: distribute the drift back over ONLY this specific sub-path
            if len(sub_xs) > 1:
                err_x = sub_xs[-1] - rel_xs[-1]
                err_y = sub_ys[-1] - rel_ys[-1]
                for j in range(len(rel_xs)):
                    t = j / (len(rel_xs) - 1)
                    new_xs[start_idx + j] = rel_xs[j] + err_x * t
                    new_ys[start_idx + j] = rel_ys[j] + err_y * t
        else:
            i += 1

    return new_xs, new_ys


def draw_bundle(G, k=2, d=2, draw_orig=True, highlight_node=None, highlight_radius=10, initial_lens_center=None, lens_radius=25, snap_strength=0.7, bundle_strength=0.75):
    # 1. Extract positions and compute edge lengths
    pos = nx.get_node_attributes(G, "pos")
    for u, v in G.edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        G.edges[u, v]["length"] = math.hypot(x1 - x2, y1 - y2)

    # 2. Run bundling ONCE (computationally heavy)
    print("Computing edge bundles...")
    bundle = edge_path_bundling(G, k, d)

    # 3. Set up the Matplotlib figure and state
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create UI menu
    plt.subplots_adjust(bottom=0.3)

    # define positions for sliders
    ax_snap = fig.add_axes([0.2, 0.22, 0.6, 0.03])
    ax_bundle = fig.add_axes([0.2, 0.17, 0.6, 0.03])
    ax_lens_rad = fig.add_axes([0.2, 0.12, 0.6, 0.03])
    ax_high_rad = fig.add_axes([0.2, 0.07, 0.6, 0.03])

    # create the sliders, passing the initial function values as defaults
    slider_snap = Slider(ax_snap, 'Snap Strength', 0.0, 1.0, valinit=snap_strength)
    slider_bundle = Slider(ax_bundle, 'Bundle Strength', 0.0, 1.0, valinit=bundle_strength)
    slider_lens_rad = Slider(ax_lens_rad, 'Lens Radius', 0.0, 25.0, valinit=lens_radius)
    slider_high_rad = Slider(ax_high_rad, 'Highlight Radius', 1.0, 50.0, valinit=highlight_radius)

    # We use a dictionary to store state so it can be modified by the nested functions
    state = {'lens_center': initial_lens_center,
             'highlight_node': highlight_node,
             'sliders': [slider_snap, slider_bundle, slider_lens_rad, slider_high_rad]
             }

    def update_plot(val=None):
        """Clears and redraws the graph with the current lens position."""
        ax.clear()

        # grab current slider vals from UI
        cur_snap = slider_snap.val
        cur_bundle = slider_bundle.val
        cur_lens_rad = slider_lens_rad.val
        cur_high_rad = slider_high_rad.val

        # Draw original graph
        if draw_orig:
            nx.draw(
                G, pos, ax=ax, with_labels=False, edge_color="lightgray",
                node_color="black", node_size=1
            )

        # Draw the lens circle
        center = state['lens_center']
        if center is not None:
            lens = Circle((center[0], center[1]), cur_lens_rad, fill=False, edgecolor="lightgrey", linewidth=2, zorder=5)
            ax.add_patch(lens)

        # Draw bundled graph
        for e, path in bundle.items():
            bundled_xs = np.array([pos[n][0] for n in path])
            bundled_ys = np.array([pos[n][1] for n in path])
            num_pts = len(path)

            t_vals = np.linspace(0, 1, num_pts)
            straight_xs = (1 - t_vals) * bundled_xs[0] + t_vals * bundled_xs[-1]
            straight_ys = (1 - t_vals) * bundled_ys[0] + t_vals * bundled_ys[-1]

            xs = (1 - cur_bundle) * straight_xs + cur_bundle * bundled_xs
            ys = (1 - cur_bundle) * straight_ys + cur_bundle * bundled_ys

            highlight = False
            if state['highlight_node'] is not None:
                highlight = bundle_near_node(path, pos, state['highlight_node'], cur_high_rad)

            colour = "blue" if highlight else "black"

            # Apply the relaxed lens distortion
            if center is not None:
                xs, ys = relax_path_in_lens(xs, ys, center, cur_lens_rad, strength=cur_snap)

            # Plot splines or straight lines
            if len(xs) > 3:
                t = np.linspace(0, 1, len(xs))
                t_smooth = np.linspace(0, 1, 100)
                spl_x = make_interp_spline(t, xs, k=3)
                spl_y = make_interp_spline(t, ys, k=3)
                ax.plot(spl_x(t_smooth), spl_y(t_smooth), color=colour, linewidth=0.5, zorder=3)

            elif len(xs) == 3:
                t = np.linspace(0, 1, len(xs))
                t_smooth = np.linspace(0, 1, 100)
                spl_x = make_interp_spline(t, xs, k=2)
                spl_y = make_interp_spline(t, ys, k=2)
                ax.plot(spl_x(t_smooth), spl_y(t_smooth), color=colour, linewidth=0.5, zorder=3)

            else:
                ax.plot(xs, ys, color=colour, linewidth=0.5, zorder=3)

        if state['highlight_node'] is not None:
            node_x, node_y = pos[state['highlight_node']]
            ax.scatter(node_x, node_y, s=30, color="gray", zorder=20, edgecolors="white")

        ax.set_title("Interactive Bundled Graph (Click to move lens)")
        ax.axis("equal")

        guide_text = (
            "Controls:\n"
            "Left click: Move lens, "
            "Right click: Remove lens\n"
            "H: Highlight nearest node (clicking H again on the highlighted node will remove)\n"
            "C: Clear lens and highlight selections"
        )

        ax.text(
            0.01, 0.99, guide_text,
            transform=ax.transAxes,  # makes coords relative (0–1)
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='left'
        )

        fig.canvas.draw_idle()  # Efficiently redraw the canvas

    # register slider changes to update plot function
    slider_snap.on_changed(update_plot)
    slider_bundle.on_changed(update_plot)
    slider_lens_rad.on_changed(update_plot)
    slider_high_rad.on_changed(update_plot)

    def on_click(event):
        """Handles mouse click events on the canvas."""
        # Ignore clicks outside the actual plotting axes (like on the toolbar)
        if event.inaxes != ax:
            return

        if event.button == 3: # if it's a right click remove the lens
            state['lens_center'] = None

        else:
            # Update the lens center to the click coordinates and redraw
            state['lens_center'] = (event.xdata, event.ydata)

        update_plot()

    def on_key(event):
        """Handles key press events on the canvas."""
        if event.inaxes != ax:
            return

        if event.key == "h":
            mouse_x, mouse_y = event.xdata, event.ydata

            # find the closest node to the mouse
            closest_node = None
            min_dist = float('inf')

            for node, n_pos in pos.items():
                dist = math.hypot(mouse_x - n_pos[0], mouse_y - n_pos[1])
                if min_dist > dist:
                    min_dist = dist
                    closest_node = node

            if state['highlight_node'] == closest_node:
                state['highlight_node'] = None

            else:
                state['highlight_node'] = closest_node

            update_plot()

        elif event.key == "c":
            state['lens_center'] = None
            state['highlight_node'] = None
            update_plot()


    # 4. Connect the click/key events to our handler function
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)


    # 5. Trigger the first draw and show the plot
    update_plot()
    plt.show()