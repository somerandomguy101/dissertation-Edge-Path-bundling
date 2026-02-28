import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, splprep
import numpy as np
from sympy.abc import alpha

from edge_path_bundling import *

pos = {
    "A": (0, 0),
    "B": (1, 3),
    "C": (3, 6),
    "D": (5, 8),
    "E": (7, 10),
    "F": (9, 11),
    "G": (12, 12),
}

edges = [
    ("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "G"), ("A", "G"), ("C", "F")
]

G = {n: [] for n in pos}
for u,v in edges:
    G[u].append((v, (u,v)))
    G[v].append((u, (u, v)))

# edge lengths
def dist(a,b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

lengths = {e: dist(pos[e[0]], pos[e[1]]) for e in edges}

#-------- run bundling --------
bundles = edge_path(G, edges, lengths, 3, 1)

#-------- draw --------
plt.figure()

# original edges
for u,v in edges:
    x = [pos[u][0], pos[v][0]]
    y = [pos[u][1], pos[v][1]]
    plt.plot(x, y, alpha=0.3, color='black')

# bundled edges
for e,path in bundles.items():
    pts = [e[0]]
    for pe in path:
        pts.append(pe[1])

    xs = [pos[p][0] for p in pts]
    ys = [pos[p][1] for p in pts]


    X_Y_spline = make_interp_spline(xs, ys)
    X_ = np.linspace(np.array(xs).min(), np.array(xs).max(), 500)
    Y_ = X_Y_spline(X_)

    plt.plot(X_, Y_, color='black')

# nodes
for n,(x,y) in pos.items():
    plt.scatter(x,y, color='black')
    plt.text(x+0.02,y+0.02,n)

plt.title("Simple Edge Bundling Test")
plt.axis("equal")
plt.show()