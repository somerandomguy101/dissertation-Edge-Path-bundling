import matplotlib.pyplot as plt
from sympy.abc import alpha

from edge_path_bundling import *

pos = {
    "U": (0, 0),
    "V": (2, 1),
    "X": (4, 1),
    "W": (3, -1),
}

edges = [
    ("U", "V"), ("U", "W"), ("V", "X"), ("V", "W"), ("X", "W")
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
    plt.plot(x, y, alpha=0.3)

# bundled edges
for e,path in bundles.items():
    pts = [e[0]]
    for pe in path:
        pts.append(pe[1])

    xs = [pos[p][0] for p in pts]
    ys = [pos[p][1] for p in pts]
    plt.plot(xs, ys, linewidth=3)

# nodes
for n,(x,y) in pos.items():
    plt.scatter(x,y)
    plt.text(x+0.02,y+0.02,n)

plt.title("Simple Edge Bundling Test")
plt.axis("equal")
plt.show()