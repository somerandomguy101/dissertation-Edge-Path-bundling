import networkx as nx

def edge_path_bundling(G, k, d):

    lock = {e: False for e in G.edges}
    skip = {e: False for e in G.edges}

    weight = {
        e: (G.edges[e]["length"] ** d) for e in G.edges
    }

    sorted_edges = sorted(G.edges, key=lambda e: weight[e], reverse=True)
    control_points = {}

    for e in sorted_edges:
        if lock[e]:
            continue

        skip[e] = True
        s, t = e

        # build temporary graph excluding skipped edges
        H = nx.Graph()
        for u, v in G.edges:
            if not skip[(u, v)]:
                H.add_edge(u, v, weight=weight[(u, v)])

        try:
            path_nodes = nx.shortest_path(H, s, t, weight="weight")
        except nx.NetworkXNoPath:
            skip[e] = False
            continue

        # convert node path -> edge path
        path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))

        path_length = sum(weight[m] for m in path_edges)

        if path_length > k * G.edges[e]["length"]:
            skip[e] = False
            continue

        for m in path_edges:
            lock[m] = True

        control_points[e] = path_nodes

    return control_points