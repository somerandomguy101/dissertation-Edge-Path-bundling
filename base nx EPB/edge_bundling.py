import networkx as nx

def norm(e):
    return tuple(sorted(e))

def edge_path_bundling(G, k, d):

    lock = {norm(e): False for e in G.edges}
    skip = {norm(e): False for e in G.edges}

    weight = {
        norm(e): (G.edges[e]["length"] ** d) for e in G.edges
    }

    sorted_edges = sorted(G.edges, key=lambda e: weight[norm(e)], reverse=True)
    control_points = {}

    for e in sorted_edges:
        e = norm(e)
        if lock[e]:
            continue

        skip[norm(e)] = True
        s, t = e

        # build temporary graph excluding skipped edges
        H = nx.Graph()
        H.add_nodes_from(G.nodes)

        for u, v in G.edges:
            if not skip[norm((u, v))]:
                H.add_edge(u, v, weight=weight[norm((u, v))])

        try:
            path_nodes = nx.shortest_path(H, s, t, weight="weight")
        except nx.NetworkXNoPath:
            skip[norm(e)] = False
            continue

        # convert node path -> edge path
        path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))

        path_length = sum(G.edges[m]["length"] for m in path_edges)

        if path_length > k * G.edges[e]["length"]:
            skip[norm(e)] = False
            continue

        for m in path_edges:
            lock[norm(m)] = True

        control_points[e] = path_nodes

    return control_points