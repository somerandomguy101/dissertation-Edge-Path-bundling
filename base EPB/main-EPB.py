import heapq

def dijkstra(G, source, target, weights, skip):
    pq = [(0, source, [])]
    visited = set()

    while pq:
        dist, node, path = heapq.heappop(pq)

        if node in visited:
            continue
        visited.add(node)

        if node == target:
            return path

        for neighbor, edge_id in G[node]:
            if skip[edge_id]:
                continue
            if neighbor not in visited:
                heapq.heappush(pq, (dist + weights[edge_id], neighbor, path + [edge_id]))

    return None

def edge_path_bundling(G, edges, DG_lengths, k, d):
    # initialise dicts
    lock = {e: False for e in edges}
    skip = {e: False for e in edges}
    weight = {e: (DG_lengths[e] ** d) for e in edges}

    # sort edges descending by weight
    sorted_edges = sorted(edges, key=lambda e: weight[e], reverse=True)

    control_points = {}

    for e in sorted_edges:
        if lock[e]:
            continue

        skip[e] = True
        s, t = e

        p = dijkstra(G, s, t, weight, skip)

        if p is None:
            skip[e] = False
            continue

        if sum(weight[m] for m in p) > k * DG_lengths[e]:
            skip[e] = False
            continue

        # lock edges in path
        for m in p:
            lock[m] = True

        control_points[e] = p

    return control_points


