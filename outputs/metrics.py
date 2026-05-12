"""
metrics.py
==========

Quantitative evaluation utilities for the Orthogonal Lens edge-path bundling
system. Produces the numbers used in the "Evaluation using Metrics" section
of the dissertation.

Two families of metrics are computed:

1.  Ink-style measures, in the spirit of Wallinger et al. (2022):
        - raw_ink:        total length of the original edge set (straight).
        - cumulative_ink: sum of routing-path lengths over every edge
                          (bundled edges follow control points; unbundled
                          edges remain straight). This represents drawn
                          ink ignoring overlap.
        - unique_ink:     total length of the *distinct* underlying edges
                          that appear in the rendered drawing. Bundled
                          paths share segments, so this captures visual
                          compression.
        - ink_ratio:      unique_ink / raw_ink.  < 1 indicates clutter
                          reduction at the rendering level.
        - mean_detour:    average over bundled edges of
                          (path_length / direct_edge_length).  Bounded
                          above by k.

2.  Geometric distortion introduced by the perpendicular lens, computed
    by replaying the relax_path_in_lens transformation over a sample of
    lens placements:
        - mean_displacement     mean Euclidean distance moved by points
                                that lie inside the lens.
        - p95_displacement      95th-percentile of the same.
        - max_displacement      worst-case displacement observed.
        - mean_length_change    mean absolute change in segment length
                                for segments crossing the lens boundary.
        - boundary_drift        mean residual error at the join between
                                in-lens and out-of-lens path segments
                                after the local error correction pass.

Usage
-----
    import networkx as nx
    from edge_bundling import edge_path_bundling
    from metrics import (
        compute_ink_metrics,
        compute_lens_distortion_metrics,
        sample_lens_centers,
        format_table,
    )

    # G is your NetworkX graph with a 'pos' attribute on every node.
    bundle = edge_path_bundling(G, k=2, d=2)

    ink   = compute_ink_metrics(G, bundle)
    lens  = compute_lens_distortion_metrics(
        G, bundle,
        bundle_strength = 0.75,
        snap_strength   = 0.7,
        lens_radius     = 25,
        lens_centers    = sample_lens_centers(G, n=30, radius=25, seed=0),
    )

    print(format_table("Airlines", ink, lens))

The script is intentionally self-contained: the geometric helpers and the
relax routine are duplicated here so it does not pull in matplotlib.
They are the same routines used in drawv2.py.
"""

from __future__ import annotations

import math
import random
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx


# ---------------------------------------------------------------------------
# Geometric helpers (mirrors of those in drawv2.py)
# ---------------------------------------------------------------------------
def _norm(e: Tuple) -> Tuple:
    return tuple(sorted(e))


def _point_distance(p, a, b) -> float:
    px, py = p
    x1, y1 = a
    x2, y2 = b
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx = x1 + t * dx
    cy = y1 + t * dy
    return math.hypot(px - cx, py - cy)


def _segment_in_lens(a, b, center, radius) -> bool:
    return _point_distance(center, a, b) < radius


def _segment_angle(a, b) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _snap_perpendicular(angle: float) -> float:
    return round(angle / (math.pi / 2)) * (math.pi / 2)


def _relax_path_in_lens(xs, ys, center, radius, strength=0.7):
    """Identical to drawv2.relax_path_in_lens; copied to avoid a
    matplotlib import."""
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    new_xs = list(xs)
    new_ys = list(ys)

    i = 0
    while i < len(xs) - 1:
        if _segment_in_lens((xs[i], ys[i]), (xs[i + 1], ys[i + 1]),
                            center, radius):
            start_idx = i
            while i < len(xs) - 1 and _segment_in_lens(
                    (xs[i], ys[i]), (xs[i + 1], ys[i + 1]),
                    center, radius):
                i += 1
            end_idx = i

            sub_xs = xs[start_idx:end_idx + 1]
            sub_ys = ys[start_idx:end_idx + 1]
            rel_xs = [sub_xs[0]]
            rel_ys = [sub_ys[0]]

            for j in range(1, len(sub_xs)):
                pa = (rel_xs[j - 1], rel_ys[j - 1])
                pb = (sub_xs[j], sub_ys[j])
                angle = _segment_angle(pa, pb)
                target_angle = _snap_perpendicular(angle)
                diff = (target_angle - angle + math.pi) % (2 * math.pi) - math.pi
                relaxed_angle = angle + diff * strength
                length = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
                rel_xs.append(pa[0] + length * math.cos(relaxed_angle))
                rel_ys.append(pa[1] + length * math.sin(relaxed_angle))

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


def _ensure_edge_lengths(G: nx.Graph) -> None:
    pos = nx.get_node_attributes(G, "pos")
    for u, v in G.edges:
        if "length" not in G.edges[u, v]:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            G.edges[u, v]["length"] = math.hypot(x1 - x2, y1 - y2)


# ---------------------------------------------------------------------------
# Ink / clutter metrics
# ---------------------------------------------------------------------------
def compute_ink_metrics(G: nx.Graph, bundle: Dict) -> Dict[str, float]:
    """Return ink-based clutter measures for the bundled drawing."""
    _ensure_edge_lengths(G)

    raw_ink = sum(G.edges[u, v]["length"] for u, v in G.edges)

    cumulative_ink = 0.0
    detours: List[float] = []
    used_underlying: set = set()
    bundled_keys = {_norm(e) for e in bundle}

    for e in G.edges:
        e_n = _norm(e)
        direct_len = G.edges[e]["length"]
        if e_n in bundled_keys:
            path_nodes = bundle[e_n]
            path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
            path_len = sum(G.edges[m]["length"] for m in path_edges)
            cumulative_ink += path_len
            detours.append(path_len / direct_len if direct_len > 0 else 1.0)
            for m in path_edges:
                used_underlying.add(_norm(m))
        else:
            cumulative_ink += direct_len
            used_underlying.add(e_n)

    unique_ink = sum(G.edges[u, v]["length"] for u, v in G.edges
                     if _norm((u, v)) in used_underlying)

    return {
        "raw_ink":          raw_ink,
        "cumulative_ink":   cumulative_ink,
        "unique_ink":       unique_ink,
        "ink_ratio":        unique_ink / raw_ink if raw_ink else float("nan"),
        "cumulative_ratio": cumulative_ink / raw_ink if raw_ink else float("nan"),
        "n_total_edges":    G.number_of_edges(),
        "n_bundled_edges":  len(bundle),
        "bundled_fraction": len(bundle) / G.number_of_edges() if G.number_of_edges() else 0.0,
        "mean_detour":      mean(detours) if detours else float("nan"),
        "median_detour":    median(detours) if detours else float("nan"),
        "max_detour":       max(detours) if detours else float("nan"),
    }


# ---------------------------------------------------------------------------
# Lens distortion metrics
# ---------------------------------------------------------------------------
def _bundle_polyline(G: nx.Graph, bundle: Dict, edge: Tuple,
                     bundle_strength: float) -> Tuple[List[float], List[float]]:
    """Reconstruct the same polyline drawv2 would render for `edge`."""
    pos = nx.get_node_attributes(G, "pos")
    e_n = _norm(edge)
    if e_n not in bundle:
        s, t = edge
        return [pos[s][0], pos[t][0]], [pos[s][1], pos[t][1]]

    path = bundle[e_n]
    bx = [pos[n][0] for n in path]
    by = [pos[n][1] for n in path]

    # Same blend used in drawv2.update_plot
    n = len(path)
    if n <= 1:
        return bx, by
    t = [i / (n - 1) for i in range(n)]
    sx = [(1 - tt) * bx[0] + tt * bx[-1] for tt in t]
    sy = [(1 - tt) * by[0] + tt * by[-1] for tt in t]
    xs = [(1 - bundle_strength) * sx[i] + bundle_strength * bx[i] for i in range(n)]
    ys = [(1 - bundle_strength) * sy[i] + bundle_strength * by[i] for i in range(n)]
    return xs, ys


def sample_lens_centers(G: nx.Graph, n: int, radius: float,
                        seed: int = 0) -> List[Tuple[float, float]]:
    """Pick `n` lens centres at locations where the lens will actually
    intersect at least one bundled edge segment.

    Centres are sampled by jittering randomly chosen node positions; this
    keeps the lens placements representative of the dense regions a user
    would actually probe.
    """
    rng = random.Random(seed)
    pos = nx.get_node_attributes(G, "pos")
    nodes = list(pos)
    centres = []
    while len(centres) < n and nodes:
        node = rng.choice(nodes)
        x, y = pos[node]
        # small jitter so we don't always land on a node
        cx = x + rng.uniform(-radius / 2, radius / 2)
        cy = y + rng.uniform(-radius / 2, radius / 2)
        centres.append((cx, cy))
    return centres


def compute_lens_distortion_metrics(
        G: nx.Graph,
        bundle: Dict,
        bundle_strength: float,
        snap_strength: float,
        lens_radius: float,
        lens_centers: Sequence[Tuple[float, float]],
) -> Dict[str, float]:
    """Quantify how much the perpendicular-lens transform displaces points
    and stretches segments.

    For each lens centre, every bundled polyline that touches the lens is
    replayed through `_relax_path_in_lens`. We collect:

      - displacement of each point that was inside the lens;
      - absolute length change of each segment that straddles the lens
        boundary (i.e. one endpoint inside, one outside);
      - residual drift at the boundary join after the relax routine's
        own error-correction pass.
    """
    _ensure_edge_lengths(G)

    displacements: List[float] = []
    boundary_length_changes: List[float] = []
    boundary_drifts: List[float] = []

    for cx, cy in lens_centers:
        for edge in bundle:
            xs0, ys0 = _bundle_polyline(G, bundle, edge, bundle_strength)
            xs1, ys1 = _relax_path_in_lens(xs0, ys0, (cx, cy),
                                           lens_radius, strength=snap_strength)

            inside = [math.hypot(x - cx, y - cy) < lens_radius
                      for x, y in zip(xs0, ys0)]

            if not any(inside):
                continue

            for i, was_in in enumerate(inside):
                if was_in:
                    d = math.hypot(xs1[i] - xs0[i], ys1[i] - ys0[i])
                    displacements.append(d)

            for i in range(len(xs0) - 1):
                if inside[i] != inside[i + 1]:
                    L0 = math.hypot(xs0[i + 1] - xs0[i], ys0[i + 1] - ys0[i])
                    L1 = math.hypot(xs1[i + 1] - xs1[i], ys1[i + 1] - ys1[i])
                    boundary_length_changes.append(abs(L1 - L0))
                    # drift at the join: how far the relaxed end-of-section
                    # vertex sits from where the un-relaxed path left it
                    boundary_drifts.append(
                        math.hypot(xs1[i] - xs0[i], ys1[i] - ys0[i])
                        if not inside[i + 1] else
                        math.hypot(xs1[i + 1] - xs0[i + 1],
                                   ys1[i + 1] - ys0[i + 1])
                    )

    def _p95(seq):
        if not seq:
            return float("nan")
        s = sorted(seq)
        idx = max(0, int(round(0.95 * (len(s) - 1))))
        return s[idx]

    return {
        "n_lens_placements":       len(lens_centers),
        "n_points_evaluated":      len(displacements),
        "mean_displacement":       mean(displacements) if displacements else float("nan"),
        "median_displacement":     median(displacements) if displacements else float("nan"),
        "p95_displacement":        _p95(displacements),
        "max_displacement":        max(displacements) if displacements else float("nan"),
        "mean_length_change":      mean(boundary_length_changes) if boundary_length_changes else float("nan"),
        "max_length_change":       max(boundary_length_changes) if boundary_length_changes else float("nan"),
        "mean_boundary_drift":     mean(boundary_drifts) if boundary_drifts else float("nan"),
        "max_boundary_drift":      max(boundary_drifts) if boundary_drifts else float("nan"),
        "displacement_to_radius":  (mean(displacements) / lens_radius) if displacements else float("nan"),
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------
def format_table(name: str, ink: Dict, lens: Dict) -> str:
    lines = [f"== {name} ==",
             "  -- Ink / clutter --"]
    for k in ("raw_ink", "cumulative_ink", "unique_ink",
              "ink_ratio", "cumulative_ratio",
              "n_total_edges", "n_bundled_edges", "bundled_fraction",
              "mean_detour", "median_detour", "max_detour"):
        v = ink[k]
        lines.append(f"    {k:20s} {v:>10.4f}" if isinstance(v, float)
                     else f"    {k:20s} {v:>10}")
    lines.append("  -- Lens distortion --")
    for k in ("n_lens_placements", "n_points_evaluated",
              "mean_displacement", "median_displacement",
              "p95_displacement", "max_displacement",
              "mean_length_change", "max_length_change",
              "mean_boundary_drift", "max_boundary_drift",
              "displacement_to_radius"):
        v = lens[k]
        lines.append(f"    {k:24s} {v:>10.4f}" if isinstance(v, float)
                     else f"    {k:24s} {v:>10}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------
def evaluate(G: nx.Graph,
             k: float = 2.0,
             d: float = 2.0,
             bundle_strength: float = 0.75,
             snap_strength: float = 0.7,
             lens_radius: float = 25.0,
             n_lens_samples: int = 30,
             dataset_name: str = "dataset",
             seed: int = 0) -> Dict[str, Dict]:
    """One-shot helper: bundle, sample lens centres, compute everything,
    and print a summary table."""
    from edge_bundling import edge_path_bundling   # local import on purpose

    _ensure_edge_lengths(G)
    bundle = edge_path_bundling(G, k=k, d=d)

    ink = compute_ink_metrics(G, bundle)
    centres = sample_lens_centers(G, n=n_lens_samples,
                                  radius=lens_radius, seed=seed)
    lens = compute_lens_distortion_metrics(
        G, bundle,
        bundle_strength=bundle_strength,
        snap_strength=snap_strength,
        lens_radius=lens_radius,
        lens_centers=centres,
    )

    print(format_table(dataset_name, ink, lens))
    return {"ink": ink, "lens": lens}


if __name__ == "__main__":
    raise SystemExit(
        "Import this module from your own script that has loaded G, e.g.:\n"
        "    from metrics import evaluate\n"
        "    evaluate(G, dataset_name='Airlines')"
    )