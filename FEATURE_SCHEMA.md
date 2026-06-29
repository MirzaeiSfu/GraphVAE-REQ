# Optimal Feature Schema — Grid, Triangular Grid, Lobster

This document explains every node/edge feature used by the "optimal" schema
(`factorbase_motif_pipeline/best_grid.py`, `best_triangular_grid.py`,
`best_lobster.py`, and the matching loaders in `data.py` /
`dataset_feature_utils/`). For each feature: what it measures, a concrete
worked example, and the **actual percentage of each category** measured
across the same 100-graph dataset both `data.py` and `best_*.py` generate
(grid/triangular grid: `range(10, 20)` × `range(10, 20)` = 100 graphs;
lobster: `nx.random_lobster(80, 0.7, 0.7)`, seed 1234, filtered to 10–100
nodes, 100 graphs).

Percentages were measured directly from the MySQL databases created by
`best_grid.py`/`best_triangular_grid.py`/`best_lobster.py` (`grid_optimal_features`,
`triangular_grid_optimal_features`, `lobster_optimal_features`), which use
the identical generation parameters and formulas as `data.py`'s loaders.

---

## Grid

100 graphs, 21,025 nodes, 78,300 directed edge-rows (39,150 undirected edges,
each stored twice).

### Node feature: `distance_to_boundary` (1 value, 5 categories)

How far a node is from the nearest of the grid's 4 edges, computed
**per-axis** (`min(row, width-1-row, col, height-1-col)`) — this is the
corrected version of the old formula, which used a single
`grid_size = max(width, height)` for both axes and over-estimated distance
on the shorter axis of a non-square grid.

| Value | Label | Meaning | % of nodes |
|---|---|---|---|
| 1 | Boundary | distance 0 (sits on the outer edge) | 25.68% |
| 2 | Near-Boundary | distance 1 | 21.88% |
| 3 | Near-Center | distance 2–3 | 32.34% |
| 4 | Center | distance 4–5 | 16.37% |
| 5 | Deep-Center | distance > 5 | 3.73% |

**Example**: in a 14×14 grid (rows/cols 0–13), node `(0, 5)` sits in row 0 →
distance 0 → category **1 (Boundary)**. Node `(6, 7)` (near the middle) →
`min(6, 7, 7, 6) = 6` → category **5 (Deep-Center)**, since 6 > 5.

### Edge feature 1: `edge_axis` (2 categories)

Whether the edge runs along a row (horizontal) or a column (vertical).

| Value | Label | % of edges |
|---|---|---|
| 1 | Horizontal | 50.00% |
| 2 | Vertical | 50.00% |

Exactly 50/50 by construction — a square grid is symmetric under 90°
rotation, so orientation carries no information on its own (this is why it
showed zero mutual information with anything else when we checked the
learned FactorBase model — it's still useful as raw context for other
features, just not "interesting" on its own).

**Example**: edge `(3,4)–(3,5)` (same row, adjacent columns) → **Horizontal**.
Edge `(3,4)–(4,4)` (same column, adjacent rows) → **Vertical**.

### Edge feature 2: `edge_square_count` (2 categories)

How many unit grid-squares (4-cycles) the edge is a side of. This is the
same quantity as the "square count" / 4-cycle motif already tracked
elsewhere in the eval pipeline (`graph_statistics.py`).

| Value | Label | Meaning | % of edges |
|---|---|---|---|
| 1 | One-Square | boundary edge, touches 1 square | 13.79% |
| 2 | Two-Squares | interior edge, touches 2 squares | 86.21% |

**Example**: edge `(0,4)–(0,5)` is on the top row (boundary) → touches only
1 unit square → **One-Square**. Edge `(5,4)–(5,5)` (interior row) → touches
2 squares → **Two-Squares**.

### Edge feature 3: `edge_boundary_band` (5 categories, same scale as `distance_to_boundary`)

`min(distance_to_boundary(u), distance_to_boundary(v))` for the edge's two
endpoints — lets the relational structure relate orientation/square-count to
*where in the grid* the edge sits.

| Value | Label | % of edges |
|---|---|---|
| 1 | Boundary | 26.56% |
| 2 | Near-Boundary | 22.48% |
| 3 | Near-Center | 32.69% |
| 4 | Center | 15.26% |
| 5 | Deep-Center | 3.00% |

**Example**: edge `(0,4)–(0,5)` — both endpoints have `distance_to_boundary = 1`
(Boundary) → `edge_boundary_band = 1` (Boundary).

> Note: `edge_boundary_band` and `edge_square_count` are both fully
> determined by `distance_to_boundary`/position — that's intentional (they
> let the relational model query position-derived facts directly as edge
> attributes), but it means they carry no information *independent* of
> `distance_to_boundary` itself.

---

## Triangular Grid

100 graphs, 12,800 nodes, 67,450 directed edge-rows (33,725 undirected edges).

### Node feature 1: `distance_to_boundary` (5 categories — but only 4 ever occur)

Same idea as grid's version, using the lattice's actual row/col bounds.

| Value | Label | % of nodes |
|---|---|---|
| 1 | Boundary | 31.45% |
| 2 | Near-Boundary | 28.13% |
| 3 | Near-Center | 35.16% |
| 4 | Center | 5.27% |
| 5 | Deep-Center | **0% — never occurs at this dataset's scale (width/height 10–19)** |

**Example**: a corner vertex of the lattice → distance 0 → **Boundary**.

### Node feature 2: `num_3cycles` (real triangle count, 1-based)

Number of triangles the node participates in, plus 1 (so the stored value
is always ≥ 1). Ties to the well-known "triangle count" motif.

| Value | Raw triangles | % of nodes |
|---|---|---|
| 2 | 1 | 1.95% |
| 3 | 2 | 11.52% |
| 4 | 3 | 9.96% |
| 5 | 4 | 10.74% |
| 7 | 6 | 65.82% |

(Values 1 and 6 — i.e. 0 and 5 raw triangles — don't occur at this scale.)

**Example**: a fully-interior node (degree 6, every consecutive neighbor pair
connected) participates in 6 triangles → stored as **7**.

### Node feature 3: `num_hexagons` (real induced-6-cycle count, 1-based)

Number of *actual* induced 6-cycles (hexagons) the node belongs to, found by
exhaustive cycle search, plus 1. This replaces the old schema's
`num_6cycles`, which was a fake `degree >= 4` proxy, not a real hexagon
detector (verified by querying the old learned FactorBase model: the old
`num_6cycles` was 100% determined by degree, carrying zero independent
information).

| Value | Raw hexagons | % of nodes |
|---|---|---|
| 1 | 0 | 1.95% |
| 2 | 1 | 13.48% |
| 3 | 2 | 11.13% |
| 4 | 3 | 19.92% |
| 5 | 4 | 6.45% |
| 6 | 5 | 8.79% |
| 7 | 6 | 38.28% |

**Example**: a node at the center of the lattice, surrounded by 6 complete
hexagonal "flower petals" of triangles, belongs to 6 hexagons → stored as **7**.

### Edge feature 1: `edge_direction` (3 categories)

Which of the triangular lattice's 3 axes the edge runs along.

| Value | Label | % of edges |
|---|---|---|
| 1 | Horizontal | 33.36% |
| 2 | Positive-60 | 33.36% |
| 3 | Negative-60 | 33.28% |

Almost perfectly uniform (1/3 each) — the triangular lattice is symmetric
across its 3 axes.

### Edge feature 2: `edge_hexagons` (real induced-6-cycle edge participation, 1-based)

How many hexagons the edge is a side of, plus 1.

| Value | Raw hexagons | % of edges |
|---|---|---|
| 1 | 0 | 6.60% |
| 2 | 1 | 36.92% |
| 3 | 2 | 56.49% |

**Example**: an edge shared by two adjacent hexagonal flowers → 2 hexagons →
stored as **3**.

### Edge feature 3: `edge_triangle_count` (real value, 0/1/2 — not 1-based)

Literal number of triangles the edge participates in (count of common
neighbors of its two endpoints).

| Value | Meaning | % of edges |
|---|---|---|
| 0 | Zero-Triangles | **0% at this scale** |
| 1 | One-Triangle | 12.97% |
| 2 | Two-Triangles | 87.03% |

**Example**: a fully interior edge borders 2 triangles (one on each side) →
**2**. A boundary edge borders only 1 → **1**.

---

## Lobster

100 graphs, 5,362 nodes, 10,524 directed edge-rows (5,262 undirected edges).
Generated with `nx.random_lobster(80, 0.7, 0.7)`, seed 1234, filtered to
10–100 nodes per graph.

### Node feature 1: `node_degree` (4 categories)

| Value | Label | Degree | % of nodes |
|---|---|---|---|
| 1 | Leaf | 1 | 69.27% |
| 2 | Branch | 2–3 | 16.15% |
| 3 | Hub | 4–5 | 7.76% |
| 4 | SuperHub | 6+ | 6.83% |

**Example**: a literal tree leaf (one neighbor) → **Leaf**. A node with 3
neighbors → **Branch**.

### Node feature 2: `spine_role` (4 categories)

The lobster's "spine" is approximated as the tree's diameter path. This
feature replaces the old `distance_to_spine`, whose "Far-Spine" (>3 hops)
category never occurs at this generation scale — that dead category's slot
is reused here to flag the two structurally-special spine *endpoints*
separately from internal spine nodes.

| Value | Label | Meaning | % of nodes |
|---|---|---|---|
| 1 | Spine-Endpoint | one of the 2 tips of the spine path | 3.73% |
| 2 | Spine-Internal | on the spine, not a tip | 15.55% |
| 3 | Near-Spine | 1 hop off the spine | 31.15% |
| 4 | Off-Spine | 2+ hops off the spine | 49.57% |

**Example**: the very first node of the spine path → **Spine-Endpoint**. A
leaf hanging off a branch two hops from the spine → **Off-Spine**.

### Node feature 3: `subtree_size` (3 categories)

Size of the branch-component attached at each spine point (computed by
removing spine edges and measuring the resulting connected components).

| Value | Bucket | % of nodes |
|---|---|---|
| 1 | 1–5 nodes | 23.78% |
| 2 | 6–20 nodes | 54.81% |
| 3 | 21+ nodes | 21.41% |

### Node feature 4: `eccentricity` (3 categories)

Longest shortest-path distance from the node to any other node in its graph.

| Value | Bucket | % of nodes |
|---|---|---|
| 1 | 1–5 | 12.01% |
| 2 | 6–10 | 62.14% |
| 3 | 11+ | 25.85% |

### Edge feature 1: `edge_type` (3 categories)

| Value | Label | Meaning | % of edges |
|---|---|---|---|
| 1 | Spine-Edge | both endpoints on the spine | 17.75% |
| 2 | Branch-Edge | exactly one endpoint on the spine | 31.74% |
| 3 | Leaf-Edge | neither endpoint on the spine | 50.51% |

**Example**: an edge connecting two consecutive spine nodes → **Spine-Edge**.
An edge from a spine node out to its first branch → **Branch-Edge**.

### Edge feature 2: `depth_pair` (6 categories)

A genuinely *relational* feature: pairs the two endpoints' spine-distance
(capped at 2 hops) and encodes the sorted pair. Lets the model query "what
kind of spine-relationship does this edge connect" directly.

| Value | Label | Endpoint depths | % of edges |
|---|---|---|---|
| 1 | Spine-Spine | (0,0) | 17.75% |
| 2 | Spine-Branch | (0,1) | 31.74% |
| 3 | Branch-Leaf | (1,2) | 50.51% |
| 4 | Branch-Branch | (1,1) | 0%* |
| 5 | Spine-Leaf | (0,2) | 0%* |
| 6 | Leaf-Leaf | (2,2) | 0%* |

\* Categories 4–6 are structurally impossible in a lobster tree under this
capped-distance definition at this dataset's branching parameters — every
observed edge happens to connect adjacent depth levels, never two nodes both
1+ hops off the spine. (Note: in this run, `depth_pair`'s observed
distribution is identical to `edge_type`'s — both end up encoding the same
spine-adjacency information for this dataset's branching pattern.)

**Example**: an edge from a spine node to its directly-attached branch node
→ depths (0,1) → **Spine-Branch**.

### Edge feature 3: `terminal_edge` (2 categories)

Whether the edge touches at least one Leaf (degree-1) node — replaces the
old `endpoint_degree_pair` (21 categories, mostly redundant with `node_degree`
and too sparse for this dataset's size).

| Value | Label | % of edges |
|---|---|---|
| 1 | Non-Terminal | 29.42% |
| 2 | Terminal (touches a Leaf) | 70.58% |

**Example**: the edge connecting a branch node to its leaf tip → **Terminal**.
An edge between two internal spine nodes → **Non-Terminal**.

---

## Where these numbers come from

```
mysql -ufbuser grid_optimal_features            -e "SELECT <col>, COUNT(*) FROM nodes/edges GROUP BY <col>;"
mysql -ufbuser triangular_grid_optimal_features  -e "..."
mysql -ufbuser lobster_optimal_features          -e "..."
```

These databases were created by `best_grid.py`/`best_triangular_grid.py`/
`best_lobster.py` and use the exact same generation parameters and feature
formulas as `data.py`'s `_build_grid_graph_features`/
`_build_triangular_grid_graph_features`/`_build_lobster_graph_features` — so
the percentages above describe the actual training-data distribution
`main.py` will see.
