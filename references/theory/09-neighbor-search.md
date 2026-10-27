# 09 · Neighbour search and in-browser cost

## Summary
Brute force computes every distance: no build step, $O(nd)$ per query, and it works with any distance function. KD-trees and ball trees prune whole regions using the triangle inequality, which pays off at low dimension, few neighbours and many queries. For the app, the expensive step is the decision-boundary grid. Budget it explicitly.

## Math

| Method | Build | Query (one point) | Needs a true metric? | Wins when |
|---|---|---|---|---|
| Brute force | none (store $X$) | $O(DN)$ | no (any function, incl. $p<1$, cosine) | small $N$, large $D$, $k\gtrsim N/2$, few queries |
| KD-tree | "very fast" (axis splits, no $D$-dim distances). Wikipedia: $O(n\log n)$ with linear-time median, $O(n\log^2 n)$ with sorting | ≈$O(D\log N)$ for $D\lesssim20$, approaching $O(DN)$ for large $D$ | yes; scikit-learn: euclidean, manhattan, chebyshev, minkowski only | low $D$, many queries |
| Ball tree | costlier than KD-tree (scikit-learn gives no O-bound) | ≈$O(D\log N)$ | yes (pruning uses the triangle inequality) | structured or manifold data, moderate $D$, non-Minkowski metrics |

$N$ = training rows, $D$ = features (scikit-learn notation). Quoted bounds are from the scikit-learn user guide unless marked.

**Pruning rules** (Cornell CS4780 note 16). Let $d_{\text{best}}$ be the distance to the current $k$-th best candidate.
- KD-tree: skip the far side of a split if the distance $d_w$ to the splitting wall satisfies $d_w>d_{\text{best}}$.
- Ball tree: skip a ball with centre $\mathbf{c}$ and radius $\rho$ if $\delta(\mathbf{x}_q,\mathbf{c})-\rho\ge d_{\text{best}}$. Both rules depend on the triangle inequality.

**scikit-learn `algorithm` parameter**: `'auto'` (default), `'ball_tree'`, `'kd_tree'`, `'brute'`. `'auto'` picks `'brute'` if any of these hold:
- input is sparse;
- `metric='precomputed'`;
- $D>15$;
- $k\ge N/2$;
- the metric is not valid for either tree.

Otherwise it picks the first of kd_tree, ball_tree that supports the metric. The source also forces brute for Minkowski $p<1$ and picks ball_tree for weighted Minkowski. `leaf_size` (default 30) trades build time, memory and query overhead. Near 1, node-traversal overhead slows queries. As it approaches $N$, a query becomes essentially brute force.

**In-browser back-of-envelope (boundary map).** Brute-force cost of a $G\times G$ prediction grid:

$$\text{coordinate ops}\approx G^2\cdot n\cdot d,\qquad\text{distance evaluations}=G^2\cdot n,\qquad\text{selection}\approx G^2\cdot n\ \text{comparisons}.$$

With $n=5000$, $d=10$, $G=200$:

| Quantity | Value |
|---|---|
| grid cells $G^2$ | 40,000 |
| distance evaluations $G^2 n$ | $2\times10^{8}$ |
| coordinate ops $G^2 n d$ (each = subtract, abs/square, add) | $2\times10^{9}$ (~$6\times10^9$ flops) |
| top-$k$ selection work | ~$2\times10^{8}$ comparisons (more with a heap: $\times\log k$) |

At an **assumed** single-thread JavaScript throughput of $10^8$–$10^9$ simple float ops/s (unverified; benchmark on target devices), $6\times10^9$ flops takes roughly 6 to 60 s. That is far too slow for live slider updates.

**Reductions**
1. **2D slice trick.** A boundary map with $d>2$ is a slice in features $(a,b)$ with the others fixed at $\mathbf{x}^{\text{ref}}$. For Minkowski $p<\infty$, precompute per training row $c_i=\sum_{j\notin\{a,b\}}|x^{\text{ref}}_j-x_{ij}|^p$ once ($n\,d$ ops). Then $$\delta(\mathbf{g},\mathbf{x}_i)^p=c_i+|g_a-x_{ia}|^p+|g_b-x_{ib}|^p,$$ so the per-cell cost drops to $O(n)$ with 2 coordinates: $4\times10^8$ coordinate ops in the example. For L∞ use $c_i=\max_{j\notin\{a,b\}}|\cdot|$ and a max. Rank by $\delta^p$ without taking the root (note 02).
2. **Caching.** Distances do not depend on $k$ or the weighting scheme. Cache each cell's sorted top-$k_{\max}$ list once (40,000 × 50 Int32 indices ≈ 8 MB, plus distances). Then $k$ and weighting sliders only re-vote, which costs $O(G^2 k_{\max})$. Metric, scaling, encoding and slice changes force recomputation.
3. **Progressive / adaptive grids.** Render $G=50$ first (2,500 cells, $1.25\times10^7$ distances), then refine to 200. Or subdivide only cells whose corner labels disagree, so the boundary gets resolution and flat regions don't.
4. **Parallelism.** Use Web Workers (split rows of the grid), or a GPU fragment/compute shader with one pixel per cell looping over $n$ training rows.
5. **Index in the slice.** The $c_i$ offsets break a plain 2D KD-tree, though a tree over $(x_a,x_b)$ can still prune with the lower bound $\delta^p\ge c_{\min}+\ldots$. That is probably not worth the complexity at $n\le10^4$.
6. **Subsample for the map only**, and label it as such, while predictions and metrics still use all rows.

## Geometric meaning
- **Brute force:** lines from the query to every point. All are measured.
- **KD-tree:** axis-aligned boxes nested by alternating splits. The search descends to the query's box, then visits neighbouring boxes only if the current neighbour ball crosses their wall.
- **Ball tree:** nested circles. A whole circle is skipped when it lies entirely outside the current neighbour ball.
- **High $D$:** the neighbour ball is nearly as wide as the data (note 07), so it crosses almost every wall or ball and pruning fails. Cost falls back to brute force plus overhead.

## Effect on predictions
None in principle: all three methods return the exact same $k$ nearest rows, **except when distances tie at rank $k$**, where the method and the row order can pick different rows (note 04). Speed is the only intended difference.

## Pitfalls & edge cases
- **Tree methods need a true metric:** no $p<1$, no cosine (cosine is brute-only in scikit-learn), and no arbitrary similarity scores.
- **Re-standardising, re-encoding or changing the metric** invalidates any tree built on the old features.
- **Floating-point:** computing L2 as $\lVert a\rVert^2+\lVert b\rVert^2-2a\cdot b$ (fast matrix form) can give tiny negative values or nonzero self-distances. Exact 0 for duplicates matters for the zero-distance weighting rule (note 04). Prefer direct differences in the app.
- **Large $k$:** pruning weakens as $k$ approaches $N$. scikit-learn's auto rule switches to brute at $k\ge N/2$.

## Visualization ideas
- **Search race:** brute vs KD-tree side by side on the same query. Visited points flash and pruned boxes grey out. Counters show "distances computed: 5000 vs 143". A $D$ slider (adding noise dims) shows the tree's count climbing toward $N$.
- **KD-tree build animation:** recursive median splits drawn as lines, alternating axes, with leaf boxes holding at most `leaf_size` points.
- **Ball-tree pruning:** the query's current neighbour circle against nested balls. When $\delta(\mathbf{x}_q,\mathbf{c})-\rho\ge d_{\text{best}}$, the ball fades, and the inequality is shown with live numbers colour-linked to the drawn radius and centre distance.
- **Cost meter for the boundary map:** "$G^2\cdot n\cdot d$ = 2.0 × 10⁹ ops" updates as the user changes $G$ or uploads data, with an estimated time from a one-off micro-benchmark run at page load.

## Sources
- scikit-learn neighbors user guide (1.9.1), https://scikit-learn.org/stable/modules/neighbors.html. Supports: brute $O[DN]$ query and $O[DN^2]$ all-pairs; KD-tree build "very fast", $O[\log N]$ distance computations, $D<20$ remark, $O[D\log N]$ to $O[DN]$; ball tree costlier build, $O[D\log N]$ query; auto rules (verbatim list); leaf_size effects; "If very few query points will be required, brute force is better".
- scikit-learn KNeighborsClassifier API. Supports: `algorithm` values, `leaf_size=30`, sparse input forces brute.
- scikit-learn source `_base.py`, `_kd_tree.pyx.tp`, `_ball_tree.pyx.tp`. Supports: auto selection code ($D>15$, $k\ge N/2$, $p<1$ → brute, weighted Minkowski → ball_tree), VALID_METRICS per tree.
- Cornell CS4780 lecture note 16, https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote16.html. Supports: KD-tree wall pruning and its inequality, ball-tree pruning via triangle inequality, KD-trees fail in high $D$, $O(dn)$ brute cost.
- Wikipedia "k-d tree", https://en.wikipedia.org/wiki/K-d_tree. Supports: build $O(n\log n)$ / $O(n\log^2 n)$, average $O(\log n)$ NN query for random points, degrades toward linear, high-$D$ no better than exhaustive search.
- Wikipedia "Ball tree", https://en.wikipedia.org/wiki/Ball_tree. Supports: ball definition, k-d construction algorithm, pruning when distance(t, pivot) − radius ≥ current k-th distance, Omohundro 1989 reference.
- JavaScript throughput figure: **assumption, unverified**. All op counts are computed, not sourced.
