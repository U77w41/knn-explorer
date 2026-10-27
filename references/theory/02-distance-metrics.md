# 02 · Distance metrics

## Summary
The distance function decides which training rows count as "near", so changing it can change the neighbour set and the prediction even when the data stay the same. The Minkowski family (p = 1, 2, ∞) covers most use, p < 1 breaks the triangle inequality, and cosine, Mahalanobis, Hamming and Gower handle direction, correlation, binary and mixed-type data.

## Math

**Metric axioms** (Wikipedia "Metric space"). For all $\mathbf{x},\mathbf{y},\mathbf{z}$:
1. $\delta(\mathbf{x},\mathbf{x})=0$
2. Positivity: $\mathbf{x}\ne\mathbf{y}\Rightarrow\delta(\mathbf{x},\mathbf{y})>0$
3. Symmetry: $\delta(\mathbf{x},\mathbf{y})=\delta(\mathbf{y},\mathbf{x})$
4. Triangle inequality: $\delta(\mathbf{x},\mathbf{z})\le\delta(\mathbf{x},\mathbf{y})+\delta(\mathbf{y},\mathbf{z})$

KD-trees and ball trees rely on axiom 4 for pruning (note 09).

**Minkowski / $L_p$.** For $\mathbf{x},\mathbf{y}\in\mathbb{R}^d$ and $p>0$:

$$D_p(\mathbf{x},\mathbf{y})=\Big(\sum_{j=1}^{d}|x_j-y_j|^p\Big)^{1/p}.$$

- $p=1$: Manhattan / city-block, $\sum_j|x_j-y_j|$.
- $p=2$: Euclidean, $\sqrt{\sum_j(x_j-y_j)^2}$.
- $p\to\infty$: Chebyshev, $\max_j|x_j-y_j|$.
- $p\ge1$ gives a metric (Minkowski inequality).
- For a fixed pair, $D_p$ does not increase as $p$ grows: $\lVert\mathbf{v}\rVert_{p+a}\le\lVert\mathbf{v}\rVert_p$ for $a\ge0$ (Lp space article, which says this also holds for $0<p<1$).

**Fractional $0<p<1$ is not a metric.** Counterexample (Wikipedia, numbers checked): $\mathbf{a}=(0,0)$, $\mathbf{b}=(0,1)$, $\mathbf{c}=(1,1)$, $p=0.5$:

$$D_{0.5}(\mathbf{a},\mathbf{c})=(1^{0.5}+1^{0.5})^{2}=4\ >\ D_{0.5}(\mathbf{a},\mathbf{b})+D_{0.5}(\mathbf{b},\mathbf{c})=1+1=2.$$

Dropping the outer exponent, $d_p=\sum_j|x_j-y_j|^p$, gives a metric for $0<p<1$, but not a norm: its unit ball is not convex. The outer power $1/p$ is monotone, so $D_p$ and $\sum_j|x_j-y_j|^p$ rank neighbours identically. Neighbour sets match, but distance-weighted votes differ. In the same way, squared Euclidean gives the same neighbours as Euclidean.

**Unit balls in 2D.** $\{(u,v): |u|^p+|v|^p\le1\}$, boundary $|u|^p+|v|^p=1$:
- $p=1$: diamond (square rotated 45°, vertices at $(\pm1,0),(0,\pm1)$).
- $p=2$: circle.
- $p\to\infty$: axis-aligned square $[-1,1]^2$.
- $1<p<2$ and $p>2$: superellipses ("squircles" for large $p$).
- $p=2/3$: the astroid. $p<1$ generally: concave four-pointed star with cusps on the axes.

Since $D_p$ shrinks as $p$ grows, the balls are nested: diamond ⊂ circle ⊂ square.

**Cosine.** Similarity $S_C(\mathbf{a},\mathbf{b})=\dfrac{\mathbf{a}\cdot\mathbf{b}}{\lVert\mathbf{a}\rVert\,\lVert\mathbf{b}\rVert}$, distance $D_C=1-S_C$. If $\lVert\mathbf{a}\rVert=\lVert\mathbf{b}\rVert=1$:

$$\lVert\mathbf{a}-\mathbf{b}\rVert_2^2=2\,(1-S_C)\quad\Rightarrow\quad D_C=\tfrac12\lVert\mathbf{a}-\mathbf{b}\rVert_2^2 .$$

So Euclidean distance on L2-normalised rows gives the same neighbour order as cosine distance. $D_C$ is not a metric: it fails the triangle inequality, and $D_C(\mathbf{a},2\mathbf{a})=0$. It is undefined for a zero vector.

**Mahalanobis.** With covariance $\Sigma$ (estimated on the training data):

$$d_M(\mathbf{x},\mathbf{y})=\sqrt{(\mathbf{x}-\mathbf{y})^\top\Sigma^{-1}(\mathbf{x}-\mathbf{y})}=\big\lVert\Sigma^{-1/2}(\mathbf{x}-\mathbf{y})\big\rVert_2 ,$$

where $\Sigma^{-1/2}$ is the inverse of the symmetric square root. This is Euclidean distance after whitening, $\mathbf{z}=\Sigma^{-1/2}\mathbf{x}$. If $\Sigma=\mathrm{diag}(\sigma_1^2,\dots,\sigma_d^2)$ it reduces to $\sqrt{\sum_j (x_j-y_j)^2/\sigma_j^2}$, which is z-scored Euclidean and scikit-learn's `seuclidean` with `V`=variances. NCA and LMNN learn a matrix $M=L^\top L$ in place of $\Sigma^{-1}$ (scikit-learn NCA docs; Weinberger & Saul 2009).

**Hamming.** scikit-learn: $\text{hamming}(\mathbf{x},\mathbf{y})=N_{\text{unequal}}(\mathbf{x},\mathbf{y})/N_{\text{tot}}$, i.e. the fraction of differing coordinates. On 0/1 vectors the count of differing coordinates equals both $L_1$ and $L_2^2$.

**Gower (mixed types).** Similarity between rows $i,j$ over $d$ variables:

$$S_{ij}=\frac{\sum_{k=1}^{d}w_{ijk}\,s_{ijk}}{\sum_{k=1}^{d}w_{ijk}},\qquad s_{ijk}=\begin{cases}1-\dfrac{|x_{ik}-x_{jk}|}{R_k} & \text{numeric, } R_k=\text{range of variable }k\\[4pt] \mathbb{1}[x_{ik}=x_{jk}] & \text{categorical/binary}\end{cases}$$

$w_{ijk}\ge0$ (usually 1), and distance $=1-S_{ij}\in[0,1]$. Each numeric term is range-scaled, so Gower bakes in min–max scaling.

**Worked example: the metric changes the nearest neighbour.** Query $\mathbf{q}=(0,0)$, $A=(3,0)$ (class red), $B=(2,2)$ (class blue), $k=1$.

| $p$ | $D_p(\mathbf{q},A)$ | $D_p(\mathbf{q},B)$ | 1-NN |
|---|---|---|---|
| 0.5 | 3 | 8 | A |
| 1 | 3 | 4 | A |
| 2 | 3 | 2.828 | B |
| 3 | 3 | 2.520 | B |
| ∞ | 3 | 2 | B |

The flip happens where $3=2\cdot2^{1/p}$, i.e. $p^*=1/\log_2 1.5\approx1.71$. In general, large-$p$ metrics favour "small in every coordinate" ($B$), and small-$p$ metrics favour "zero in most coordinates" ($A$).

**Metrics accepted by scikit-learn `KNeighborsClassifier`** (scikit-learn 1.9.1: `sklearn.neighbors.VALID_METRICS`, printed from an installed 1.9.1, matching the main-branch source):
- `brute`: braycurtis, canberra, chebyshev, cityblock, correlation, cosine, dice, euclidean, hamming, haversine, jaccard, l1, l2, mahalanobis, manhattan, minkowski, nan_euclidean, precomputed, rogerstanimoto, russellrao, seuclidean, sokalsneath, sqeuclidean, yule. (`sokalmichener` is only added when SciPy < 1.17.)
- `kd_tree`: chebyshev, cityblock, euclidean, infinity, l1, l2, manhattan, minkowski, p.
- `ball_tree`: braycurtis, canberra, chebyshev, cityblock, dice, euclidean, hamming, haversine, infinity, jaccard, l1, l2, mahalanobis, manhattan, minkowski, p, pyfunc, rogerstanimoto, russellrao, seuclidean, sokalmichener, sokalsneath.
- Any Python callable is also accepted (ball tree or brute; not kd_tree).
- Default `metric='minkowski'`, `p=2`. The `p` parameter must be > 0. For $0<p<1$, scikit-learn forces brute force with the warning "Mind that for 0 < p < 1, Minkowski metrics are not distance metrics", and raises an error if `kd_tree`/`ball_tree` is requested explicitly.
- `mahalanobis` takes `metric_params={'V': cov}` or `{'VI': inv_cov}`. `minkowski` accepts weights `w`: $\big(\sum_j w_j|x_j-y_j|^p\big)^{1/p}$.

## Geometric meaning
- **2D:** the metric sets the shape of the "neighbourhood ball" that grows around the query: diamond (L1), circle (L2), square (L∞), star (p<1), ellipse (Mahalanobis). The $k$ nearest points are the first $k$ that this growing shape touches.
- **Cosine:** distance depends only on direction from the origin. Points on the same ray are identical, so neighbourhoods are wedges (cones), not balls. After z-scoring the origin is the data mean, so cosine compares "direction from the average row".
- **Mahalanobis:** an ellipse aligned with the principal axes of $\Sigma$, with semi-axes $\propto\sqrt{\lambda_j}$ (eigenvalues of $\Sigma$). Points along the long axis count as closer.
- **3D:** octahedron (L1), sphere (L2), cube (L∞).
- **High-D:** the shapes drift apart. The unit L1 ball still has vertices at distance 1 on the axes, but its faces sit only $1/\sqrt d$ from the origin in L2. The unit L∞ cube's corners sit $\sqrt d$ from the origin in L2. So the metrics disagree more and more as $d$ grows (note 07 covers how contrast depends on $p$).

## Effect on predictions
- Changing $p$ changes the neighbour ranking whenever coordinate differences are uneven across features (example above). It never changes rankings in 1D, where all $L_p$ are equal.
- Small $p$ (≤ 1) lets one large coordinate difference be outweighed by many small ones. Large $p$ (→∞) is driven by the single worst coordinate.
- Cosine ignores magnitude. Two rows proportional to each other are identical.
- Mahalanobis undoes correlation and scale. On z-scored data with uncorrelated features it is nearly the same as L2.
- Boundary shape changes too (note 06): straight bisectors for L2/Mahalanobis, axis-and-diagonal polylines for L1/L∞, curves for other $p$.

## Pitfalls & edge cases
- **p < 1:** no triangle inequality, so tree search is invalid (scikit-learn falls back to brute). Weighted votes use $D_p$, not $d_p$.
- **p = ∞ on integer-valued features:** many exact ties, because the max of integer differences is an integer. L1 on integer data has the same problem, and one-hot data gives distances in steps of 1 or 2 (note 05).
- **Cosine with a zero row** (e.g. all features at the mean after z-scoring) is undefined. Decide on a fallback.
- **Mahalanobis:** $\Sigma$ is singular when $d\ge n$, when features are collinear, or with a full one-hot group (its columns always sum to 1, so their sum has zero variance). Use a pseudo-inverse or shrinkage, or drop a column. Estimate $\Sigma$ on training rows only.
- **Hamming in scikit-learn is a fraction**, not a count. It treats real values as equal/unequal only, so `hamming` on continuous columns makes every differing value cost the same.
- **Gower:** $R_k$ comes from the training data, so a test value outside that range gives $s_{ijk}<0$. Clip it or document it. Gower is not built into scikit-learn's KNN and needs a callable or precomputed matrix.
- **Wikipedia's Mahalanobis page** writes $\Sigma=S^\top S$ and $d=\lVert S^{-1}(\mathbf{x}-\mathbf{y})\rVert$. That matches the formula above when $S$ is the symmetric square root. For a non-symmetric factor (e.g. Cholesky, $\Sigma=LL^\top$) the correct whitening is $L^{-1}$.

## Visualization ideas
- **Unit-ball morph:** a $p$ slider from 0.25 to ∞ (log scale, with stops at 0.5, 1, 2, ∞) morphs the neighbourhood shape around the query in real time. The $k$-th neighbour stays highlighted, and the ball radius shrinks or grows to keep touching it. Beyond $p<1$ the shape turns concave and a "not a metric" badge appears.
- **Neighbour-flip demo:** preset the 3-point example ($\mathbf{q}$, $A$, $B$). Dragging $p$ across 1.71 swaps which point lights up, while a live table shows $D_p(\mathbf{q},A)$ and $D_p(\mathbf{q},B)$.
- **Per-coordinate decomposition:** for a selected neighbour, draw the axis-aligned "staircase" path (Δx, Δy legs) next to the straight segment. Colour-link the Δx leg to the $|x_1-y_1|^p$ term and the Δy leg to $|x_2-y_2|^p$. Bar widths show each term's share of $\sum_j|x_j-y_j|^p$. For L∞, highlight only the winning (max) leg.
- **Triangle-inequality break:** three draggable points with the edges labelled by $D_p$. Edge $ac$ turns red when $D_p(a,c)>D_p(a,b)+D_p(b,c)$, which happens whenever $p<1$ and $b$ sits at the corner.
- **Mahalanobis whitening:** two linked panels, original space with an ellipse and whitened space with a circle. Dragging a point in one moves its image in the other. Neighbour sets match in both.
- **Cosine wedges:** draw rays from the origin through each training point and shade the angular wedge containing the $k$ nearest by angle. Pair this with a "normalise rows" toggle that snaps all points onto the unit circle.

## Sources
- Wikipedia "Minkowski distance", https://en.wikipedia.org/wiki/Minkowski_distance (raw wikitext read). Supports: $D_p$ definition, metric for $p\ge1$ via Minkowski inequality, the $p<1$ counterexample $(0,0),(1,1),(0,1)$, $d_p$ without exponent being a metric with non-convex balls, Chebyshev limit.
- Wikipedia "Lp space", https://en.wikipedia.org/wiki/Lp_space (raw wikitext). Supports: astroid = unit circle for $p=2/3$, superellipse link, $\lVert x\rVert_{p+a}\le\lVert x\rVert_p$, concave unit ball and no triangle inequality for $0<p<1$.
- Wikipedia "Metric space", https://en.wikipedia.org/wiki/Metric_space (raw wikitext). Supports: axioms.
- Wikipedia "Cosine similarity", https://en.wikipedia.org/wiki/Cosine_similarity (raw wikitext). Supports: $D_C=1-S_C$, $\lVert A-B\rVert^2=2(1-S_C)$ for unit vectors, same ordering after L2 normalisation, not a metric.
- Wikipedia "Mahalanobis distance", https://en.wikipedia.org/wiki/Mahalanobis_distance (raw wikitext). Supports: definition, $\Sigma=S^\top S$, Euclidean after whitening, ellipsoid interpretation.
- Wikipedia "Gower's distance", https://en.wikipedia.org/wiki/Gower%27s_distance (raw wikitext). Supports: $S_{ij}$ formula, range-normalised numeric term, 0/1 categorical term, values in [0,1].
- scikit-learn DistanceMetric docs, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html. Supports: hamming = N_unequal/N_tot, mahalanobis args V/VI, seuclidean V, minkowski w, BallTree callable must be a true metric.
- scikit-learn KNeighborsClassifier API (1.9.1), https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html. Supports: default metric/p, p > 0.
- scikit-learn source `_base.py`, `_kd_tree.pyx.tp`, `_ball_tree.pyx.tp`, `metrics/pairwise.py` (raw.githubusercontent.com, main branch). Supports: VALID_METRICS lists, p<1 brute-only warning/error. Lists cross-checked by printing `VALID_METRICS` from an installed scikit-learn 1.9.1.
- scikit-learn NCA section, https://scikit-learn.org/stable/modules/neighbors.html. Supports: $\lVert L(x_i-x_j)\rVert^2=(x_i-x_j)^\top M(x_i-x_j)$, $M=L^\top L$.
- Weinberger & Saul 2009, local PDF. Supports: Mahalanobis metric = global linear transform before Euclidean kNN.
- Worked-example numbers and $p^*\approx1.7095$: computed locally (Python), not from a source.
