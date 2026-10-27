# 06 · Decision boundaries

## Summary
For 1-NN with Euclidean distance, the decision regions are unions of Voronoi cells, and the boundary is the set of Voronoi edges between differently labelled points. For $k>1$ the plane splits into order-$k$ Voronoi cells with a fixed neighbour set, and the boundary smooths as $k$ grows. Apps draw boundaries by evaluating the classifier on a grid, and that approach has its own resolution and tie artefacts.

## Math

**Voronoi cell of training point $\mathbf{p}_i$** (the set of locations for which $\mathbf{p}_i$ is the nearest site):

$$V_i=\{\mathbf{x}:\ \delta(\mathbf{x},\mathbf{p}_i)\le\delta(\mathbf{x},\mathbf{p}_j)\ \ \forall j\ne i\}.$$

**1-NN decision region for class $c$:** $\bigcup_{i:\,y_i=c}V_i$. The boundary is the set of shared edges $V_i\cap V_j$ with $y_i\ne y_j$. Edges between same-class cells are not boundary.

**Euclidean bisector of $\mathbf{p},\mathbf{q}$:**

$$\{\mathbf{x}:\ \lVert\mathbf{x}-\mathbf{p}\rVert_2=\lVert\mathbf{x}-\mathbf{q}\rVert_2\}=\Big\{\mathbf{x}:\ (\mathbf{q}-\mathbf{p})^\top\mathbf{x}=\tfrac12\big(\lVert\mathbf{q}\rVert^2-\lVert\mathbf{p}\rVert^2\big)\Big\}.$$

This is the line through the midpoint perpendicular to $\mathbf{pq}$. Each cell is an intersection of $n-1$ half-planes, hence a convex polygon (possibly unbounded).

**Mahalanobis bisector:** $2(\mathbf{q}-\mathbf{p})^\top\Sigma^{-1}\mathbf{x}=\mathbf{q}^\top\Sigma^{-1}\mathbf{q}-\mathbf{p}^\top\Sigma^{-1}\mathbf{p}$. Still a straight line, but perpendicular to $\mathbf{pq}$ only in whitened coordinates.

**L1 bisector** (checked numerically for $\mathbf{p}=(0,0)$, $\mathbf{q}=(4,2)$): a vertical ray $x=3$ for $y\le0$, a 45° segment from $(3,0)$ to $(1,2)$, and a vertical ray $x=1$ for $y\ge2$. In general, with $\Delta=\mathbf{q}-\mathbf{p}$:
- $|\Delta_x|>|\Delta_y|$: two rays parallel to the $y$-axis joined by a 45° segment.
- $|\Delta_y|>|\Delta_x|$: two rays parallel to the $x$-axis joined by a 45° segment.
- $|\Delta_x|=|\Delta_y|$ (points on a 45° diagonal): the equidistant set contains **two whole quarter-plane regions**, not just a line. In the numeric check with $\mathbf{q}=(2,2)$, 37% of a 14×14 window was exactly tied.

**L∞ bisector.** Using $\max(|a|,|b|)=\tfrac12(|a+b|+|a-b|)$, L∞ is L1 in coordinates rotated 45° and scaled by ½. So the shapes rotate by 45°. Checked for $(0,0),(4,2)$: a vertical segment $x=2$ for $0\le y\le2$, joined to two 45° rays through $(3,-1),(5,-3)$ and $(1,3),(-1,5)$. The degenerate case moves to **axis-aligned pairs** ($\Delta_x=0$ or $\Delta_y=0$): for $(0,0),(2,0)$, about 38% of the window was tied.

**Other $p$.** The bisector is a curve through the midpoint (checked for $p=3$). For $p<1$ the curve can fold back: in the numeric check for $p=0.5$, a horizontal line crossed it twice.

**Order-$k$ Voronoi diagram.** An order-$k$ cell is the set of locations whose $k$ nearest sites are one particular set $T$, $|T|=k$:

$$V_T=\{\mathbf{x}:\ \max_{i\in T}\delta(\mathbf{x},\mathbf{p}_i)\le\min_{j\notin T}\delta(\mathbf{x},\mathbf{p}_j)\}.$$

With uniform weights, the k-NN prediction is constant on each $V_T$, the majority label of $T$. So the k-NN boundary lies on the edges of the order-$k$ diagram, specifically those where the majority label changes. With distance weights the prediction varies *inside* $V_T$, and boundaries become curves that cut through cells.

**Multiclass:** region of class $c$ = $\{\mathbf{x}: \hat y(\mathbf{x})=c\}$. Boundaries are where the top two scores tie, and points where three regions meet are where three scores tie.

**Grid rendering cost:** evaluating a $G\times G$ grid costs $G^2$ predictions, i.e. $O(G^2\,n\,d)$ with brute force (note 09).

## Geometric meaning
- **1-NN, L2:** a mosaic of convex polygons around every training point, coloured by class. The boundary is a jagged polyline made of straight perpendicular-bisector pieces. A single mislabelled point gets its own "island" polygon.
- **1-NN, L1/L∞:** cell edges run only horizontal/vertical plus 45° diagonals, which looks like city blocks. On integer-grid data, whole patches are tied (see Pitfalls).
- **Growing $k$:** islands disappear, the boundary straightens and lengthens, and minority pockets get absorbed. At $k=n$ everything is the majority colour.
- **Multiclass:** tiled regions with triple-junction points; with uniform even $k$, some cells are tie regions.
- **3D:** cells become convex polyhedra and bisectors become planes (L2). The boundary is a surface, so show 2D slices.
- **$d>2$:** any 2D picture is either a **slice** (other features fixed) or a **projection** (e.g. PCA). In a slice, boundaries are exact for that slice only. In a projection, the boundary of the real model is not generally a function of the 2 shown coordinates, so a "projection boundary map" is an approximation or a different model. Label which one is shown.

## Effect on predictions
- **$k$:** boundary complexity goes down as $k$ goes up (cf. effective df $n/k$, note 03).
- **Metric:** changes the bisector shapes and can move whole cells (note 02).
- **Scaling:** stretches axes, which tilts L2 bisectors in original units and can reassign cells (note 05).
- **Weighting:** uniform gives piecewise-constant regions on order-$k$ cells; $1/d$ gives curved boundaries and small "capture discs" around every training point.
- **Tie rule:** decides the colour of tie regions. With scikit-learn's rule (note 04) they all go to the first class in sorted label order.

## Pitfalls & edge cases
- **2D tie regions under L1/L∞** are common with integer or one-hot features. A renderer that colours "argmax" will silently paint them with the tie-break class. Give ties their own hatch colour.
- **Duplicates with conflicting labels** give coincident sites, an ill-defined Voronoi cell and a tie everywhere that point is nearest.
- **Grid aliasing:** with $G$ cells per axis, any region narrower than about one cell can be missed entirely. That includes the tiny $1/d$ capture discs and thin slivers between close opposite-class points. (Inference from point sampling; marching squares only looks at cell corners.)
- **Marching squares saddles:** cells whose corners alternate in value are ambiguous, and an implementation has to pick a connection. Wikipedia describes using the cell-centre average. The d3 pages I read don't say what d3 does.
- **Contouring needs scalar fields.** For multiclass, contour each class's probability or indicator separately and accept overlaps or gaps at junctions, or draw a raster instead of contours.
- **Slices change with the fixed values.** Moving the "other features" slider changes the map even though the model is unchanged.

## Visualization ideas
- **Voronoi mode (k=1):** draw the exact Voronoi diagram (L2) with cell edges thin and grey, and edges between different-class cells thick in a boundary colour. Hovering a cell highlights its site and shows "every location here has this point as its nearest neighbour". Toggling the metric to L1/L∞ switches to a raster, because a Euclidean Voronoi routine cannot draw L1/L∞ cells. The raster shows the 45° and axis-aligned edges.
- **Bisector playground:** two draggable points of different classes, with their bisector drawn live under the selected metric. Rotating one point around the other under L1 makes a quarter-plane flash into a tie region at exactly 45°. Under L∞ it flashes when the points line up horizontally or vertically.
- **Order-k cells:** for small $n$ (≤ 30), colour each grid cell by the identity of its neighbour set (hashed hue) beside the class-coloured map. Increasing $k$ shows cells merging, and class edges are visibly a subset of cell edges.
- **Boundary smoothing filmstrip:** thumbnails at $k=1,3,7,15,31,n$.
- **Probability shading:** colour saturation = $\max_c\hat p_c$, with a 0.5 contour for binary problems. Grid cells resolved by a tie rule are hatched.
- **Resolution slider ($G$ = 25 → 400):** show render time and how small islands appear or vanish. Optionally refine only cells whose 4 corners disagree.
- **Slice controls for $d>2$:** choose the two axis features and set the others with sliders (defaulting to the query's values). Draw a caption "slice at income = 52,000, tenure = 3".

## Sources
- Wikipedia "Voronoi diagram", https://en.wikipedia.org/wiki/Voronoi_diagram (raw wikitext). Supports: cell definition; closed half-space bounded by the perpendicular bisector; cell = intersection of $n-1$ half-spaces, so a convex polygon; boundaries are segments/rays/lines; Manhattan/Mahalanobis cells are possible but "the equidistant locus for two points may fail to be subspace of codimension 1, even in the two-dimensional case"; nth-order Voronoi cell definition; used for 1-NN classification; Fortune's algorithm $O(n\log n)$.
- d3-contour docs, https://d3js.org/d3-contour and https://d3js.org/d3-contour/contour. Supports: marching squares on a rectangular grid; `size([n,m])`; values[i + j·n]; output GeoJSON MultiPolygons for values ≥ threshold; `smooth` (linear interpolation) on by default.
- Wikipedia "Marching squares", https://en.wikipedia.org/wiki/Marching_squares. Supports: 2×2 cells, 16 cases, linear interpolation, saddle ambiguity.
- UWO CS4442 slides. Supports: Voronoi tessellation as a 1-NN visual with decision boundary.
- Cornell CS4780 lecture note 02. Supports: $k=n$ gives a constant prediction.
- L1/L∞/p=3/p=0.5 bisector shapes and tie-area percentages: **computed locally** (Python grid scan), not from a source. The general case analysis is my derivation, consistent with the Wikipedia codimension remark.
