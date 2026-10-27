# 04 · Voting, weighting and ties

## Summary
Each of the $k$ neighbours casts a vote of weight $w_i$. Uniform weights give a plain majority vote, while distance-based weights let closer neighbours count more. Two separate kinds of tie need explicit rules: rows tied at the $k$-th distance (which rows are in?) and classes tied in the vote (which label wins?). scikit-learn resolves them in different and non-obvious ways.

## Math

Class score and probability (from note 01): $S_c=\sum_{i\in N_k}w_i\,\mathbb{1}[y_i=c]$ and $\hat p_c=S_c/\sum_{c'}S_{c'}$. Let $d_i=\delta(\mathbf{x}_q,\mathbf{x}_i)$ for the neighbours, sorted so that $d_1\le d_2\le\dots\le d_k$.

| Scheme | Weight $w_i$ | Source |
|---|---|---|
| Uniform (majority vote) | $1$ | scikit-learn `weights='uniform'` |
| Inverse distance | $1/d_i$ | scikit-learn `weights='distance'`; Wikipedia |
| Inverse squared distance | $1/d_i^2$ | TU Dortmund lecture notes |
| Dudani (1976) linear | $\begin{cases}\dfrac{d_k-d_i}{d_k-d_1} & d_k\ne d_1\\[4pt] 1 & d_k=d_1\end{cases}$ | TU Dortmund notes; Lenz et al. eq. 3 (original paper not read) |
| Gaussian kernel | $\exp\!\big(-a_i^2/2\big)$ with $a_i=d_i/h$ | kernel from Lenz et al. Table 1, which uses $h=d_k$ |
| Custom | any $g(d_i)\ge0$ | scikit-learn callable `weights` |

Symbols: $d_1$ is the nearest neighbour distance, $d_k$ the $k$-th, and $h$ a bandwidth. In the Gaussian row, Lenz et al. rescale distances as $d_i^*=d_i/d_k$, so $h=d_k$. A fixed global $h$ is the other common choice, and the app should state which one it uses.

**Dudani gives the $k$-th neighbour zero weight.** Example: $d=(1,2,4)$ gives $w=(1,\ 0.667,\ 0)$, and $d=(0.5,1,1,2,2.5)$ gives $w=(1,0.75,0.75,0.25,0)$. So "$k$ neighbours" under Dudani means $k-1$ effective voters, unless several rows tie at $d_k$.

**Zero distance (scikit-learn `weights='distance'`).** From `_get_weights` in the source: if any of a query's neighbour distances is exactly 0, the zero-distance neighbours get weight 1.0 and all others get 0.0 for that query. Otherwise $w_i=1/d_i$. So an exact match overrides all other neighbours, and if several exact matches disagree, they split the vote equally. Checked by running scikit-learn 1.9.1: two duplicates at the query with labels 0 and 1, plus two farther rows of class 1, give `predict_proba` = [0.5, 0.5].

**Probabilities.** scikit-learn adds $w_i$ into the bin of $y_i$ and divides by the row sum. With uniform weights the possible values are $\{0,1/k,\dots,1\}$. These are not calibrated probabilities.

**Hubness-aware weighting** (advanced, Radovanović et al. 2010): $w_k(\mathbf{x})=\exp(-h_B(\mathbf{x},k))$, where $h_B=(BN_k(\mathbf{x})-\mu_{BN_k})/\sigma_{BN_k}$ is the standardised count of "bad" k-occurrences, i.e. how often $\mathbf{x}$ appears in other points' neighbour lists with a mismatched label (note 07).

### Ties

**(a) Distance ties at rank $k$** ($d_k=d_{k+1}$ with different labels). Docs warning, verbatim: "if it is found that two neighbors, neighbor k+1 and k, have identical distances but different labels, the results will depend on the ordering of the training data." Checked by running scikit-learn 1.9.1 (brute, kd_tree, ball_tree) on 1D toys: with $k=1$ and two equidistant rows, the one that came **first in the training data** was chosen, and swapping the row order flipped the prediction. With $k=3$, reordering which of two equidistant rows sits earlier changed the prediction from 1 to 0. Treat "first in row order wins" as observed behaviour, not a documented guarantee.

**(b) Vote ties** (two or more classes share the top $S_c$). In the scikit-learn source every path picks the **class with the smallest index in `classes_`**:
- uniform weights: `scipy.stats.mode` returns the smallest value;
- the fast brute-force path takes `argmax` of the probabilities, which returns the first maximum;
- weighted voting uses `weighted_mode`, which updates only on strictly greater counts.

`classes_` comes from `np.unique(y)`, so it is sorted, and the tie goes to the lexicographically/numerically smallest label, regardless of which tied class has the nearer neighbours. Checked: with $k=2$, neighbours "dog" (listed first) and "cat", the prediction is "cat" with `predict_proba` [0.5, 0.5].

**Other tie rules in the literature**
- Cover & Hart (footnote 1): with ties for the nearest neighbour, "decide the most popular category among the ties".
- Cornell CS4780: on a vote tie, fall back to the result for a smaller $k$.
- Other options: include all rows tied at $d_k$ (variable-size neighbourhood), use distance weights, or break randomly with a seed.

## Geometric meaning
- **Uniform:** the label is constant inside each order-$k$ Voronoi cell (note 06), so boundaries follow cell edges.
- **Inverse distance:** weight blows up near each training point, so every training point "wins" a small disc around itself. The boundary passes close to minority points, and the map shows tiny islands around isolated points even at large $k$. Boundaries become curves that cut through order-$k$ cells.
- **Dudani / Gaussian:** smoother falloff. The neighbourhood behaves like a soft-edged ball, so the farthest neighbour barely matters.
- **Distance ties** are where two training points sit on the ball's boundary at once. On integer or one-hot data this is common.
- **Vote-tie regions** are where class scores are equal. For uniform even-$k$ binary problems these are whole regions in 2D, not just lines. Under scikit-learn's rule they all go to the first class, which quietly enlarges its territory on the map.

## Effect on predictions
- Distance weighting makes large $k$ less harmful (far points count little) and reduces vote ties (real-valued weights rarely tie exactly).
- `weights='distance'` makes training-set accuracy 100% for every $k$ (zero-distance rule). Checked on random labels with $k=1,5,15$.
- Dudani at $k=1$ or $k=2$: at $k=2$ the second neighbour has weight 0, so the result equals 1-NN unless $d_1=d_2$.
- Vote-tie rule bias: on a balanced binary problem with even $k$, scikit-learn assigns every tie to the alphabetically first class. Accuracy, and the per-class recall in the confusion matrix, shift toward it.

## Pitfalls & edge cases
- **$1/d$ with $d=0$**: division by zero. You need an explicit rule; scikit-learn's is above.
- **Floating-point near-zeros:** $d=10^{-12}$ is not 0, so that neighbour gets weight $10^{12}$ and dominates. Consider snapping $d<\epsilon$ to 0, and document $\epsilon$.
- **Callable weights that return all zeros** make scikit-learn raise "All neighbors of some sample is getting zero weights".
- **Dudani with all equal distances** ($d_1=d_k$) is defined as uniform. Implement that branch.
- **Gaussian with fixed $h$ too small:** all weights underflow to 0. Use $h=d_k$ or log-sum-exp.
- **Lexicographic order of string labels:** "10" sorts before "9" as strings. Whether labels are parsed as numbers or strings changes `classes_` order, and therefore tie outcomes (note 05).
- **Row order matters** for rank-$k$ ties, so shuffling the uploaded CSV can change predictions. The app should detect and flag ties rather than hide them.

## Visualization ideas
- **Weight-scheme switcher** (uniform / 1/d / 1/d² / Dudani / Gaussian): each neighbour's marker size, and the thickness of its link line to the query, scales with $w_i$. The vote bars grow by $w_i$ per neighbour, colour-linked to the matching term in $S_c$.
- **Weight curve panel:** plot $w(d)$ for the active scheme over $[0,d_k]$, with the $k$ neighbours as dots on the curve. For Dudani the $k$-th dot sits at 0, labelled "this neighbour has no say".
- **Zero-distance demo:** drag the query onto a training point. Under 1/d, all other link lines vanish and an "exact match overrides" badge appears.
- **Tie inspector:** when $d_k=d_{k+1}$, draw the ball boundary in a warning colour with both tied points pulsing, plus a "row order decides: row #12 beats #40" caption. A "shuffle rows" button shows the prediction flip. For vote ties, show equal-height bars and an arrow to the winning class with the rule "smallest label in sorted order".
- **Tie map overlay:** on the boundary map, hatch the grid cells whose prediction came from a tie rule (distance or vote). Even $k$ on binary data lights up large hatched regions.

## Sources
- scikit-learn neighbors user guide, https://scikit-learn.org/stable/modules/neighbors.html. Supports: uniform vs distance weights, callable weights, warning on $k$/$k+1$ ties and training order.
- scikit-learn KNeighborsClassifier API (1.9.1), https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html. Supports: weights parameter text, tie warning text, classes in lexicographic order.
- scikit-learn source `_base.py` (`_get_weights`, `np.unique(..., return_inverse=True)`), https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/neighbors/_base.py. Supports: zero-distance rule, class encoding.
- scikit-learn source `_classification.py` (`predict`, `predict_proba`), https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/neighbors/_classification.py. Supports: `stats.mode`/`weighted_mode`/`argmax` paths, probability normalisation, all-zero-weights error.
- scikit-learn source `utils/extmath.py` (`weighted_mode`), https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/utils/extmath.py. Supports: "If there is more than one such value, only the first is returned"; strict `>` update.
- Local runs of scikit-learn 1.9.1 (scratch scripts, not a URL). Supports: tie and zero-distance behaviours stated as "checked".
- TU Dortmund ML bits (E. Schubert), https://dm.cs.tu-dortmund.de/mlbits/class-knn-motivation. Supports: Dudani formula with $d_k=d_1$ case, $1/d_i$ and $1/d_i^2$, Dudani citation (IEEE Trans. SMC 6(4):325–327, 1976).
- Lenz, Bollaert, Cornelis, "A unified weighting framework for evaluating nearest neighbour classification", arXiv 2311.16872 (read via https://ar5iv.labs.arxiv.org/html/2311.16872). Supports: Dudani eq. 3, rescaled distance $d_i/d_k$, Gaussian kernel $e^{-a^2/2}$, conventions for zero distances.
- OpenAlex record for Dudani 1976, https://api.openalex.org/works/doi:10.1109/TSMC.1976.5408784. Supports: bibliographic details, closed access.
- Wikipedia "k-nearest neighbors algorithm". Supports: $1/d$ weighting.
- Cover & Hart 1967, local PDF. Supports: footnote 1 tie rule.
- Cornell CS4780 lecture note 02. Supports: tie fallback to smaller $k$.
- Radovanović et al. 2010, local PDF. Supports: $h_B$ and $w_k(x)=\exp(-h_B(x,k))$.
- Dudani 1976 original: **not read** (closed access). The formula is second-hand from two independent sources that agree.
