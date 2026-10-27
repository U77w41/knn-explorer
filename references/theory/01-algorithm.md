# 01 · The k-NN algorithm

## Summary
k-NN classifies a query point by looking up the $k$ training rows closest to it and taking a (possibly weighted) vote of their labels. "Training" only stores the data, so all the work happens at prediction time, at a cost of $O(n\cdot d)$ distance arithmetic per query with brute force. Class probabilities are the vote shares.

## Math

**Notation used across all notes**

| Symbol | Meaning |
|---|---|
| $\mathcal{D}=\{(\mathbf{x}_i,y_i)\}_{i=1}^{n}$ | training set, $n$ rows |
| $\mathbf{x}_i\in\mathbb{R}^d$ | feature vector of row $i$ ($d$ features after encoding) |
| $y_i\in\{c_1,\dots,c_M\}$ | class label, $M$ classes |
| $\mathbf{x}_q$ | query point |
| $\delta(\mathbf{a},\mathbf{b})$ | distance function (see note 02) |
| $\delta_{(j)}$ | $j$-th smallest of $\delta(\mathbf{x}_q,\mathbf{x}_1),\dots,\delta(\mathbf{x}_q,\mathbf{x}_n)$ |
| $N_k(\mathbf{x}_q)$ | index set of the $k$ nearest training rows |
| $w_i$ | vote weight of neighbour $i$ ($w_i=1$ for uniform voting; see note 04) |
| $\mathbb{1}[\cdot]$ | indicator: 1 if true, else 0 |

**Neighbour set** (Cornell CS4780 formulation): $N_k(\mathbf{x}_q)$ is any set of $k$ indices such that every row outside it is at least as far as the farthest row inside it:

$$|N_k|=k,\qquad \forall j\notin N_k:\ \delta(\mathbf{x}_q,\mathbf{x}_j)\ \ge\ \max_{i\in N_k}\delta(\mathbf{x}_q,\mathbf{x}_i).$$

When several rows share the distance $\delta_{(k)}$ this set is not unique (note 04).

**Prediction steps (exact)**
1. Compute $\delta_i=\delta(\mathbf{x}_q,\mathbf{x}_i)$ for $i=1..n$.
2. Select the $k$ smallest $\delta_i$, giving $N_k(\mathbf{x}_q)$.
3. Compute weights $w_i$ for $i\in N_k$.
4. Class scores: $$S_c(\mathbf{x}_q)=\sum_{i\in N_k(\mathbf{x}_q)} w_i\,\mathbb{1}[y_i=c].$$
5. Probabilities (what scikit-learn's `predict_proba` returns): $$\hat p(c\mid\mathbf{x}_q)=\frac{S_c(\mathbf{x}_q)}{\sum_{c'}S_{c'}(\mathbf{x}_q)}.$$ With uniform weights this is just (votes for $c$)$/k$.
6. Label: $\hat y=\arg\max_c S_c(\mathbf{x}_q)$, with a tie rule (note 04).

**Cost per query (brute force).** Step 1 is $n$ distances of $d$ coordinates each, so $O(n\,d)$ (Cornell CS4780 note 16 gives $O(dn)$; scikit-learn gives brute-force query time $O[DN]$). Step 2 adds $O(n)$ with a selection algorithm or $O(n\log k)$ with a size-$k$ heap (standard, not separately sourced). Steps 3 to 6 are $O(k+M)$.

**Cover & Hart (1967) asymptotic bound for the 1-NN rule.** Let $R^*$ be the Bayes risk (lowest achievable error) and $R=\lim_{n\to\infty}R(n)$ the large-sample error of the **1-NN** rule. For $M=2$ (their eq. 13):

$$R^*\ \le\ R\ \le\ 2R^*(1-R^*).$$

For $M$ classes (their eq. 29, proved as eq. 38):

$$R^*\ \le\ R\ \le\ R^*\!\left(2-\frac{M}{M-1}\,R^*\right).$$

Conditions: $X$ is a separable metric space, and with probability one $\mathbf{x}$ is either a continuity point of every class density or a point of non-zero probability mass. The paper says the bounds are as tight as possible. Pointwise, the limiting 1-NN conditional risk is (their eq. 31)

$$r(\mathbf{x})=1-\sum_{i=1}^{M}\hat\eta_i(\mathbf{x})^2,$$

compared with the Bayes conditional risk $r^*(\mathbf{x})=1-\max_i\hat\eta_i(\mathbf{x})$ (eq. 32). Here $\hat\eta_i(\mathbf{x})$ is the posterior probability of class $i$ at $\mathbf{x}$. For $M=2$: $r=2\eta(1-\eta)$ vs $r^*=\min(\eta,1-\eta)$.

Section VIII of the same paper covers odd $k$ with $M=2$. As $n\to\infty$ with $k$ fixed, the conditional $k$-NN risk $r_k(\mathbf{x})$ falls monotonically in $k$ towards $\min\{\hat\eta_1,1-\hat\eta_1\}$, so the upper bounds on $R_k$ also fall towards $R^*$ (eqs. 45 to 48).

**Origin.** According to Cover & Hart (p. 22), Fix & Hodges (1951) made "the first formulation of a rule of the nearest neighbor type". They studied the $k_n$-NN rule and proved it consistent when $k_n\to\infty$ and $k_n/n\to0$. I could not obtain the Fix & Hodges report itself (DTIC blocked the download), so this is second-hand via Cover & Hart and Wikipedia.

## Geometric meaning
- 2D: the query is a point. The $k$ neighbours are the points inside the smallest "ball" around it that holds $k$ of them. The ball's radius is $\delta_{(k)}$ and its shape depends on the metric: a circle for L2, a diamond for L1, a square for L∞ (note 02).
- The vote is a tally of colours inside that ball.
- 1-NN splits the plane into Voronoi cells. k-NN splits it into order-$k$ Voronoi cells, each with a fixed neighbour set (note 06).
- High-D: the same ball, but it has to grow to almost the full data range before it captures $k$ points (note 07).

## Effect on predictions
- No fitted parameters. Adding or removing a single training row can change predictions only for queries whose neighbour set includes that row.
- Prediction depends on four choices: $k$, $\delta$ (metric + scaling + encoding), the weights $w_i$, and the tie rule.
- Probabilities are coarse. With uniform weights they can only be multiples of $1/k$ (e.g. $k=5$ gives $\{0,0.2,\dots,1\}$).
- As $n\to\infty$ the 1-NN error is at most about $2R^*$. That is a large-sample guarantee only and says nothing about small $n$.

## Pitfalls & edge cases
- Wikipedia says the multi-class Cover–Hart bound is for "the asymptotic k-NN error rate". The paper proves it for the **1-NN** rule. Attribute it to 1-NN in the app.
- With fewer than $k$ training rows, $k$-NN is undefined. scikit-learn raises an error (requires `n_neighbors <= n_samples_fit`).
- Equidistant rows at rank $k$ make $N_k$ ambiguous. Results then depend on row order (note 04).
- "Lazy" means fast fitting and slow prediction. A decision-boundary map is thousands of predictions (note 09).
- Predicting on the training rows themselves: each row is its own nearest neighbour at distance 0 (note 08).

## Visualization ideas
- **Step-through pipeline** (Next button or auto-play), with the formula panel highlighting the active step:
  1. Query point drops in (draggable).
  2. Thin lines from query to every training point. Each line is coloured by the training point's class and labelled with $\delta_i$. Hovering a line shows the per-feature terms of $\delta_i$ (note 02).
  3. A sorted bar chart of all $\delta_i$ animates into ascending order. The first $k$ bars stay saturated and the rest fade.
  4. A metric-shaped ball grows from the query until it touches the $k$-th point. Its radius label $\delta_{(k)}$ links to $\delta_{(k)}$ in the formula.
  5. The vote tally: one stacked bar per class, each neighbour adding a block of height $w_i$. Block colour = class colour = colour of $\mathbb{1}[y_i=c]$ in $S_c$.
  6. The bars normalise into $\hat p(c\mid\mathbf{x}_q)$. The winner gets a highlight, and a tie flashes a "tie rule" badge.
- **Operation counter** beside the playground showing "distances computed: $n$, coordinate ops: $n\cdot d$". Fitting shows 0 ops, which makes "lazy" concrete.
- **Cover–Hart widget:** an $R^*$ slider from 0 to $(M-1)/M$ and an $M$ selector, plotting the band between $R^*$ and $R^*(2-\frac{M}{M-1}R^*)$. A second panel for $M=2$ plots $r(\eta)=2\eta(1-\eta)$ against $r^*(\eta)=\min(\eta,1-\eta)$ over $\eta\in[0,1]$, so the user can see the factor-2 gap is widest when $\eta$ is small.

## Sources
- Cover & Hart 1967, local PDF `papers/pdf/1967-cover-hart-nearest-neighbor-pattern-classification.pdf` (from https://isl.stanford.edu/~cover/papers/transIT/0021cove.pdf). Supports: bound eqs. 13, 29, 38; conditions of the theorem; eqs. 31–32; k-NN section VIII eqs. 45–48; Fix & Hodges description on p. 22; tie footnote.
- Cornell CS4780 lecture note 02, https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html. Supports: formal definition of $S_\mathbf{x}$, mode of labels, 1-NN ≤ 2× Bayes statement.
- Cornell CS4780 lecture note 16, https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote16.html. Supports: $O(dn)$ per test input.
- scikit-learn 1.9.1 user guide, https://scikit-learn.org/stable/modules/neighbors.html. Supports: brute-force query $O[DN]$, majority vote description.
- scikit-learn 1.9.1 KNeighborsClassifier API, https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html. Supports: `predict_proba` returns class probabilities with classes in lexicographic order.
- scikit-learn source `_classification.py`, https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/neighbors/_classification.py. Supports: probabilities = per-class sum of weights divided by row total.
- Wikipedia "k-nearest neighbors algorithm", https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm. Supports: Fix & Hodges 1951 origin, computation deferred until evaluation (lazy), the loose "k-NN" wording of the bound (flagged above).
- Goldberger et al. 2004 (NCA), local PDF. Supports: "must store and search through the entire training set in order to classify a single test point".
- Fix & Hodges 1951: **not read** (download blocked). Claims about it are second-hand.
