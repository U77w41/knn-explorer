# 07 · Curse of dimensionality

## Summary
As the number of features grows, a neighbourhood holding a fixed fraction of the data has to span almost the whole range of every feature. Distances from a query to all points also bunch together, so "nearest" means less. Lower-$p$ metrics keep more contrast, and high-dimensional data grows "hubs" that turn up in many neighbour lists. One-hot encoding feeds all of this by adding dimensions.

## Math

**Hypercube neighbourhood edge length** (ESL §2.5 "Local Methods in High Dimensions"; the ESL text was not read, the formula is verified via two lecture decks reproducing it). Data uniform in the unit hypercube $[0,1]^p$, where $p$ here is the number of features. A sub-cube capturing a fraction $r$ of the data has volume $r$, so its edge length is

$$e_p(r)=r^{1/p}.$$

| $p$ | $e_p(0.01)$ | $e_p(0.1)$ |
|---|---|---|
| 1 | 0.01 | 0.10 |
| 2 | 0.10 | 0.32 |
| 3 | 0.22 | 0.46 |
| 10 | 0.63 | 0.79 |
| 100 | 0.955 | 0.977 |

ESL's Fig. 2.6 caption, as reproduced in IIT Madras slides: "In ten dimensions we need to cover 80% of the range of each coordinate to capture 10% of the data." Cornell CS4780 states the same thing as $\ell\approx(k/n)^{1/d}$. With $n=1000$, $k=10$: $\ell=0.1,\ 0.63,\ 0.955,\ 0.9954$ at $d=2,10,100,1000$. Keeping $\ell=0.1$ needs $n=k\cdot10^d$.

**Distance concentration, Beyer et al.** (Theorem 1 in UW–Madison TR 1377, 1998; the ICDT 1999 paper is the published version). Notation:
- $m$: dimensionality index;
- $n$: number of data points (held fixed);
- $P_{m,i}$: data points, $Q_m$: query;
- $d_m$: distance function;
- $0<p<\infty$: a constant;
- $\mathrm{DMIN}_m=\min_i d_m(P_{m,i},Q_m)$ and $\mathrm{DMAX}_m=\max_i d_m(P_{m,i},Q_m)$.

$$\text{If}\quad \lim_{m\to\infty}\mathrm{var}\!\left(\frac{d_m(P_{m,1},Q_m)^p}{\mathbb{E}\big[d_m(P_{m,1},Q_m)^p\big]}\right)=0\quad\text{then}\quad\forall\varepsilon>0:\ \lim_{m\to\infty}P\big[\mathrm{DMAX}_m\le(1+\varepsilon)\,\mathrm{DMIN}_m\big]=1 .$$

In words: if the relative spread of distances vanishes, the farthest point ends up within a factor $(1+\varepsilon)$ of the nearest. The conditions are broader than i.i.d. features. The abstract reports the effect "for as few as 10–15 dimensions" in experiments.

**Relative contrast, Aggarwal, Hinneburg & Keim (2001).** They write $k$ for the norm parameter; here it is $p$, to match note 02. For $N$ points and a query at the origin, with $D\max^p_d,D\min^p_d$ the farthest and nearest $L_p$ distances in dimension $d$, the relative contrast is

$$\mathrm{RC}_p(d)=\frac{D\max^p_d-D\min^p_d}{D\min^p_d}.$$

- Lemma 1 / Corollaries 1–2: $\lim_{d\to\infty}\mathbb{E}\big[(D\max^p_d-D\min^p_d)/d^{1/p-1/2}\big]$ lies between constants $C_p$ and $(N-1)C_p$. So the absolute gap grows like $d^{1/p-1/2}$: it diverges for $p=1$, stays bounded for $p=2$, and shrinks to 0 for $p\ge3$.
- Theorem 2 (uniform data, $N=2$): $\lim_{d\to\infty}\mathbb{E}\big[\mathrm{RC}_p(d)\cdot\sqrt d\big]=C\sqrt{1/(2p+1)}$. So relative contrast decays like $1/\sqrt d$ for every $p$, but with a larger constant for smaller $p$.
- Theorem 3 / Corollary 4 extend this to fractional $p=f=1/l$, $l$ an integer: smaller $f$ gives more contrast.
- Table 2 ($N=10$ uniform points, 1000 trials): the probability that L2's contrast is below L1's is 85.0% at $d=2$, 91.3% at $d=4$, 95.6% at $d=10$ and 98.2% at $d=100$.
- Table 3 (UCI data): nearest-neighbour class-match counts mostly fall as the norm parameter rises from 0.1 to 10. The paper notes that L∞ was often worse than picking a random record.

**Own simulation** (uniform $[0,1]^d$, $n=1000$, 20 random queries, mean $\mathrm{RC}$, seed 0; illustration only, not a source):

| $d$ | $p=0.5$ | $p=1$ | $p=2$ | $p=\infty$ |
|---|---|---|---|---|
| 10 | 4.75 | 3.40 | 2.72 | 2.41 |
| 100 | 0.72 | 0.55 | 0.43 | 0.35 |
| 1000 | 0.18 | 0.15 | 0.12 | 0.09 |

**Concentration of squared radius** (Wikipedia, uniform in $[-1,1]^d$): $r^2=\sum_i x_i^2$ has mean $d/3$ and variance $4d/45$. So $r^2/d\approx\mathcal{N}\big(1/3,\ (2/\sqrt{45d})^2\big)$, and the relative spread shrinks like $1/\sqrt d$.

**Hubness, Radovanović, Nanopoulos & Ivanović (2010).**
- $N_k(\mathbf{x})$: the number of times $\mathbf{x}$ appears among the $k$ nearest neighbours of the other points. Its mean is fixed because $\sum_{\mathbf{x}}N_k(\mathbf{x})=k\,|D|$.
- Skewness: $S_{N_k}=\mathbb{E}\big[(N_k-\mu_{N_k})^3\big]/\sigma_{N_k}^3$.
- As dimensionality rises, $N_k$ becomes right-skewed. "Hubs" have very large $N_k$, and "antihubs" have $N_k\approx0$ and behave like outliers.
- For i.i.d. uniform data with Euclidean distance, the paper reports skewness 0.121, 1.541 and 5.445 at $d=3,20,100$ (footnote 5, for its Fig. 1). Own check ($n=2000$, $k=10$): 0.03, 2.26, 3.57, with 0, 25 and 92 points never chosen as anyone's neighbour.
- In unimodal data, points close to the data mean tend to become hubs.
- Hubness tracks intrinsic dimensionality more closely than the raw feature count.
- "Bad" k-occurrences $BN_k(\mathbf{x})$ count appearances in neighbour lists with a mismatched label. Bad hubs hurt k-NN classification.

## Geometric meaning
- **2D vs 10D vs 100D:** in 2D a small square around the query holds 1% of uniform data. In 10D the "small" cube must span 63% of every axis; in 100D, 95.5%. The neighbourhood is no longer local.
- **Distance histogram:** the histogram of distances from one query to all points is wide in 2D. In high $d$ it becomes a narrow spike far from 0, so nearest and farthest are almost the same.
- **Volume near the surface:** uniform points in a high-dimensional cube sit on a thin shell at radius $\approx\sqrt{d/3}$ (for $[-1,1]^d$). Both the centre and the corners are empty.
- **Hubs:** in the k-NN graph (arrow from each point to its neighbours), a few nodes collect many incoming arrows.
- **Low intrinsic dimension rescues k-NN:** data on a 2D surface inside 100D behaves more like 2D (Cornell CS4780). Irrelevant noise features hurt; relevant ones can add contrast (Wikipedia).

## Effect on predictions
- With many irrelevant features, neighbours become close to random draws and accuracy falls toward the majority-class rate.
- Choosing $p\le1$ can preserve more neighbour contrast in high $d$ (Aggarwal et al.). This is empirical and distribution-dependent, not a guarantee.
- Feature selection, dimension reduction (PCA) or metric learning (NCA, LMNN) cut the effective dimension.
- Hubs bias predictions toward their labels. Hub-aware weighting (note 04) counters that.

## Pitfalls & edge cases
- **One-hot blow-up:** a categorical with $K$ levels adds $K$ columns (`OneHotEncoder`). Ten categoricals with 20 levels each add 200 dimensions. The extra dimensions are not "real" in the same way: within one feature, exactly one column is 1. But each one-hot feature adds a 0-or-2 (L1) jump to every distance, so with mostly-categorical data distances take only a few distinct values and **ties at rank $k$ become frequent**.
- **High $d$ also kills tree indexes** (note 09). scikit-learn's `auto` switches to brute force when $D>15$.
- **"Curse" depends on the data.** Theorems assume conditions (e.g. vanishing relative variance) that clustered real data may violate. Beyer et al. identify workloads where nearest neighbour stays meaningful.
- **Notation clash:** ESL uses $p$ for dimension, Aggarwal et al. use $k$ for the norm parameter, and Beyer et al. use $m$ for dimension and $p$ for an exponent. The app should pick one convention. These notes use $d$ = dimension, $p$ = Minkowski parameter, except where quoting ESL's $e_p(r)$.

## Visualization ideas
- **Edge-length slider:** a $d$ slider from 1 to 100 and an $r$ slider (fraction of data). Show a bar per dimension filled to $e_d(r)$, with the formula $r^{1/d}$ beside it, colour-linked. At $d=10$, $r=0.1$, the bars read 79%.
- **Distance-histogram morph:** sample uniform data at the chosen $d$, then plot the histogram of distances from a random query with DMIN and DMAX marked. The relative contrast $\frac{\mathrm{DMAX}-\mathrm{DMIN}}{\mathrm{DMIN}}$ is shown live and colour-linked to the two markers. Overlay the histograms for $p=0.5,1,2,\infty$ to show contrast ordering.
- **Add-noise-features demo on the user's data:** a button "+5 random noise columns" re-runs CV. The accuracy-vs-noise-columns curve drops, and the 2D neighbour highlights scatter.
- **Hubness bar chart:** a histogram of $N_k$ over training points, with the skewness value. Clicking a hub highlights it on the scatter along with all points that list it as a neighbour, coloured by whether labels match (good vs bad occurrences).
- **One-hot dimension counter:** after encoding, show "your 6 columns became 47 dimensions", plus the number of distinct distance values to a query (exposing tie risk).

## Sources
- ESL table of contents, https://sisis.rz.htw-berlin.de/inh2012/12427576.pdf. Supports: §2.5 title.
- FAU MLISP lecture 5, https://machinelearning.tf.fau.de/pdfs/mlisp/Lecture5.pdf. Supports: $l_p(r)=r^{1/p}$, example $r=1/1000$, $p=10$ gives 0.5. Illustrations credited to Hastie/Tibshirani/Friedman.
- IIT Madras PRML slides, https://www.cse.iitm.ac.in/~vplab/courses/PRML/3_1.pdf. Supports: ESL Fig. 2.6 caption quote ("80% ... 10%").
- Cornell CS4780 lecture note 02, https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html. Supports: $\ell\approx(k/n)^{1/d}$ table, $n=k\cdot10^d$, distance concentration figure, low intrinsic dimension.
- Beyer, Goldstein, Ramakrishnan, Shaft, UW–Madison TR 1377 (1998), local PDF `papers/pdf/1998-beyer-...-tr1377.pdf`. Supports: Definitions 1 and 3, Theorem 1 and its proof, the 10–15-dimension remark (abstract), "special workloads".
- Aggarwal, Hinneburg, Keim (ICDT 2001), local PDF. Supports: relative contrast definition, Lemma 1, Corollaries 1–2, Theorem 2, fractional metrics (Section 3, Theorem 3, Corollary 4), Tables 2 and 3.
- Radovanović, Nanopoulos, Ivanović (JMLR 2010), local PDF. Supports: $N_k$ definition, $S_{N_k}$, footnote 5 values, $\sum N_k=k|D|$, hubs near the mean, antihubs, intrinsic dimensionality, $BN_k$.
- Wikipedia "Curse of dimensionality", https://en.wikipedia.org/wiki/Curse_of_dimensionality (raw wikitext). Supports: $r^2$ mean $d/3$, variance $4d/45$, shell radius, Beyer limit statement, hubs, relevant vs irrelevant dimensions.
- scikit-learn neighbors guide. Supports: `auto` uses brute when $D>15$.
- scikit-learn preprocessing guide. Supports: one-hot produces `n_categories` binary features.
- Own simulation numbers (relative contrast table, hubness check): computed locally with NumPy, seed 0.
