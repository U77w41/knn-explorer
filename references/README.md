# References

Source material for rebuilding KNN Explorer, collected on 2026-10-27 before writing any code. Every formula and behaviour in `theory/` was checked against at least one cited source. The scikit-learn tie and weighting behaviours were also re-run against scikit-learn 1.9.1.

## Layout

| Folder | What's in it | In git? |
|---|---|---|
| `theory/` | Ten notes (algorithm, distance metrics, choosing k, voting/weighting/ties, preprocessing, decision boundaries, curse of dimensionality, evaluation, neighbour search, feature scaling deep dive). Each one has Math → Geometric meaning → Effect on predictions → Pitfalls → Visualization ideas → Sources. | yes |
| `papers/` | `README.md` lists 15 papers, books and docs with citation, URL, access note and which theory note uses each. | yes |
| `papers/pdf/` | 7 open-access PDFs (Cover & Hart 1967, Beyer et al. 1998 TR, Aggarwal et al. 2001, NCA 2004, LMNN 2009, Radovanović et al. 2010, de Amorim et al. 2023). | no, local study copies |
| `visual-exemplars/` | `existing-tools.md` (survey of 12 interactive explainers), `design-patterns.md` (explorable-explanation principles, colour-blind palettes), `tech-stack.md` (library versions, sizes, timings, recommended stack, what to delete from the Lovable template). | yes |
| `visual-exemplars/screenshots/` | 10 screenshots of the surveyed tools. | no, third-party content |
| `datasets/` | 6 CSVs (target in the last column) plus `README.md` with source, licence, shape, class balance, missing values and what each one teaches. | yes, all CC0 / CC BY 4.0 / generated |
| `datasets/dummy/` | Two candidates for the app's default dataset, **fruit** (apple / orange / pear) and **bank** (normal / suspicious / mule, synthetic). Each has 200 rows, a seeded generator that reproduces the CSV exactly, a binary variant, preview PNGs and a `README.md` showing it passes every teaching target. | yes, generated |
| `repo-audit.md` | What the Lovable version does and why it doesn't build or deploy. | yes |

## Which reference feeds which part of the app

| App feature | Read first |
|---|---|
| Step-by-step prediction for one query point | `theory/01`, `theory/04`, `visual-exemplars/existing-tools.md` (Kanaries vote table, Galles playback bar) |
| Metric switcher, Minkowski p, unit balls | `theory/02`, `theory/06` |
| k slider, bias–variance, validation curve | `theory/03`, `theory/08` |
| Weighting schemes and tie handling | `theory/04` |
| Upload, column typing, one-hot | `theory/05`, `datasets/README.md` (penguins, German credit, wine) |
| Scaling everywhere: scaler choice, its effect on neighbours, votes, k, boundaries, accuracy | `theory/10`, `theory/05` |
| Default dummy dataset | `datasets/dummy/README.md` |
| Decision-boundary map | `theory/06`, `theory/09`, `visual-exemplars/tech-stack.md` §2 |
| More than two features | `theory/07`, `visual-exemplars/tech-stack.md` §6 |
| Colour-linked formulas | `visual-exemplars/design-patterns.md` |

## Findings that should shape the design

1. **Ties are common and need their own colour.** Under L1, two points on a 45° diagonal are equidistant over whole areas, not just along a line. L∞ does the same for points that share an x or y value. Integer and one-hot data hit this often. (`theory/06`)
2. **scikit-learn breaks vote ties toward the label that sorts first**, even when the other class has the nearer neighbours. Ties at the k-th distance go to whichever row comes first in the file, so shuffling rows can change predictions. (`theory/04`)
3. **With 1/d weighting, an exact match takes all the weight**, so training-set accuracy is 100% at every k. Validation has to use held-out rows or leave-one-out. (`theory/04`, `theory/08`)
4. **Dudani weighting gives the k-th neighbour zero weight**, so k = 2 behaves like 1-NN. (`theory/04`)
5. **Z-scoring one-hot columns inflates rare categories.** A mismatch on a category held by 1% of rows costs about 10× more after scaling. (`theory/05`)
6. **Boundary maps are expensive.** n = 5,000 rows, d = 10 features and a 200 × 200 grid is about 2 × 10⁹ operations. The map has to run in a web worker and cache each cell's neighbour list, so moving the k or weighting slider doesn't recompute distances. (`theory/09`)
7. **2-D projections misplace true neighbours.** PCA or UMAP views need a visible caveat, and the neighbour links must come from the full-dimensional distances. (`theory/07`, `visual-exemplars/tech-stack.md` §6)
8. **Scaling is just a choice of feature weights.** For any per-feature shift and stretch, distance in scaled space equals a weighted Minkowski distance in raw units, with weight 1/sⱼᵖ. Centring never changes a Minkowski-metric prediction, though it does change cosine. Min-max, z-score, robust and max-abs differ only in the stretch sⱼ. (`theory/10`)
9. **Raw distances are often one feature.** On wine, proline supplies 99.7% of the squared distance unscaled, and 28.8% after z-scoring. The 5-NN sets before and after share 7.6% of neighbours (Jaccard), and 29.8% of predictions flip. (`theory/10`)
10. **Scaling isn't always a win.** It slightly hurts iris (0.963 → 0.953), and quantile-normal makes every one-hot mismatch cost 10.4. The best k also moves with the scaler (wine: 1 unscaled, 25 z-scored). The app should show the trade-off, not preach "always scale". (`theory/10`)
11. **German Credit is a good accuracy trap.** KNN on its numeric columns scores 0.65–0.67, below the 0.70 you get by always predicting "good". (`datasets/README.md`)

## Gaps

These are link-only because the host blocked automated download. Get them by hand if you want local copies:

- *The Elements of Statistical Learning*: https://hastie.su.domains/ElemStatLearn/ (sections 2.3, 2.5, 7.3, 13.3)
- *ISLP / ISLR2*: https://www.statlearning.com/
- Fix & Hodges 1951: https://apps.dtic.mil/sti/pdfs/ADA800276.pdf
- Omohundro 1989, ball trees: https://www1.icsi.berkeley.edu/ftp/pub/techreports/1989/tr-89-063.pdf
- Dudani 1976 (closed access, IEEE): https://doi.org/10.1109/TSMC.1976.5408784

The ESL equation and page numbers cited in `theory/03` and `theory/07` were confirmed only through lecture notes that reproduce them. Check them against the book once you have it.
