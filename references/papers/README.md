# Papers and books

Local study copies live in `pdf/` (git-ignored, see `.gitignore` here). Every downloaded file was checked with `file`/`pdfinfo` and is a real PDF over 50 KB. "Link-only" means no local copy, with the reason given. Theory notes are in `../theory/`.

---

### Fix & Hodges (1951): origin of the k-NN rule
- **Citation:** Fix, E., & Hodges, J. L. (1951). *Discriminatory Analysis. Nonparametric Discrimination: Consistency Properties.* Report No. 4, Project No. 21-49-004, USAF School of Aviation Medicine, Randolph Field, Texas.
- **URL:** https://apps.dtic.mil/sti/pdfs/ADA800276.pdf
- **Local file:** link-only. DTIC returned a "Service unavailable / The request is blocked" page instead of the PDF, and per instructions I did not retry by other means.
- **Access:** US government technical report on DTIC (public).
- **Contribution:** the first nearest-neighbour-type rule. Per Cover & Hart, it studies the $k_n$-NN rule and proves consistency when $k_n\to\infty$ and $k_n/n\to0$. Gives the "1951 origin" anchor for the app's history panel. **Not read directly**, so every claim about it is second-hand.
- **Used in:** 01, 03.

### Cover & Hart (1967): 1-NN error ≤ about twice Bayes
- **Citation:** Cover, T. M., & Hart, P. E. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*, IT-13(1), 21–27. doi:10.1109/TIT.1967.1053964
- **URL:** https://isl.stanford.edu/~cover/papers/transIT/0021cove.pdf (linked from Cover's publication list, https://isl.stanford.edu/~cover/cover-papers.html)
- **Local file:** `pdf/1967-cover-hart-nearest-neighbor-pattern-classification.pdf` (1.04 MB, 7 pages, scanned)
- **Access:** IEEE-copyrighted, author-hosted copy on Stanford's Information Systems Lab site.
- **Contribution:**
  - Asymptotic bounds $R^*\le R\le 2R^*(1-R^*)$ (eq. 13) and $R^*\le R\le R^*(2-\frac{M}{M-1}R^*)$ (eq. 29/38) for the 1-NN rule.
  - Pointwise risks $r=1-\sum\hat\eta_i^2$ vs $r^*=1-\max\hat\eta_i$.
  - An admissibility example where 1-NN beats every k-NN.
  - k-NN bounds decreasing in $k$ (section VIII) and a tie-handling footnote.
- **Used in:** 01, 03, 04.

### Dudani (1976): distance-weighted k-NN
- **Citation:** Dudani, S. A. (1976). The distance-weighted k-nearest-neighbor rule. *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-6(4), 325–327. doi:10.1109/TSMC.1976.5408784
- **URL:** https://doi.org/10.1109/TSMC.1976.5408784
- **Local file:** link-only. Closed access: OpenAlex reports `is_oa: false`, and no legitimate open copy was found.
- **Contribution:** linear weights $w_i=(d_k-d_i)/(d_k-d_1)$ (1 if $d_k=d_1$). The nearest neighbour gets 1 and the $k$-th gets 0. The formula is taken from two secondary sources that agree (TU Dortmund notes; Lenz et al., arXiv 2311.16872).
- **Used in:** 04.

### Beyer, Goldstein, Ramakrishnan & Shaft (1998/1999): when is "nearest neighbor" meaningful?
- **Citation (published):** Beyer, K., Goldstein, J., Ramakrishnan, R., & Shaft, U. (1999). When is "nearest neighbor" meaningful? In *Database Theory: ICDT'99*, LNCS 1540, 217–235. Springer. doi:10.1007/3-540-49257-7_15
- **Citation (local copy):** same authors, *When Is "Nearest Neighbor" Meaningful?*, Technical Report #1377, Computer Sciences Dept., University of Wisconsin–Madison, June 1998.
- **URL:** https://minds.wisconsin.edu/handle/1793/60174 (PDF bitstream: https://minds.wisc.edu/bitstreams/739ed6c2-564d-4e77-af43-5348d0dca8ef/download)
- **Local file:** `pdf/1998-beyer-goldstein-ramakrishnan-shaft-when-is-nearest-neighbor-meaningful-tr1377.pdf` (2.49 MB, 22 pages, scanned). This is the tech-report version, not the ICDT camera-ready.
- **Access:** UW–Madison institutional repository (MINDS@UW). No item-specific licence is shown.
- **Contribution:** Theorem 1. If the relative variance of distances goes to 0 as dimension grows, then $P[\mathrm{DMAX}\le(1+\varepsilon)\mathrm{DMIN}]\to1$. The paper also defines "unstable" queries, reports the effect at 10–15 dimensions, and names workloads where NN stays meaningful. It is the formal basis for the "all distances look the same" visual.
- **Used in:** 07.

### Aggarwal, Hinneburg & Keim (2001): distance metrics in high dimension
- **Citation:** Aggarwal, C. C., Hinneburg, A., & Keim, D. A. (2001). On the surprising behavior of distance metrics in high dimensional space. In *Database Theory: ICDT 2001*, LNCS 1973, 420–434. Springer. doi:10.1007/3-540-44503-X_27
- **URL:** https://kops.uni-konstanz.de/entities/publication/da11ec73-e8e1-43d1-96fe-0d0359996a6c
- **Local file:** `pdf/2001-aggarwal-hinneburg-keim-surprising-behavior-distance-metrics.pdf` (227 KB, 15 pages)
- **Access:** University of Konstanz repository (KOPS), green open access, CC BY-NC-ND 2.0.
- **Contribution:**
  - Relative contrast $(D\max-D\min)/D\min$ decays like $1/\sqrt d$ for every $L_p$, with a larger constant for smaller $p$; $D\max-D\min$ scales like $d^{1/p-1/2}$.
  - L1 beats L2 in contrast with high probability (Table 2).
  - Fractional $p<1$ does better still.
  - Supports a "p slider vs dimension" visual.
- **Used in:** 02, 07.

### Goldberger, Roweis, Hinton & Salakhutdinov (2004): Neighbourhood Components Analysis
- **Citation:** Goldberger, J., Roweis, S., Hinton, G., & Salakhutdinov, R. (2004). Neighbourhood components analysis. In *Advances in Neural Information Processing Systems 17* (NIPS 2004).
- **URL:** https://proceedings.neurips.cc/paper_files/paper/2004/file/42fe880812925e520249e808937738d2-Paper.pdf
- **Local file:** `pdf/2004-goldberger-roweis-hinton-salakhutdinov-neighbourhood-components-analysis.pdf` (267 KB, 8 pages)
- **Access:** NeurIPS proceedings, free to read.
- **Contribution:** learns a linear map $L$ (equivalently a Mahalanobis $M=L^\top L$) by maximising a softmax leave-one-out k-NN score. It can project to 2D for visualisation. It names k-NN's two weaknesses: storing and searching all the data, and choosing the metric. A candidate "learned metric" extension. Its formulas are mirrored in scikit-learn's NCA docs.
- **Used in:** 01, 02.

### Weinberger & Saul (2009): Large Margin Nearest Neighbor
- **Citation:** Weinberger, K. Q., & Saul, L. K. (2009). Distance metric learning for large margin nearest neighbor classification. *Journal of Machine Learning Research*, 10, 207–244.
- **URL:** https://jmlr.org/papers/v10/weinberger09a.html (PDF: https://jmlr.org/papers/volume10/weinberger09a/weinberger09a.pdf)
- **Local file:** `pdf/2009-weinberger-saul-large-margin-nearest-neighbor.pdf` (1.98 MB, 38 pages)
- **Access:** JMLR, open access.
- **Contribution:** a Mahalanobis metric learned so that each point's $k$ "target neighbours" (same class) sit inside a perimeter, with differently labelled "impostors" pushed out by a unit margin (eq. 10). Its Fig. 1 (before/after neighbourhoods) is a strong template for a metric-learning visual.
- **Used in:** 02.

### Radovanović, Nanopoulos & Ivanović (2010): hubness
- **Citation:** Radovanović, M., Nanopoulos, A., & Ivanović, M. (2010). Hubs in space: Popular nearest neighbors in high-dimensional data. *Journal of Machine Learning Research*, 11(86), 2487–2531.
- **URL:** https://jmlr.org/papers/v11/radovanovic10a.html (PDF: https://jmlr.org/papers/volume11/radovanovic10a/radovanovic10a.pdf)
- **Local file:** `pdf/2010-radovanovic-nanopoulos-ivanovic-hubs-in-space.pdf` (1.09 MB, 45 pages)
- **Access:** JMLR, open access.
- **Contribution:**
  - k-occurrence counts $N_k(x)$ become right-skewed as (intrinsic) dimension grows, which creates hubs and antihubs.
  - Hubs sit near the data mean.
  - "Bad" hubs hurt k-NN, and a hubness-aware vote weight $\exp(-h_B)$ helps.
  - Suggests an "$N_k$ histogram + hub highlighting" visual.
- **Used in:** 04, 07.

### Omohundro (1989): ball trees
- **Citation:** Omohundro, S. M. (1989). *Five Balltree Construction Algorithms.* Technical Report TR-89-063, International Computer Science Institute, Berkeley.
- **URL:** https://www1.icsi.berkeley.edu/ftp/pub/techreports/1989/tr-89-063.pdf
- **Local file:** link-only. The download failed with "Connection reset by peer", and per instructions I did not retry by other means.
- **Access:** ICSI technical report (public).
- **Contribution:** ball-tree construction algorithms, the structure behind scikit-learn's `BallTree`. The notes rely on scikit-learn docs, Wikipedia and Cornell CS4780 for ball-tree facts instead. **Not read.**
- **Used in:** 09.

### Hastie, Tibshirani & Friedman: *The Elements of Statistical Learning* (ESL)
- **Citation:** Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.
- **URL:** https://hastie.su.domains/ElemStatLearn/
- **Local file:** link-only. WebFetch reported the site disallows automated fetching (robots), so per instructions I did not download it.
- **Access:** free PDF on the authors' site (for manual download).
- **Contribution:**
  - §2.3.2 nearest-neighbour methods and effective parameters $N/k$.
  - §2.5 hypercube edge length $e_p(r)=r^{1/p}$.
  - §7.3 bias–variance with the k-NN decomposition $\sigma^2+\text{bias}^2+\sigma^2/k$.
  - §7.10 cross-validation; ch. 13 prototype/k-NN methods.
  - Section numbers verified from a library TOC scan; the formulas were verified through lecture notes that reproduce them. **Equation and page numbers inside ESL are unverified.**
- **Used in:** 03, 07.

### James, Witten, Hastie, Tibshirani & Taylor: *An Introduction to Statistical Learning* (ISLP / ISLR2)
- **Citation:** James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python.* Springer. (R edition: James, Witten, Hastie & Tibshirani, 2nd ed., 2021.)
- **URL:** https://www.statlearning.com/ (PDF links point to https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html and https://hastie.su.domains/ISLR2/ISLRv2_corrected_June_2023.pdf.download.html)
- **Local file:** link-only. Same host as ESL, with the same robots refusal.
- **Access:** statlearning.com offers the PDFs under "Download the PDFs here". The site footer reads "All Rights Reserved"; I found no explicit licence statement.
- **Contribution:** accessible chapter-level treatment of KNN, bias–variance and cross-validation, suitable as "further reading" links in the app. **Not read; no claims in the notes rely on it.**
- **Used in:** none (further reading only).

### de Amorim, Cavalcanti & Cruz (2023): the choice of scaling technique matters
- **Citation:** de Amorim, L. B. V., Cavalcanti, G. D. C., & Cruz, R. M. O. (2023). The choice of scaling technique matters for classification performance. *Applied Soft Computing*. doi:10.1016/j.asoc.2022.109924. Preprint: arXiv:2212.12343v1 (23 Dec 2022). The volume and article number are **unverified**, because the arXiv page gives only the DOI.
- **URL:** https://arxiv.org/abs/2212.12343 (PDF: https://arxiv.org/pdf/2212.12343v1). The paper points to code at "github.com/amorimlb/scaling matters" (the underscore was lost in text extraction; not checked).
- **Local file:** `pdf/2022-de-amorim-cavalcanti-cruz-choice-of-scaling-technique-matters.pdf` (1.20 MB, 37 pages). This is the arXiv preprint, not the journal version.
- **Access:** arXiv preprint. The abstract page shows no explicit licence.
- **Contribution:**
  - Five scalers (Standard, Min-max, MaxAbs, Robust, Quantile) plus no scaling, × 20 classifiers, × 82 binary KEEL datasets with imbalance ratios up to 129.
  - 5-fold stratified CV, with each fold scaled on its own training part. Metrics are F1 and G-mean, with Friedman tests.
  - KNN ($k=5$) mean F1: none 0.51, SS 0.55, MM 0.54, MA 0.53, RS 0.54, QT 0.56. G-mean: 0.56 / 0.60 / 0.58 / 0.57 / 0.59 / 0.60.
  - The Friedman p-value across scalers for KNN is 0.018 (F1) and 0.073 (G-mean). KNN is less sensitive than the SVMs, MLP, perceptron and GP, and trees are insensitive.
  - A poorly chosen scaler can be worse than none, and imbalance makes the differences larger.
  - Section 6 summarises earlier KNN-specific studies (Singh & Singh 2020; Raju et al. 2020).
  - Fig. 1 uses quantile output `normal`. The output setting used in the experiments was not found in the parts read.
- **Used in:** 10.

### Ahsan, Mahmud, Saha, Gupta & Siddique (2021): scaling methods vs 11 classifiers on one dataset
- **Citation:** Ahsan, M. M., Mahmud, M. A. P., Saha, P. K., Gupta, K. D., & Siddique, Z. (2021). Effect of data scaling methods on machine learning algorithms and model performance. *Technologies*, 9(3), 52. doi:10.3390/technologies9030052
- **URL:** https://www.mdpi.com/2227-7080/9/3/52 (PDF link: https://www.mdpi.com/2227-7080/9/3/52/pdf)
- **Local file:** link-only. The PDF request returned an "Access Denied" page (CDN edge) instead of the PDF, and per instructions I did not retry by other means. The HTML article was read with WebFetch.
- **Access:** MDPI open access. The fetched page's licence link points to CC BY 3.0 (not double-checked).
- **Contribution:**
  - 11 classifiers × 6 scalers (Normalization "NR", Standard, MinMax, MaxAbs, Robust, Quantile), plus no scaling, on the UCI Heart Disease data (303 rows, 13 attributes).
  - KNN accuracy (Table 5): none 0.75, NR 0.79, SS 0.86, MM 0.85, MA 0.87, RS 0.84, QT 0.87.
  - Caveats:
    - $k$ and the metric are not reported.
    - The protocol is described both as an 80/20 split and as 10-fold CV.
    - The paper does not say whether scalers were fitted on training data only.
    - "NR" is not tied to a specific scikit-learn class.
    - It is one small dataset, so treat the result as anecdotal.
- **Used in:** 10.

### Singh & Singh (2020): 14 normalisation methods with KNN
- **Citation:** Singh, D., & Singh, B. (2020). Investigating the impact of data normalization on classification performance. *Applied Soft Computing*, 97, 105524. doi:10.1016/j.asoc.2019.105524 (as cited by de Amorim et al.)
- **URL:** https://doi.org/10.1016/j.asoc.2019.105524
- **Local file:** link-only. The DOI redirects to Elsevier's linking hub, which returned an empty page to WebFetch, and a web search found no open copy. Open-access status is **unverified**.
- **Contribution:** second-hand only, via de Amorim et al. §6. It applies 14 normalisation methods to 21 datasets with KNN as the only classifier. Most methods improve accuracy, but no scaling is not always the worst, and only Pareto scaling and power transformation were significantly better than no scaling. **Not read.**
- **Used in:** 10 (mentioned only).

### scikit-learn documentation: preprocessing guide and two scaling examples
- **Citation:** scikit-learn developers. "Preprocessing data" (user guide §8.3), "Importance of Feature Scaling" and "Compare the effect of different scalers on data with outliers". scikit-learn 1.9.1 documentation, 2026.
- **URL:** https://scikit-learn.org/stable/modules/preprocessing.html, https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html, https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html
- **Local file:** link-only. These are HTML documentation pages with no PDF, read with WebFetch on 2026-10-27.
- **Access:** project documentation (BSD licence).
- **Contribution:**
  - Formulas and defaults for every scaler in note 10, including the Yeo–Johnson/Box–Cox definitions and the quantile mapping $G^{-1}(F(X))$.
  - "Importance of Feature Scaling": KNN ($k=20$) on wine's `proline` and `hue` gives "completely different models" with and without `StandardScaler`. PCA plus logistic regression reaches 35.19% test accuracy unscaled vs 96.30% standardised. It warns that scaling can hurt when the small-scale features are not predictive.
  - "Compare the effect…" (California housing, `MedInc` vs `AveOccup`): MinMax "compresses all inliers into the narrow range [0, 0.005]". RobustScaler is unaffected by a few large outliers. The quantile transformer collapses outliers to the bounds ("saturation artifacts"). `Normalizer` puts samples on the unit circle.
- **Used in:** 05, 10.
