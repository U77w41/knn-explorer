# Tech stack options: static KNN explorer on GitHub Pages

Constraints: static hosting only, user data never leaves the browser, interactive 2-D canvas with heavy per-pixel compute, math rendering, CSV upload.

## How the numbers were obtained (2026-10-27)
- **Version, licence, publish date:** `npm view <pkg>` against registry.npmjs.org.
- **min+gzip.** I used three methods; each table row says which.
  - **dist:** `gzip -9` of the minified bundle the package ships (from `npm pack`). UMD bundles exclude their dependencies.
  - **esb:** an entry importing the named symbols, bundled with `esbuild 0.28.2 --bundle --minify --format=esm` (React externalised for React libraries), then `gzip -9`. This includes dependencies and tree-shaking, which is what an app actually ships.
  - **doc:** the vendor's own table (Plotly's `dist/README.md`).
- **Bundlephobia:** the API answered once (d3-delaunay 6.0.4: 19,167 B min / 6,917 B gzip, which agrees with the dist and esb figures). Every later request returned `client_error`, so I measured locally instead.
- **Timings:** Node 22.23, 2-vCPU Xeon 2.1 GHz. These are indicative only, since browsers differ.

## 1. Framework (brief)

| Option | Current (npm) | Notes |
|---|---|---|
| **Vite + React + TS (keep)** | vite 8.3.5, react / react-dom 19.3.1, typescript 7.0.2 (Apache-2.0), @vitejs/plugin-react 6.1.2 / plugin-react-swc 4.3.3 | The repo is on Vite 5.4 and React 18.3, which work. React 19 + react-dom/client is **68.8 KB** gz (esb). Vite 8 needs Node `^20.19.0 \|\| >=22.12.0`. react-router 8.5.0 needs React ≥ 19.2.7. Largest ecosystem for sliders, popovers and tables (shadcn/Radix already present). |
| Svelte 5 / SvelteKit static | svelte 5.57.3, @sveltejs/kit 3.1.0, @sveltejs/adapter-static 4.0.0 (all MIT) | Smaller runtime, built-in tweened/spring stores that suit step animations. Costs a full rewrite and loses the shadcn/React components. |
| Plain TS + Vite | — | Zero framework cost. Fine for the canvas, but the UI state (wizard, linked highlight, many controls) becomes hand-rolled. |

**Verdict:** stay on Vite + React + TS. Decide whether to start a fresh scaffold (Vite 8, React 19) and port `src/lib/knnEngine.ts`, or repair the template in place (§9). A fresh scaffold is less work than pruning the template.

## 2. Rendering

### Sizes and licences

| Library | Version | Licence | min+gzip | Method |
|---|---|---|---|---|
| d3 (everything) | 7.9.0 | ISC | 92.4 KB | dist `d3.min.js` |
| d3-delaunay | 6.0.4 | ISC | 7.0 KB (7.2 esb) | dist |
| d3-contour | 4.0.2 | ISC | 2.4 KB (2.0 esb) | dist |
| d3-scale (linear + ordinal, with deps) | 4.0.2 | ISC | 8.2 KB | esb |
| d3-scale-chromatic (`schemeTableau10` only) | 3.1.0 | ISC | 0.2 KB | esb |
| d3-geo `geoPath` (to draw contour GeoJSON) | 3.1.1 | ISC | 2.2 KB | esb |
| plotly.js full (`plotly.js-dist-min`) | 4.1.2 | MIT | 1,476 KB (README: 1.49 MB) | dist / doc |
| plotly.js cartesian (has `heatmap`, `contour`) | 4.1.2 | MIT | 499 KB (README: 504 kB) | dist / doc |
| plotly.js basic (bar, pie, scatter only) | 4.1.2 | MIT | 398 KB (README: 401 kB) | dist / doc |
| react-plotly.js (wrapper only, via `/factory`) | 4.1.0 | MIT | 1.4 KB + plotly | esb |
| @observablehq/plot (with its d3 deps) | 0.6.17 | ISC | 132.9 KB (UMD without d3: 68.8 KB) | esb / dist |
| @visx/heatmap | 4.0.0 | MIT | 1.3 KB | esb |
| @visx/delaunay (`voronoi`) | 4.0.0 | MIT | 8.0 KB | esb |
| @visx/scale | 4.0.0 | MIT | 11.9 KB | esb |
| regl | 2.1.1 (published 2024-11-12) | MIT | 28.4 KB (41.1 esb of default entry) | dist `regl.min.js` |
| Canvas 2D / raw WebGL | — | — | 0 | — |

Plotly bundle contents (from https://unpkg.com/plotly.js@4.1.2/dist/README.md): **basic** has bar, pie and scatter; **cartesian** adds box, contour, heatmap, histogram, histogram2d(contour), image, scatterternary and violin; `scattergl` is only in gl2d, strict and full.

### Fit for each job

| Job | Canvas 2D (+ d3 math) | React SVG (d3-scale / visx) | Plotly | Observable Plot | regl / WebGL |
|---|---|---|---|---|---|
| Scatter + boundary heatmap at 200×200 (40k cells) | **Best.** Fill an `ImageData` (one pixel per cell) and `drawImage` it scaled with smoothing off. | Poor. 40k `<rect>` nodes. visx `HeatmapRect` is SVG, so use it for small grids only. | Easy (`heatmap` + `scatter`, cartesian bundle) | Raster mark exists (in the docs sidebar; not reviewed) | Best for huge n or grid: KNN per fragment |
| Animated neighbour lines and growing unit ball (L1 diamond, L2 circle, Lp superellipse \|x\|ᵖ+\|y\|ᵖ=rᵖ) | Good. Redraw an overlay canvas each frame. | **Best.** A few dozen elements, CSS/JS tweens, hit-testable, accessible. | Awkward. Shapes in `layout.shapes`, every frame re-renders through Plotly. | Not built for per-frame animation | Overkill |
| Voronoi overlay (k=1) | Good. `voronoi.renderCell(i, ctx)` | Good. `voronoi.renderCell(i)` returns an SVG path, fine for n ≲ few k | Not native | `voronoi`, `voronoiMesh`, `delaunayLink`, `hull` marks | — |
| Linked highlighting across views | Manual hit-testing (`delaunay.find(x, y)` gives the L2 nearest point) | **Best.** DOM events and shared state | Its own event model, harder across custom panels | Pointer/tip interactions (sidebar; not reviewed) | Manual |
| Small static multiples (k = 1/5/15, p = 1/2/∞), accuracy-vs-k line | Fine | Fine | Fine but heavy | **Best** (concise) | — |

**Caveats:**
- **Voronoi is Euclidean only.** d3-delaunay builds Euclidean Voronoi cells, which equal the 1-NN regions only for p = 2. For L1, L∞ or other p, the 1-NN regions differ, so draw them from the grid. Label the overlay "Voronoi (L2)".
- **`delaunay.find(x, y, i)`:** "Returns the index of the input point that is closest to the specified point". This is L2 nearest-point hover, useful for picking points, not for KNN under other metrics. Docs: https://d3js.org/d3-delaunay/delaunay and https://d3js.org/d3-delaunay/voronoi
- **d3-contour:** "computes contour polygons by applying marching squares to a rectangular array of numeric values" (https://npmjs.com/package/d3-contour). It outputs one GeoJSON MultiPolygon per threshold, and `smooth` (on by default) interpolates edges. For a class grid, run one contour per class on its 0/1 indicator (or vote share) at threshold 0.5 to get crisp boundary lines over the heatmap. Draw with `d3.geoPath(null, ctx)`.
- **visx:** "low-level visualization components", "pick and choose the packages you need" (https://cdn.jsdelivr.net/gh/airbnb/visx@master/README.md). Peer deps allow React 18 or 19.
- **regl:** "Fast functional WebGL", MIT, no dependencies (https://cdn.jsdelivr.net/npm/regl@2.1.1/README.md). Example shaders are GLSL ES 1.0 style. Use it only if Canvas plus a worker is too slow (see timings).
- **Observable Plot marks:** https://observablehq.github.io/plot/marks/delaunay

### Compute timing (brute force, JS typed arrays, insertion top-k)

| Grid | n | d | k | p | ms |
|---|---|---|---|---|---|
| 200×200 | 60 | 2 | 5 | 2 | 31 |
| 200×200 | 344 | 2 | 5 | 2 | 113 |
| 200×200 | 1000 | 2 | 15 | 2 | 382 |
| 200×200 | 1000 | 2 | 15 | **3** (`**` pow) | **5816** |
| 200×200 | 569 | 30 | 5 | 2 | 1076 |
| 100×100 | 1000 | 2 | 15 | 2 | 95 |

**Takeaways:**
- Run the grid in a worker.
- Render coarse to fine (100² then 200²).
- Write special-case loops for p = 1, 2 and ∞. Generic `pow` is about 15× slower, so for integer p use repeated multiplication.
- Skip the final `^(1/p)` while ranking, since it's monotonic.
- At n ≈ 1000 and p = 2, 200² is under 0.4 s with no tree needed.
- A grid over 2 chosen features of d-dimensional data is a **slice**: the other features are fixed. Say so in the UI.

### Recommendation
Stack three layers in one plot component:
1. **Bottom:** Canvas 2D for the heatmap and the d3-contour boundary lines.
2. **Middle:** Canvas 2D for training points (≤ ~5k).
3. **Top:** React SVG for the query, neighbour lines, unit ball, Voronoi cell of the hovered point and focus rings.

Use d3 modules for math only (scales, delaunay, contour), not for DOM. Use Observable Plot (lazy-loaded) or hand-rolled SVG for the small charts. Drop Plotly: 400–1,500 KB for things the canvas layer does better.

## 3. Math: KaTeX vs MathJax

| | KaTeX | MathJax |
|---|---|---|
| Version / licence | 0.19.1 (2026-10-23) / MIT | 4.1.3 / Apache-2.0 |
| Size (dist, gz) | `katex.min.js` 76.1 KB + `katex.min.css` 3.5 KB + 20 woff2 fonts (254 KB total; the browser fetches only the faces a formula uses) | `tex-chtml.js` 280.9 KB (includes newcm font data); `tex-chtml-nofont.js` 256.2 KB; `tex-svg.js` 615.2 KB |
| Speed model | "renders its math synchronously and doesn't need to reflow the page" (README) | v4 fonts are split into ranges loaded on demand, so rendering can be **asynchronous**. Use `typesetPromise()`. Preloading all ranges "can significantly slow initial page processing" (docs) |
| Coverage | "much (but not all) of LaTeX" | Fuller LaTeX and accessibility tooling |
| React | Call `katex.render(tex, el, opts)` in a `useLayoutEffect` (or `renderToString` + memo). **react-katex 3.1.0** (MIT) depends on `katex ^0.16.0`, so it would pin 0.16.x, and its peer range is React `<20`. | **better-react-mathjax 3.0.3** (MIT, 2.4 KB): loads MathJax **from a CDN (cdnjs) by default** unless `src` is set; documents MathJax 2 and 3, and the page doesn't mention v4. |
| Hover-linking terms | `\htmlClass`, `\htmlId`, `\htmlData`, `\htmlStyle` once `trust` is enabled and `strict` relaxed (https://katex.org/docs/supported) | `\class`/`\cssId` extensions (not reviewed here) |

Sources: https://cdn.jsdelivr.net/npm/katex@0.19.0/dist/README.md, https://docs.mathjax.org/en/v4.1/output/fonts.html, https://npmjs.com/package/better-react-mathjax

**Pick KaTeX**, used directly through a ~20-line `<Tex>` component. Enable `trust: (ctx) => ctx.command === '\\htmlData'` so only our own data attributes pass. The equations are ours, never user input. Bundle the KaTeX CSS and fonts. No CDN is involved, which keeps the "nothing leaves the browser" promise.

## 4. CSV: Papa Parse
- **Package:** `papaparse` 5.7.0 (2026-08-24), MIT, **6.9 KB** gz (dist; 7.3 KB esb). Types: `@types/papaparse` 5.5.2. Docs: https://www.papaparse.com/docs
- **Config to use:**
  - `header: true`. Duplicate names are renamed and listed in `meta.renamedHeaders`.
  - `transformHeader: h => h.trim()`.
  - `skipEmptyLines: 'greedy'`.
  - `dynamicTyping`: the function form ("should return a boolean value for each field number (or name if using headers)"), so the column-typing step can force ID-like or code-like columns to stay strings.
- **Missing values:** `dynamicTyping` does not treat `NA`, `?` or `null` as missing. Map a configurable NA-token list after parsing.
- **Sniffing:** use `preview: 500` to sniff types and show a preview before the full parse.
- **Large files:**
  - `Papa.parse(file, config)` reads a `File` via FileReader, with results delivered through callbacks.
  - Streaming uses `step` (per row) or `chunk` (per chunk, local or remote files only; don't combine the two). `LocalChunkSize` defaults to 10 MB.
  - With streaming, `complete` gets no results.
- **Worker mode:** `worker: true` keeps the page responsive "at some cost in speed", and `pause()`/`resume()` aren't available.
- **Recommendation:** parse inside **our own** Vite module worker (which we need for compute anyway), not `worker: true`. That gives one worker, one data copy and the parsed matrix already where the compute runs. Cap uploads at around 10–20k rows × 100 columns, because leave-one-out and k-sweep cost is O(n²·d).

## 5. Compute off the main thread
- **Vite** (https://vite.dev/guide/features#web-workers): `new Worker(new URL('./knn.worker.ts', import.meta.url), { type: 'module' })` is the **recommended** form. The `new URL()` must sit directly inside `new Worker()`, and options must be static literals. The alternative is `import KnnWorker from './knn.worker?worker'`. Workers are emitted as separate chunks, and `?worker&inline` gives base64.
- **Comlink** 4.4.2 (2024-11-07), Apache-2.0, **2.0 KB** gz (dist esm; README: "~1.2k gzip'd"). Source: https://raw.githubusercontent.com/GoogleChromeLabs/comlink/main/README.md
  - `expose(api)` in the worker, `wrap(worker)` in the UI, with `await` on every call.
  - Pass grids and matrices as `Float64Array`/`Uint8Array` wrapped in `Comlink.transfer(x, [x.buffer])` to avoid copies.
  - Use `Comlink.proxy(cb)` for progress callbacks (coarse grid ready, then fine grid ready).
- **Pattern:**
  - The worker holds the dataset (`Float64Array` n×d, row-major) and the preprocessing state.
  - API: `predictOne(q) → {distances, order, neighbours, votes}`, `grid(xr, yr, res) → Uint8Array classes + Float32Array confidence`, `sweepK(ks) → accuracy[]`.
  - Cancel stale grid jobs with a generation counter.

## 6. Projection for more than 2 features

| Lib | Version / date | Licence | min+gz | Notes |
|---|---|---|---|---|
| ml-pca | 4.1.1 (2022-11-11) | MIT | 18.7 KB (esb, includes ml-matrix) | `new PCA(X)`, `predict`, `getExplainedVariance()` (Iris PC1 ≈ 92%). Deterministic and fast. Can be inverse-mapped. |
| umap-js | 1.4.0 (2024-06-05) | MIT | 21.0 KB dist / 28.8 KB esb | `fit`, `fitAsync(X, epochCb)`, or `initializeFit` + `step()` (animatable). `nNeighbors` 15, `minDist` 0.1. `setSupervisedProjection(labels)` exists but **leaks labels into the layout**, so avoid it here. Random init. |
| tsne-js | 1.0.3 (**2016-02-22**) | Apache-2.0 | 26.1 KB dist | Unmaintained for 10 years, with ndarray deps. If t-SNE is needed, put it behind a lazy import. |

The npm README for umap-js is at https://npmjs.com/package/umap-js and the ml-pca README at https://cdn.jsdelivr.net/npm/ml-pca@4.1.1/README.md.

**Caveat to show in the UI:** KNN uses distances in the full d-dimensional (scaled, encoded) space, and a 2-D projection distorts them.
- **PCA** preserves global variance directions but squashes the dropped dimensions.
- **t-SNE and UMAP** preserve local neighbourhoods only roughly, and per Distill "you cannot see relative sizes of clusters" and "distances between well-separated clusters … may mean nothing" (https://distill.pub/2016/misread-tsne/).

**Design rules:**
- Draw the true d-D neighbours of the query on the projection, even when they look far away.
- Report neighbour preservation: the share of each point's true k-NN that are also its 2-D k-NN.
- Never draw a "decision boundary" on a t-SNE or UMAP plane. On a PCA plane, call it a "slice through the mean" (grid points inverse-projected, other components at 0), or skip it.
- Default to choosing two original features for the map. Offer PCA as a second, linked view (Setosa pattern).

## 7. Reference implementations for tests
- **scikit-learn 1.9.1** `KNeighborsClassifier` (docs: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html):
  - `weights` is `'uniform'`, `'distance'` (inverse distance) or a callable.
  - `metric='minkowski'` with `p` (p=1 is Manhattan, p=2 Euclidean). Use `algorithm='brute'` for exact fixtures.
  - `predict_proba` classes are in lexicographic order.
  - Docs note: if neighbours k and k+1 "have identical distances but different labels, the results will depend on the ordering of the training data".
- **Fixture plan:** a Python script, kept outside `src/`, writes `tests/fixtures/*.json` for each dataset in `references/datasets` × k ∈ {1,3,5,15} × p ∈ {1,2,3} plus `metric='chebyshev'` × weights ∈ {uniform, distance} × scaling ∈ {none, z-score, min-max}. Each fixture stores the preprocessing parameters (means, sds, one-hot categories), and for about 50 query rows: neighbour indices, distances, `predict_proba`, `predict`.
  - Vitest compares at 1e-9.
  - **Drop queries** where the k-th and (k+1)-th distances tie, or where class votes tie, unless testing our documented tie rule.
  - Test zero-distance queries (a query equal to a training row) separately with `weights='distance'`. The fetched sklearn page doesn't document that case, so pin our behaviour explicitly.
- **ml-knn** 3.0.0 (2019-06-29), MIT, 1.8 KB esb (https://unpkg.com/ml-knn@3.0.0/README.md): k-d tree, `distance` option (Euclidean default, custom function allowed), default `k` = number of labels + 1. No weighting is documented, so it's only useful as a second oracle for uniform votes.
- **Existing engine** (`src/lib/knnEngine.ts`), from reading the code; not run:
  - `generateDecisionBoundaryGrid` pushes `0` for any non-numeric label (`typeof result.predictedLabel === 'number' ? … : 0`), so string-labelled data paints one class.
  - Ties go to whichever label was inserted first into `votes`.
  - Distance weighting uses `1/(d + 1e-10)`.
  - Each query does a full sort.
  - Keep the distance functions. Rewrite the rest against the fixtures.

## 8. GitHub Pages deployment
- **`base`:** Vite says for `https://<USERNAME>.github.io/<REPO>/` "set `base` to `'/<REPO>/'`" (https://vite.dev/guide/static-deploy). The repo has `base: mode === 'production' ? '/knn-interactive/' : '/'`, but the repo is **knn-explorer**, so every asset 404s. Use `base: '/knn-explorer/'`, or read it from an env var so forks work.
- **Pages source:** Settings → Pages → Build and deployment → Source: **GitHub Actions** (https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).
- **Workflow:**
  - The repo's `deploy.yml` is the right shape (permissions `contents: read`, `pages: write`, `id-token: write`; concurrency group `pages`; build job, then `deploy-pages`).
  - Vite's current sample pins newer majors by SHA: `actions/checkout` v7, `actions/setup-node` v7, `actions/configure-pages` v6, `actions/upload-pages-artifact` v5, `actions/deploy-pages` v5. The repo uses checkout@v4, setup-node@v4 (node '20'), configure-pages@v4, upload-pages-artifact@v3 and deploy-pages@v4.
  - Vite 8 needs Node ≥ 20.19 or ≥ 22.12, so use `node-version: lts/*` or `22`.
- **SPA routing:**
  - **Preferred: no router.** The app is one page. Keep wizard and step state in the app, optionally mirrored to `location.hash` (`#k=5&p=2&ds=wine`) for shareable links. That needs no server support.
  - **If routes are needed: hash routing.** `createHashRouter` / `HashRouter` (react-router 8.5.0 docs import from `react-router`; `react-router-dom` latest is 7.18.4). https://reactrouter.com/api/data-routers/createHashRouter
  - **404.html trick** (https://cdn.jsdelivr.net/gh/rafgraph/spa-github-pages@gh-pages/README.md): `404.html` re-encodes the path into a query string, and `index.html` restores it with `history.replaceState`. Set `pathSegmentsToKeep = 1` for project pages. Caveats: fresh loads of deep links get a 404 response plus a redirect round trip, and Google stopped following the redirect around 2019. Not worth it here.
  - **Current bug:** `App.tsx` uses `<BrowserRouter>` with no `basename`, and its only route is `/`. Under `/knn-explorer/` that path doesn't match, so the app renders `NotFound`. That holds even after `base` is fixed.
- **Sample datasets:** put them in `public/datasets/*.csv` and load with `fetch(import.meta.env.BASE_URL + 'datasets/iris.csv')`, so the base prefix is applied.
- **Privacy:** no CDNs (KaTeX, fonts and MathJax all bundled), no analytics. Optionally add a `<meta http-equiv="Content-Security-Policy">` with `connect-src 'self'`. GitHub Pages can't set headers, and KaTeX needs `style-src 'unsafe-inline'`.

## 9. Recommended stack

| Concern | Choice | Version | Licence | min+gz |
|---|---|---|---|---|
| Build / UI | Vite + React + TypeScript | 8.3.5 / 19.3.1 / 7.0.2 (or keep 5.4 / 18.3 / 5.8 to start) | MIT / MIT / Apache-2.0 | 68.8 KB (React 19) |
| Styling / controls | Tailwind + about 9 shadcn/Radix components (button, card, badge, slider, select, switch, tabs, tooltip, alert) | as in repo | MIT | small, per component |
| Main plot | Canvas 2D (heatmap, points) + React SVG overlay (query, neighbour lines, unit ball) | — | — | 0 |
| Geometry math | d3-scale, d3-delaunay, d3-contour, d3-geo `geoPath`, d3-scale-chromatic | 4.0.2 / 6.0.4 / 4.0.2 / 3.1.1 / 3.1.0 | ISC | ≈ 20 KB (sum of separate esb builds) |
| Small charts | Hand-rolled SVG with d3-scale; Observable Plot lazy-loaded if needed | 0.6.17 | ISC | 0 / 132.9 KB lazy |
| Math | KaTeX direct (no react-katex), `trust` limited to `\htmlData` | 0.19.1 | MIT | 76.1 KB + 3.5 KB CSS + fonts on demand |
| CSV | Papa Parse, called inside our worker | 5.7.0 | MIT | 6.9 KB |
| Workers | Vite module worker + Comlink | 4.4.2 | Apache-2.0 | 2.0 KB |
| Projection | ml-pca (lazy); umap-js optional (lazy) | 4.1.1 / 1.4.0 | MIT | 18.7 / 28.8 KB |
| Tests | Vitest + scikit-learn JSON fixtures (+ ml-knn for uniform votes) | 5.0.5 / sklearn 1.9.1 / 3.0.0 | MIT / BSD-3 / MIT | dev only |
| Routing | None; state in `location.hash` | — | — | 0 |
| Deploy | GitHub Actions → Pages, `base: '/knn-explorer/'` | deploy-pages v5 | — | — |

Estimated first-load JS: about 175 KB gz (React 69 + KaTeX 76 + d3 bits 20 + Papa 7 + Comlink 2) plus app code and Radix. Plotly alone would add 400–1,476 KB.

## 10. What to delete from the Lovable template
Determined by grepping `src/` imports. Nothing was run.

**Why the build and deploy break today:**
- **`npm ci` fails:** `package-lock.json` is out of sync with `package.json`. The lock root lacks `mathjs`, `react-katex`, `@types/plotly.js`, `vitest`, `@testing-library/jest-dom`, `@testing-library/react` and `jsdom`.
- **Unresolvable imports:** `react-plotly.js` (in `ClassificationPlayground.tsx`, `RegressionPlayground.tsx`, `PreprocessingModule.tsx`) and `danfojs` (in `PreprocessingModule.tsx`) are imported but not declared. `plotly.js` is missing too. danfojs is 44.7 MB unpacked, so don't add it.
- **Deploy paths:** `base` is wrong (`/knn-interactive/`), and `BrowserRouter` has no basename (§8).

**Delete outright:**
- **Lockfiles:** `bun.lock` and `bun.lockb`. The workflow uses npm; regenerate `package-lock.json`.
- **Lovable leftovers:** the `lovable-tagger` devDependency and its use in `vite.config.ts`. The Lovable README. The Lovable meta and OG tags in `index.html`, which point at lovable.dev images. Rename `package.json` "name" (`vite_react_shadcn_ts`).
- **Unused dependencies, never imported by app code:** `mathjs` (206.5 KB gz if it were used), `@tanstack/react-query` (no server; `QueryClientProvider` only wraps), `react-router-dom` (one route), `recharts` (only `ui/chart.tsx`), `@types/plotly.js`.
- **Dependencies of unused shadcn components:** `react-hook-form`, `@hookform/resolvers`, `zod` (form); `date-fns`, `react-day-picker` (calendar); `embla-carousel-react` (carousel); `input-otp`; `cmdk` (command); `vaul` (drawer); `next-themes` (only `ui/sonner.tsx`); `sonner`.
- **Unused `src/components/ui/*`**, never imported outside `ui/` or only by other unused ui files: accordion, alert-dialog, aspect-ratio, avatar, breadcrumb, calendar, carousel, chart, checkbox, collapsible, command, context-menu, dialog, drawer, dropdown-menu, form, hover-card, input-otp, input, label, menubar, navigation-menu, pagination, popover, progress, radio-group, resizable, scroll-area, separator, sheet, sidebar, skeleton, table, textarea, toggle, toggle-group. Also `src/hooks/use-mobile.tsx`, which only `sidebar` uses.
- **Toasts:** `toast.tsx`, `toaster.tsx`, `use-toast.ts`, `sonner.tsx` and `src/hooks/use-toast.ts`. Toasters are mounted in `App.tsx` but `toast()` is never called.
- **Their Radix packages:** `@radix-ui/react-` accordion, alert-dialog, aspect-ratio, avatar, checkbox, collapsible, context-menu, dialog, dropdown-menu, hover-card, label, menubar, navigation-menu, popover, progress, radio-group, scroll-area, separator, toast, toggle, toggle-group.

**Keep:** button, card, badge, slider, select, switch, tabs, tooltip, alert, plus `@radix-ui/react-{slot,slider,select,switch,tabs,tooltip}`, `class-variance-authority`, `clsx`, `tailwind-merge`, `tailwindcss-animate` and `lucide-react`.

**Likely re-adds** once the design needs them: table (vote table), toggle-group (metric picker), resizable (workbench panes), progress (worker progress), input/label (file upload), dialog or popover (column-typing wizard), collapsible ("Show the math"). Add them back with the shadcn CLI only when used.

**Also consider:**
- Replace `react-katex` (pins katex 0.16) with direct `katex` 0.19.1.
- Turn on `strict: true` in `tsconfig.app.json`, which is currently `false` with `noImplicitAny: false`.
