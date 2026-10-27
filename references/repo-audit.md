# Audit of the Lovable version (commit 6ec2491)

Done by reading the code on 2026-10-27. Nothing was installed or run.

## It doesn't build

- `react-plotly.js` and `plotly.js` are imported by three components but not declared in `package.json`. `danfojs` is imported (and never used) in `PreprocessingModule.tsx`.
- `package-lock.json` is out of sync with `package.json`, so `npm ci` in the deploy workflow fails before the build starts.

## It wouldn't deploy correctly even if it built

- `vite.config.ts` sets `base: '/knn-interactive/'`. The repo is `knn-explorer`, so every asset 404s on GitHub Pages.
- `BrowserRouter` has no `basename`, so `/knn-explorer/` would render the NotFound page.
- `index.html` still has the "Lovable App" title and lovable.dev OG images. The header's GitHub button links to `https://github.com`.

## The KNN logic

| Area | What's there | Problem |
|---|---|---|
| Data input | CSV upload only in the Scaling tab. `split(',')`, first two columns become features, third becomes the label. | No header detection, no quoted fields, non-numeric rows silently dropped, no categorical handling, no missing-value handling, at most 2 features. |
| Classes | Plots filter `label === 0` and `label === 1`. | Multiclass and string labels don't work. The boundary grid maps every string label to 0. |
| Dimensions | Everything is 2-D. | No path for real tabular data with more than 2 features. |
| Accuracy | `calculateAccuracy(data, data, …)`. The comment calls it leave-one-out. | It's training accuracy with each point as its own neighbour, so k = 1 always shows 100%. |
| Ties | The vote winner is whichever key `Object.entries` returns first. | Undefined, and not shown to the user. |
| Minkowski p | Slider from 0.5 to 5. | p < 1 isn't a metric. Nothing tells the user. |
| Performance | The boundary grid is computed synchronously inside `useMemo`, which also calls `setState`. | Freezes the UI on larger data, and calling setState inside useMemo is a React anti-pattern. |
| Teaching | Theory tab is static text with two fixed points. | No step-by-step query walkthrough, no distance table, no neighbour highlighting, no vote breakdown, no link between formula terms and the picture. |

## Codebase weight

- 38 of the 49 files in `src/components/ui/` are never imported by app code. Two more (the toasters) are mounted but never fire. Over 30 dependencies exist only for them or are never imported at all (forms, calendar, carousel, toasts, react-query, recharts, mathjs, 21 Radix packages). The full list is in `visual-exemplars/tech-stack.md` §10.
- Two lockfiles (`bun.lock(b)` and `package-lock.json`) while the workflow uses npm.
- Tests: one placeholder (`expect(true).toBe(true)`).
- `tsconfig.app.json` has `strict: false`.

## What's worth keeping

- The distance functions and min-max/z-score helpers in `src/lib/knnEngine.ts` are correct. They're a reasonable starting point, but they'll need tie handling, typed arrays and worker support.
- The GitHub Actions Pages workflow is the right shape. It only needs the lockfile and `base` fixed.
- Tailwind setup and the class colour tokens.
