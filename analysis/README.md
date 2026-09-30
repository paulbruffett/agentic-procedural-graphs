# analysis/

Things that are **not needed to run the framework**: tools for looking at what a run produced, walkthroughs, and
(later) example notebooks. The rule for this folder: it may import `pg`, but nothing under `src/pg/` may import
from here. Deleting `analysis/` must leave `pg evolve` / `pg eval` fully working.

| File | What it is |
|---|---|
| [`paired_comparison.py`](paired_comparison.py) | Is a graph really better? Paired statistics over one or more `pg eval` run directories (bootstrap CI over tasks, exact McNemar on success) |
| [`result_charts.py`](result_charts.py) | Static charts of an eval run for reports: small-multiples bars (survived, months, steps, cost), per-seed slope chart, survival-by-month curves, and the evolution curve from an evolve log. `uv run --with matplotlib python -m analysis.result_charts runs/<stamp>-eval-<env> --evolve graphs/evolved/<run>` |
| [`graph_evolution.py`](graph_evolution.py) | How a graph changed over an evolve run: text diff, Mermaid per round, interactive timeline; also `graph_to_svg` / `graph_to_mermaid` for drawing any graph |
| [`timeline_template.html`](timeline_template.html) | The timeline page `graph_evolution.py report` fills in (self-contained apart from the dagre layout library, loaded from a CDN) |
| [`walkthrough.ipynb`](walkthrough.ipynb) | **Start here.** The walkthrough as an executable notebook: the conceptual map, then every mechanism exercised by a cell on HotpotQA, ending with a real three-round evolution and its timeline. Saved with outputs. Sections 1–6, 8, 10–11 and 13 need no API key; the live sections cost about $1.50 in total (the evolve cell reuses its run on re-open). Graph diagrams are SVGs made with Graphviz: `brew install graphviz` (the `dot` binary; the Python package is a dev dependency); without it the notebook falls back to Mermaid blocks. `uv run --with jupyterlab jupyter lab` from the repo root |
| [`img/`](img/) | Figures: `loops.svg` (the online and offline loops, embedded in the notebook); `ea-run3/` (article stills of the EnterpriseArena evolution: `round_0/1/3.png|svg`, top-to-bottom layout with that round's changes coloured, `timeline.gif`, the interactive timeline stepping through all six rounds, and `charts/`, the four result charts) |
| [`walkthrough.md`](walkthrough.md) | What the notebook cannot run cheaply: the EnterpriseArena scenario and the results, including the from-scratch evolution that took test survival from 1/20 to 17/19 |

## Watching a graph evolve

The cheapest way to build a graph and watch it change is HotpotQA: five rounds take under half an hour and cost
about $1.50 (EnterpriseArena is a day and $40-60).

```bash
uv run pg gen-data hotpotqa --seed 0 --n 300
uv run pg evolve hotpotqa --init scratch --rounds 5 --batch 10 --out graphs/evolved/hotpot-demo
uv run python -m analysis.graph_evolution report graphs/evolved/hotpot-demo
open graphs/evolved/hotpot-demo/timeline.html        # and evolution.md renders on GitHub / in any Mermaid viewer
```

`report` works on any evolve run directory, including ones made before this tool existed and ones that stopped
half-way, because it only reads what `evolve` saves as it goes: `round_k.json`, `evolution_log.jsonl` and
`edits/round_k.json`. Runs from before `edits/` was recorded cannot show what a *rejected* round proposed; the
timeline says so on those rounds.

- **`timeline.html`** - step through rounds (buttons or arrow keys); `#round=k` opens on a round, `#theme=light` forces the light palette, `#fit` scales the graph to the panel (the three combine, e.g. for screenshots). Each round shows the refiner's *proposal*
  against the graph before it: green = added, amber = revised, red dashed = removed, with the verdict, the
  validation curve (hollow = rejected) and the refiner's rationale. Click an edge for its condition / guidance /
  pitfalls and, if it was revised, the text it replaced. Node positions are fixed across rounds.
- **`evolution.md`** - the same story as a document: a summary table, then per round a Mermaid diagram and the
  text diff.
- **`diff a.json b.json`** - changes between any two graph files, e.g. a hand-edited graph against its original.

During a run, `evolve` also logs an `edge_changes` table to W&B (one row per edge a round added, revised or
removed, with its text and whether the round was accepted) next to the `graph/nodes` and `graph/edges` curves.

## Comparing graphs

```bash
uv run python -m analysis.paired_comparison runs/<stamp>-eval-<env> [more run dirs = repeats] \
    --baseline none --metrics months_survived,valuation_score,steps [--experiment <W&B group>]
```

See "Measuring whether a graph helps" in the main README for the protocol this belongs to.
