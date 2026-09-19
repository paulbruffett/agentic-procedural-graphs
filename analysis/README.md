# analysis/

Things that are **not needed to run the framework**: tools for looking at what a run produced, walkthroughs, and
(later) example notebooks. The rule for this folder: it may import `pg`, but nothing under `src/pg/` may import
from here. Deleting `analysis/` must leave `pg evolve` / `pg eval` fully working.

| File | What it is |
|---|---|
| [`paired_comparison.py`](paired_comparison.py) | Is a graph really better? Paired statistics over one or more `pg eval` run directories (bootstrap CI over tasks, exact McNemar on success) |
| [`graph_evolution.py`](graph_evolution.py) | How a graph changed over an evolve run: text diff, Mermaid per round, interactive timeline |
| [`timeline_template.html`](timeline_template.html) | The timeline page `graph_evolution.py report` fills in (self-contained apart from the dagre layout library, loaded from a CDN) |
| [`walkthrough.md`](walkthrough.md) | "Procedural Graphs Deep Dive": the concepts, then the codebase, following one real question through it |

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

- **`timeline.html`** - step through rounds (buttons or arrow keys). Each round shows the refiner's *proposal*
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
