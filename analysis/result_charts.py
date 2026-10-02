"""Static charts of an EnterpriseArena eval run (and its evolve log) for reports and articles.

    uv run --with matplotlib python -m analysis.result_charts runs/<stamp>-eval-enterprisearena \\
        [--evolve graphs/evolved/<run>] [--out analysis/img/<name>]

Writes: metrics.png (2x2 small multiples: survived, months, tool calls and cost per month survived, per graph), survival.png (fraction of companies solvent by month, per graph), and, with --evolve, evolution.png
(validation score and graph size by round). Graph specs are labelled by their file name; `none` stays `none`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker

from pg.evaluate import label_of
from pg.trajectory import read_jsonl

matplotlib.use("Agg")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order (validated)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 200})


def short(spec: str) -> str:
    if spec == "none":
        return "none"
    name = Path(spec).stem
    return {"round_0": "skeleton", "best": "evolved"}.get(name, name.replace("enterprisearena_", "").replace("_", " "))


ORDER = ["none", "skeleton", "evolved", "expert"]  # least to most procedural knowledge; anything else goes after, as run


def load(run_dir: Path, exclude: set[str] = frozenset()) -> dict[str, list]:
    """Trajectories per graph, ordered none -> skeleton -> evolved -> expert, errored episodes dropped.
    `exclude` names graphs (by their short label) to leave out."""
    specs = [r["graph"] for r in json.loads((run_dir / "metrics.json").read_text()) if short(r["graph"]) not in exclude]
    specs.sort(key=lambda s: ORDER.index(short(s)) if short(s) in ORDER else len(ORDER))
    return {s: [t for t in read_jsonl(run_dir / f"{label_of(s)}.jsonl") if not t.error] for s in specs}


def cost(t) -> float:
    return sum(v for k, v in t.usage.items() if k.endswith("cost"))


def metrics_panel(data: dict, out: Path) -> None:
    names = [short(s) for s in data]
    panels = [("survived", "companies survived", lambda ts: sum(t.success for t in ts), "{:.0f}"),
              ("months", "mean months survived", lambda ts: mean(t.metrics["months_survived"] for t in ts), "{:.0f}"),
              ("steps", "tool calls per month survived", lambda ts: sum(len(t.steps) for t in ts) / sum(t.metrics["months_survived"] for t in ts), "{:.1f}"),
              ("cost", "cost per month survived (cents)", lambda ts: 100 * sum(cost(t) for t in ts) / sum(t.metrics["months_survived"] for t in ts), "{:.2f}")]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6))
    for ax, (_, title, f, fmt) in zip(axes.flat, panels):
        vals = [f(ts) for ts in data.values()]
        bars = ax.bar(names, vals, color=SERIES[: len(names)], width=0.62)
        for b, v, ts in zip(bars, vals, data.values()):
            label = fmt.format(v) + (f"/{len(ts)}" if title.startswith("companies") else "")
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), label, ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_ylim(0, max(vals) * 1.18)
        if title.startswith("companies"):
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.tick_params(axis="x", length=0, labelsize=9); ax.tick_params(axis="y", length=0, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    fig.suptitle("Test split: 20 unseen seeds, 132 months, horizon hidden", x=0.02, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0); fig.savefig(out / "metrics.png"); plt.close(fig)


def survival(data: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    horizon = max(t.metrics["horizon"] for ts in data.values() for t in ts)
    for (spec, ts), color in zip(data.items(), SERIES):
        months = [t.metrics["months_survived"] for t in ts]
        xs = list(range(horizon + 1)); ys = [sum(m >= x for m in months) / len(months) for x in xs]
        ax.step(xs, ys, where="post", color=color, linewidth=2)
        ax.text(horizon + 1.5, ys[-1],
                f"{short(spec)}  {ys[-1]:.0%}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, horizon + 22); ax.set_xticks(range(0, horizon + 1, 20)); ax.set_ylim(0, 1.04); ax.set_xlabel("month"); ax.set_ylabel("companies still solvent")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1]); ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_title(f"Survival by month, {max(len(ts) for ts in data.values())} test seeds per graph", loc="left", fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(out / "survival.png"); plt.close(fig)


def evolution(evolve_dir: Path, out: Path) -> None:
    log = [json.loads(l) for l in (evolve_dir / "evolution_log.jsonl").read_text().splitlines() if l.strip()]
    kept, cand, edges, rounds = [], [], [], []
    for e in log:
        rounds.append(e["round"]); edges.append(int(e["graph"].split(",")[1].split()[0]))
        kept.append(e.get("val_score", kept[-1] if kept else 0.0) if e["round"] == 0 else (e["val_after"] if e["accepted"] else kept[-1]))
        cand.append(e.get("val_after"))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    ax.plot(rounds, kept, color=SERIES[2], linewidth=2, zorder=2)
    for k, v, c, e in zip(rounds, kept, cand, log):
        if k == 0:
            ax.plot(k, v, "o", color=SERIES[2], markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        elif c is None:
            ax.plot(k, v, "o", color=SURFACE, markersize=8, markeredgecolor=MUTED, markeredgewidth=1.5, zorder=3)
            ax.text(k, v + 0.03, "no run", ha="center", fontsize=8, color=MUTED)
        elif e["accepted"]:
            ax.plot(k, v, "o", color=SERIES[2], markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        else:
            ax.plot(k, c, "o", color=SURFACE, markersize=8, markeredgecolor=SERIES[1], markeredgewidth=1.8, zorder=3)
            ax.text(k, c - 0.045, f"rejected\n{c:.3f}", ha="center", va="top", fontsize=8, color=MUTED)
        if k == 0 or (c is not None and e["accepted"]):
            ax.text(k, v + 0.03, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_ylim(0.5, 1.02); ax.set_ylabel("validation score\n(fraction of months survived)")
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_title(f"Evolution: {evolve_dir.name}, 20 validation seeds, 66 months", loc="left", fontsize=10, color=INK)
    ax2.bar(rounds, edges, color=SERIES[0], width=0.55)
    for k, n in zip(rounds, edges):
        ax2.text(k, n + 0.3, str(n), ha="center", va="bottom", fontsize=8, color=INK)
    ax2.set_ylabel("edges"); ax2.set_xlabel("round"); ax2.set_xticks(rounds); ax2.set_ylim(0, max(edges) * 1.35)
    ax2.tick_params(axis="x", length=0); ax2.grid(axis="y", color=GRID, linewidth=0.8); ax2.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(out / "evolution.png"); plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--evolve", type=Path, default=None, help="evolve run directory, for evolution.png")
    p.add_argument("--out", type=Path, default=None, help="default: <run_dir>/charts")
    p.add_argument("--exclude", default="", help="graphs to leave out, by short label, comma-separated (e.g. expert)")
    a = p.parse_args(argv)
    out = a.out or a.run_dir / "charts"; out.mkdir(parents=True, exist_ok=True)
    data = load(a.run_dir, {x.strip() for x in a.exclude.split(",") if x.strip()})
    metrics_panel(data, out); survival(data, out)
    if a.evolve:
        evolution(a.evolve, out)
    print("wrote", ", ".join(sorted(p.name for p in out.glob("*.png"))), "to", out)


if __name__ == "__main__":
    main()
