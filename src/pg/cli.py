"""`pg` command line: gen-data | evolve | eval | analyze."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from pg.analyze import compare, format_report, load
from pg.config import Config
from pg.envs import make_env
from pg.envs.base import Environment
from pg.evaluate import evaluate, format_table
from pg.evolve import evolve
from pg.graph import ProceduralGraph
from pg.tracking import log_table, start_run

ENVS = ["finance", "hotpotqa", "enterprisearena"]


def gen_data(args: argparse.Namespace, cfg: Config) -> None:
    out = cfg.data_dir / args.env

    def given(*keys: str) -> dict:
        return {k: getattr(args, k) for k in keys if getattr(args, k) is not None}

    if args.env == "hotpotqa":
        from pg.envs.hotpotqa.data import generate

        counts = generate(out, args.seed, **given("n"))
    else:
        if args.env == "finance":
            from pg.envs.finance.tasks import generate
        else:
            from pg.envs.enterprisearena.data import generate
        counts = generate(out, args.seed, **given("n_train", "n_val", "n_test"))
    print(f"wrote {counts} to {out}")


def run_evolve(args: argparse.Namespace, cfg: Config, env: Environment, run) -> None:
    init = (ProceduralGraph.skeleton(f"{env.name}_evolved", env.tool_descriptions())
            if args.init == "scratch" else ProceduralGraph.load(args.init))
    out = args.out or cfg.graphs_dir / "evolved" / env.name
    best = evolve(env, init, cfg, args.rounds, args.batch, args.val_n, out, run, args.overwrite)
    print(f"best graph ({best.summary()}) saved to {out / 'best.json'}")


def run_eval(args: argparse.Namespace, cfg: Config, env: Environment, run, stamp: str) -> None:
    specs = [s.strip() for s in args.graphs.split(",") if s.strip()]
    out = cfg.runs_dir / f"{stamp}-eval-{env.name}"
    rows = evaluate(env, args.split, args.n, specs, cfg, out, run)
    print(format_table(rows))
    print(f"trajectories and metrics in {out}")


def run_analyze(args: argparse.Namespace, cfg: Config, stamp: str) -> None:
    metrics = ["success", "score"] + [m.strip() for m in args.metrics.split(",") if m.strip()]
    rows = compare(load(args.run_dirs), args.baseline, metrics)
    print(format_report(rows))
    if args.experiment:  # analysis is cheap and re-run often, so it is only logged when it belongs to an experiment
        with start_run(cfg, "analyze", f"analyze-{stamp}", vars(args)) as run:
            if run:
                log_table(run, "paired", rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="pg", description="Procedural Graphs reference implementation")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-data", help="generate / download task splits into data/<env>/")
    g.add_argument("env", choices=ENVS)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--n", type=int, default=None, help="hotpotqa: total examples (default 300)")
    g.add_argument("--n-train", type=int, default=None, help="finance (default 30) / enterprisearena (default 20)")
    g.add_argument("--n-val", type=int, default=None, help="finance (default 20) / enterprisearena (default 20)")
    g.add_argument("--n-test", type=int, default=None, help="finance (default 30) / enterprisearena (default 20)")

    e = sub.add_parser("evolve", help="self-evolve a graph with the refiner + validation gate")
    e.add_argument("env", choices=ENVS)
    e.add_argument("--init", default="scratch", help="scratch or a graph JSON path")
    e.add_argument("--rounds", type=int, default=5)
    e.add_argument("--batch", type=int, default=10)
    e.add_argument("--val-n", type=int, default=None, help="default: whole val split")
    e.add_argument("--out", type=Path, default=None, help="default: graphs/evolved/<env>/")
    e.add_argument("--overwrite", action="store_true", help="replace an earlier run's results in the output directory")

    v = sub.add_parser("eval", help="run a split with none, one or several graphs")
    v.add_argument("env", choices=ENVS)
    v.add_argument("--split", default="test")
    v.add_argument("--n", type=int, default=None, help="default: whole split")
    v.add_argument("--graphs", default="none", help="comma-separated: none,graphs/x.json,...")

    a = sub.add_parser("analyze", help="paired comparison of the graphs in one or more eval run directories")
    a.add_argument("run_dirs", type=Path, nargs="+", help="runs/<stamp>-eval-<env> directories; several = repeats")
    a.add_argument("--baseline", default="none", help="graph spec every other graph is compared against")
    a.add_argument("--metrics", default="", help="extra episode metrics, comma-separated (success and score are always reported)")

    for sp in (e, v):
        sp.add_argument("--concurrency", type=int, default=None)
    for sp in (e, v, a):
        sp.add_argument("--experiment", default=None, help="W&B group, to keep an evolve run, its evals and analysis together")

    args = p.parse_args(argv)
    cfg = Config.from_env()
    if getattr(args, "concurrency", None):
        cfg.concurrency = args.concurrency

    if args.cmd == "gen-data":
        return gen_data(args, cfg)

    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    if args.cmd == "analyze":
        return run_analyze(args, cfg, stamp)

    env = make_env(args.env, cfg)
    with start_run(cfg, args.cmd, f"{args.cmd}-{env.name}-{stamp}", vars(args)) as run:
        if args.cmd == "evolve":
            run_evolve(args, cfg, env, run)
        else:
            run_eval(args, cfg, env, run, stamp)


if __name__ == "__main__":
    main()
