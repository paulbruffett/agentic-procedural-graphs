"""`pg` command line: gen-data | run | evolve | eval."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from pg.config import Config
from pg.envs import make_env
from pg.evaluate import evaluate, format_table
from pg.evolve import evolve
from pg.graph import ProceduralGraph
from pg.tracking import start_run

ENVS = ["finance", "hotpotqa", "enterprisearena"]


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

    r = sub.add_parser("run", help="run a split with or without one graph")
    r.add_argument("env", choices=ENVS)
    r.add_argument("--split", default="test")
    r.add_argument("--n", type=int, default=10)
    r.add_argument("--graph", default="none", help="path to a graph JSON, or none")

    e = sub.add_parser("evolve", help="self-evolve a graph with the refiner + validation gate")
    e.add_argument("env", choices=ENVS)
    e.add_argument("--init", default="scratch", help="scratch or a graph JSON path")
    e.add_argument("--rounds", type=int, default=5)
    e.add_argument("--batch", type=int, default=10)
    e.add_argument("--val-n", type=int, default=None, help="default: whole val split")
    e.add_argument("--out", type=Path, default=None, help="default: graphs/evolved/<env>/")

    v = sub.add_parser("eval", help="compare several graphs on a split")
    v.add_argument("env", choices=ENVS)
    v.add_argument("--split", default="test")
    v.add_argument("--n", type=int, default=None, help="default: whole split")
    v.add_argument("--graphs", default="none", help="comma-separated: none,graphs/x.json,...")

    for sp in (r, e, v):
        sp.add_argument("--concurrency", type=int, default=None)

    args = p.parse_args(argv)
    cfg = Config.from_env()
    if getattr(args, "concurrency", None):
        cfg.concurrency = args.concurrency

    if args.cmd == "gen-data":
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
        return

    env = make_env(args.env, cfg)
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"

    with start_run(cfg, args.cmd, f"{args.cmd}-{env.name}-{stamp}", vars(args)) as run:
        if args.cmd == "evolve":
            if args.init == "scratch":
                init = ProceduralGraph.skeleton(f"{env.name}_evolved", env.tool_descriptions())
            else:
                init = ProceduralGraph.load(args.init)
            out = args.out or cfg.graphs_dir / "evolved" / env.name
            best = evolve(env, init, cfg, args.rounds, args.batch, args.val_n, out, run)
            print(f"best graph ({best.summary()}) saved to {out / 'best.json'}")
            return

        specs = [args.graph] if args.cmd == "run" else [s.strip() for s in args.graphs.split(",") if s.strip()]
        out = cfg.runs_dir / f"{stamp}-{args.cmd}-{env.name}"
        rows = evaluate(env, args.split, args.n, specs, cfg, out, run)
        print(format_table(rows))
        print(f"trajectories and metrics in {out}")


if __name__ == "__main__":
    main()
