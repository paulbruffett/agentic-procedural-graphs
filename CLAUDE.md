# procedural-graphs

Reference implementation of "Procedural Graphs: Self-Evolving Execution Structures for LLM Agents" (arXiv 2609.09153) on LangGraph.

**Read `PLAN.md` first.** It holds the approved design, the paper facts being mirrored, the module layout, the implementation order, the verification steps, and a "Build status" section listing which files already exist and two deliberate deviations from the design text. Follow it; do not re-plan.

Conventions:
- `uv` for everything: `uv sync`, `uv run pytest`, `uv run pg ...`. Python is pinned `<3.14` (LangGraph 1.2 limit).
- All LLM calls go through OpenRouter via `pg.llm.make_llm`. Never add another provider client.
- Scenario-specific code lives only under `src/pg/envs/<name>/` behind the `Environment` / `Episode` interface in `src/pg/envs/base.py`. The core (`graph`, `guidance`, `agent`, `refiner`, `evolve`, `evaluate`) must stay task-agnostic.
- Unit tests must not hit the network. Anything that calls an LLM is a smoke test run by hand via the CLI.
- Keep it simple and readable; this is a reference implementation, not a framework.
- `analysis/` holds what is not needed to run the framework (run-inspection tools, walkthroughs, future example notebooks). It may import `pg`; nothing under `src/pg/` may import from it.
- This repository is **public**. `third_party`, `data`, `runs`, `graphs/evolved` and `private` are gitignored and, on the maintainer's machine, are symlinks into `private/` (a clone of the private companion repo, set up by `sh private/link.sh`). Never `git add -f` them or add that repo as a remote here: CFO-Env is unlicensed and must not be published. Commit new runs / evolved graphs from inside `private/`.
